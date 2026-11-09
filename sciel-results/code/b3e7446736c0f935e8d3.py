"""EV routing with time windows and battery; clustering + construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cust = list(range(1, n_customers + 1))

    def check_route(route):
        if not route:
            return True, 0.0, 0.0
        total_demand = demands[route].sum()
        if total_demand > capacity:
            return False, 0.0, 0.0
        t = 0.0
        bat = float(battery)
        d = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def route_distance(route):
        if not route:
            return 0.0
        d = 0.0
        prev = 0
        for v in route:
            d += dist[prev, v]
            prev = v
        d += dist[prev, 0]
        return d

    def kmeans_cluster(points, k, n_iter=30, seed=42):
        """Simple k-means on 2D points. Returns cluster labels."""
        rng = np.random.RandomState(seed)
        n = len(points)
        if k >= n:
            return np.arange(n)
        # Initialize with k-means++
        centers = np.empty((k, 2))
        centers[0] = points[rng.randint(n)]
        for c in range(1, k):
            d2 = np.min(((points[:, None, :] - centers[None, :c, :]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum()
            idx = rng.choice(n, p=probs)
            centers[c] = points[idx]
        labels = np.zeros(n, dtype=int)
        for _ in range(n_iter):
            d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = points[mask].mean(axis=0)
        return labels

    # Determine number of routes based on capacity
    total_demand = demands[cust].sum()
    min_routes = max(1, int(np.ceil(total_demand / capacity)))
    # Try a few cluster counts
    best_routes = None
    best_dist = float('inf')

    for n_clusters in range(min_routes, min_routes + 4):
        if time.time() > deadline:
            break
        # Cluster customers
        cust_coords = coords[cust]
        labels = kmeans_cluster(cust_coords, n_clusters)
        
        # Group customers by cluster
        clusters = []
        for c in range(n_clusters):
            members = [cust[i] for i in range(n_customers) if labels[i] == c]
            if members:
                clusters.append(members)
        
        # If too many clusters, merge smallest ones
        while len(clusters) > n_clusters:
            clusters.sort(key=len)
            if len(clusters) < 2:
                break
            small = clusters.pop(0)
            # Merge into nearest cluster
            best_ci = 0
            best_d = float('inf')
            for ci, cl in enumerate(clusters):
                if not cl:
                    continue
                # Distance between centroids
                c1 = coords[small].mean(axis=0)
                c2 = coords[cl].mean(axis=0)
                d = np.sqrt(((c1 - c2) ** 2).sum())
                if d < best_d:
                    best_d = d
                    best_ci = ci
            clusters[best_ci].extend(small)
        
        # Build route for each cluster using nearest neighbor + insertion
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            # Start with nearest customer to depot
            route = []
            remaining = list(cluster)
            
            # Pick starting customer: nearest to depot
            start = min(remaining, key=lambda v: dist[0, v])
            route.append(start)
            remaining.remove(start)
            
            # Nearest neighbor insertion
            while remaining:
                best_v = None
                best_pos = None
                best_cost = float('inf')
                for v in remaining:
                    for pos in range(len(route) + 1):
                        newroute = route[:pos] + [v] + route[pos:]
                        ok, d, _ = check_route(newroute)
                        if not ok:
                            continue
                        # Additional cost
                        if pos == 0:
                            add = dist[0, v] + dist[v, route[0]] - dist[0, route[0]]
                        elif pos == len(route):
                            add = dist[route[-1], v] + dist[v, 0] - dist[route[-1], 0]
                        else:
                            add = dist[route[pos-1], v] + dist[v, route[pos]] - dist[route[pos-1], route[pos]]
                        if add < best_cost:
                            best_cost = add
                            best_v = v
                            best_pos = pos
                if best_v is None:
                    # Try just appending
                    for v in remaining:
                        newroute = route + [v]
                        ok, _, _ = check_route(newroute)
                        if ok:
                            best_v = v
                            best_pos = len(route)
                            break
                    if best_v is None:
                        break
                route = route[:best_pos] + [best_v] + route[best_pos:]
                remaining.remove(best_v)
            
            if route:
                routes.append(route)
        
        # Check all customers covered
        all_served = set()
        for r in routes:
            all_served.update(r)
        if len(all_served) != n_customers:
            # Assign remaining customers to nearest route
            for v in cust:
                if v not in all_served:
                    best_ri = None
                    best_d = float('inf')
                    for ri, r in enumerate(routes):
                        if not r:
                            continue
                        d = min(dist[v, u] for u in r)
                        if d < best_d:
                            best_d = d
                            best_ri = ri
                    if best_ri is not None:
                        # Insert into best position
                        best_pos = 0
                        best_add = float('inf')
                        r = routes[best_ri]
                        for pos in range(len(r) + 1):
                            newr = r[:pos] + [v] + r[pos:]
                            ok, _, _ = check_route(newr)
                            if not ok:
                                continue
                            if pos == 0:
                                add = dist[0, v] + dist[v, r[0]] - dist[0, r[0]]
                            elif pos == len(r):
                                add = dist[r[-1], v] + dist[v, 0] - dist[r[-1], 0]
                            else:
                                add = dist[r[pos-1], v] + dist[v, r[pos]] - dist[r[pos-1], r[pos]]
                            if add < best_add:
                                best_add = add
                                best_pos = pos
                        if best_pos is not None:
                            routes[best_ri] = routes[best_ri][:best_pos] + [v] + routes[best_ri][best_pos:]
                            all_served.add(v)
                        else:
                            routes.append([v])
                            all_served.add(v)
                    else:
                        routes.append([v])
                        all_served.add(v)
        
        if len(all_served) != n_customers:
            continue
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 2-opt intra route
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                ri = routes[i]
                n = len(ri)
                if n < 2:
                    continue
                found = False
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_ri = ri[:a] + ri[a:b+1][::-1] + ri[b+1:]
                        ok, _, _ = check_route(new_ri)
                        if not ok:
                            continue
                        d = route_distance(new_ri)
                        d0 = route_distance(ri)
                        if d < d0 - 1e-9:
                            routes[i] = new_ri
                            ri = new_ri
                            n = len(ri)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
            
            # Relocate between routes
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    v = routes[i][j]
                    src = routes[i]
                    src_without = src[:j] + src[j+1:]
                    best_gain = 1e-9
                    best_target = None
                    best_pos = None
                    best_src_without = None
                    best_new_tgt = None
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        tgt = routes[k]
                        for pos in range(len(tgt) + 1):
                            new_tgt = tgt[:pos] + [v] + tgt[pos:]
                            ok_t, _, _ = check_route(new_tgt)
                            if not ok_t:
                                continue
                            ok_s, _, _ = check_route(src_without)
                            if not ok_s:
                                continue
                            d_t = route_distance(new_tgt)
                            d_s = route_distance(src_without)
                            d0_t = route_distance(tgt)
                            d0_s = route_distance(src)
                            gain = (d_t + d_s) - (d0_t + d0_s)
                            if -gain > best_gain:
                                best_gain = -gain
                                best_target = k
                                best_pos = pos
                                best_src_without = src_without
                                best_new_tgt = new_tgt
                    if best_target is not None:
                        routes[i] = best_src_without
                        routes[best_target] = best_new_tgt
                        if not routes[i]:
                            routes.pop(i)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Swap
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for k in range(i + 1, len(routes)):
                    ri = routes[i]
                    rk = routes[k]
                    for a in range(len(ri)):
                        for b in range(len(rk)):
                            vi = ri[a]
                            vk = rk[b]
                            new_ri = list(ri)
                            new_rk = list(rk)
                            new_ri[a] = vk
                            new_rk[b] = vi
                            ok_i, _, _ = check_route(new_ri)
                            ok_k, _, _ = check_route(new_rk)
                            if not (ok_i and ok_k):
                                continue
                            d_i = route_distance(new_ri)
                            d_k = route_distance(new_rk)
                            d0_i = route_distance(ri)
                            d0_k = route_distance(rk)
                            if d_i + d_k < d0_i + d0_k - 1e-9:
                                routes[i] = new_ri
                                routes[k] = new_rk
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Evaluate
        total = sum(route_distance(r) for r in routes)
        if total < best_dist:
            best_dist = total
            best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        # Fallback: single route with all customers
        best_routes = [cust]
    
    return best_routes
# EVOLVE-BLOCK-END
