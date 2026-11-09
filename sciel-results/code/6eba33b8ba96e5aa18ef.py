"""EV routing with time windows and battery; clustered construction + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import sqrt, inf

def _euclid(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def _get_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
    return dist

def _route_distance_fast(route, dist):
    """Compute total distance of a route (customer sequence, depot implicit)."""
    if not route:
        return 0.0
    d = dist[0][route[0]]
    for i in range(len(route) - 1):
        d += dist[route[i]][route[i+1]]
    d += dist[route[-1]][0]
    return d

def _check_route_feasibility_fast(cust_seq, dist, demands, ready, due, service, 
                                   capacity, battery_cap, speed, recharge_rate, horizon, stations):
    """Check if a customer sequence is feasible. Returns (ok, total_dist, full_route)."""
    if not cust_seq:
        return True, 0.0, []
    
    total_demand = sum(demands[v] for v in cust_seq)
    if total_demand > capacity:
        return False, 0.0, []
    
    t = 0.0
    bat = battery_cap
    d_total = 0.0
    full_route = []
    prev = 0
    
    for v in cust_seq:
        dd = dist[prev][v]
        if bat >= dd - 1e-9:
            t += dd / speed
            bat -= dd
            d_total += dd
        else:
            best_s = -1
            best_cost = inf
            for s in stations:
                d1 = dist[prev][s]
                if d1 > bat + 1e-9:
                    continue
                d2 = dist[s][v]
                if d2 > battery_cap + 1e-9:
                    continue
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            if best_s == -1:
                return False, 0.0, []
            d1 = dist[prev][best_s]
            t += d1 / speed
            bat -= d1
            d_total += d1
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            full_route.append(best_s)
            d2 = dist[best_s][v]
            t += d2 / speed
            bat -= d2
            d_total += d2
            prev = best_s
        
        if t < ready[v]:
            t = ready[v]
        if t > due[v] + 1e-9:
            return False, 0.0, []
        t += service[v]
        full_route.append(v)
        prev = v
    
    dd = dist[prev][0]
    if bat >= dd - 1e-9:
        t += dd / speed
        bat -= dd
        d_total += dd
    else:
        best_s = -1
        best_cost = inf
        for s in stations:
            d1 = dist[prev][s]
            if d1 > bat + 1e-9:
                continue
            d2 = dist[s][0]
            if d2 > battery_cap + 1e-9:
                continue
            cost = d1 + d2
            if cost < best_cost:
                best_cost = cost
                best_s = s
        if best_s == -1:
            return False, 0.0, []
        d1 = dist[prev][best_s]
        t += d1 / speed
        bat -= d1
        d_total += d1
        charge_time = (battery_cap - bat) * recharge_rate
        t += charge_time
        bat = battery_cap
        full_route.append(best_s)
        d2 = dist[best_s][0]
        t += d2 / speed
        bat -= d2
        d_total += d2
    
    if t > horizon + 1e-9:
        return False, 0.0, []
    
    return True, d_total, full_route

def _cluster_customers(coords, n_customers, n_clusters):
    """Simple K-means clustering of customers."""
    cust = list(range(1, n_customers + 1))
    n = len(cust)
    if n_clusters >= n:
        return [[c] for c in cust]
    
    # Initialize centroids with spread-out points
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = np.array([coords[cust[i]] for i in indices])
    
    for _ in range(10):
        # Assign to nearest centroid
        dists = np.array([np.sqrt(((coords[c] - centroids)**2).sum(axis=1)) for c in cust])
        labels = dists.argmin(axis=1)
        
        # Update centroids
        new_centroids = []
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids.append(coords[cust][mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Build clusters
    clusters = [[] for _ in range(n_clusters)]
    for i, c in enumerate(cust):
        dists = np.sqrt(((coords[c] - centroids)**2).sum(axis=1))
        clusters[dists.argmin()].append(c)
    
    return [cl for cl in clusters if cl]

def _nearest_neighbor_route(start, remaining, dist):
    """Build a route using nearest neighbor heuristic."""
    route = [start]
    remaining = list(remaining)
    current = start
    
    while remaining:
        best_next = None
        best_dist = inf
        for v in remaining:
            d = dist[current][v]
            if d < best_dist:
                best_dist = d
                best_next = v
        if best_next is None:
            break
        route.append(best_next)
        remaining.remove(best_next)
        current = best_next
    
    return route

def _solve_instance(instance):
    t0 = time.time()
    deadline = t0 + 138.0
    
    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    N = len(coords)
    dist = _get_distance_matrix(coords)
    cust = list(range(1, n_customers + 1))
    
    # Estimate number of routes based on capacity
    total_demand = sum(demands[c] for c in cust)
    est_routes = max(1, int(np.ceil(total_demand / capacity)))
    est_routes = min(est_routes, len(cust))
    
    # Cluster and build initial routes
    clusters = _cluster_customers(coords, n_customers, est_routes)
    
    routes = []
    all_assigned = set()
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        # Sort by angle from depot for better initial route
        depot = coords[0]
        def angle(c):
            dx = coords[c][0] - depot[0]
            dy = coords[c][1] - depot[1]
            return np.arctan2(dy, dx)
        cluster_sorted = sorted(cluster, key=angle)
        
        # Try nearest neighbor from each starting point
        best_seq = None
        best_dist = inf
        for start in cluster_sorted[:5]:
            seq = _nearest_neighbor_route(start, [c for c in cluster_sorted if c != start], dist)
            ok, d, _ = _check_route_feasibility_fast(
                seq, dist, demands, ready, due, service,
                capacity, battery_cap, speed, recharge_rate, horizon, stations
            )
            if ok and d < best_dist:
                best_dist = d
                best_seq = seq
        
        if best_seq is None:
            # Fallback: try each customer individually
            for c in cluster:
                ok, d, _ = _check_route_feasibility_fast(
                    [c], dist, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations
                )
                if ok:
                    best_seq = [c]
                    best_dist = d
                    break
        
        if best_seq is not None:
            ok, d, full_route = _check_route_feasibility_fast(
                best_seq, dist, demands, ready, due, service,
                capacity, battery_cap, speed, recharge_rate, horizon, stations
            )
            if ok:
                routes.append(full_route)
                all_assigned.update(best_seq)
    
    # Check if all customers are assigned; if not, add remaining as single routes
    unassigned = [c for c in cust if c not in all_assigned]
    for c in unassigned:
        if time.time() > deadline:
            break
        ok, d, full_route = _check_route_feasibility_fast(
            [c], dist, demands, ready, due, service,
            capacity, battery_cap, speed, recharge_rate, horizon, stations
        )
        if ok:
            routes.append(full_route)
    
    # Local search
    def extract_custs(route):
        return [v for v in route if 1 <= v <= n_customers]
    
    route_dists = [_route_distance_fast(extract_custs(r), dist) for r in routes]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 2-opt intra-route
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            ri_custs = extract_custs(routes[i])
            n = len(ri_custs)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_seq = ri_custs[:a] + ri_custs[a:b+1][::-1] + ri_custs[b+1:]
                    ok, d, new_route = _check_route_feasibility_fast(
                        new_seq, dist, demands, ready, due, service,
                        capacity, battery_cap, speed, recharge_rate, horizon, stations
                    )
                    if ok and d < route_dists[i] - 1e-9:
                        routes[i] = new_route
                        route_dists[i] = d
                        improved = True
                        found = True
                        break
            if found:
                break
        if improved:
            continue
        
        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            ri_custs = extract_custs(routes[i])
            if len(ri_custs) <= 1:
                continue
            for j in range(len(ri_custs)):
                v = ri_custs[j]
                src_without = ri_custs[:j] + ri_custs[j+1:]
                ok_s, d_s, src_route = _check_route_feasibility_fast(
                    src_without, dist, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations
                )
                if not ok_s:
                    continue
                for k in range(len(routes)):
                    if k == i:
                        continue
                    rk_custs = extract_custs(routes[k])
                    for pos in range(len(rk_custs) + 1):
                        new_tgt = rk_custs[:pos] + [v] + rk_custs[pos:]
                        ok_t, d_t, tgt_route = _check_route_feasibility_fast(
                            new_tgt, dist, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate, horizon, stations
                        )
                        if not ok_t:
                            continue
                        new_pair = d_s + d_t
                        old_pair = route_dists[i] + route_dists[k]
                        if new_pair < old_pair - 1e-9:
                            routes[i] = src_route
                            routes[k] = tgt_route
                            route_dists[i] = d_s
                            route_dists[k] = d_t
                            if not routes[i]:
                                routes.pop(i)
                                route_dists.pop(i)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Swap
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            ri_custs = extract_custs(routes[i])
            for k in range(i + 1, len(routes)):
                rk_custs = extract_custs(routes[k])
                found = False
                for a in range(len(ri_custs)):
                    if found:
                        break
                    for b in range(len(rk_custs)):
                        vi = ri_custs[a]
                        vk = rk_custs[b]
                        new_ri = list(ri_custs)
                        new_rk = list(rk_custs)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i, route_i = _check_route_feasibility_fast(
                            new_ri, dist, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate, horizon, stations
                        )
                        ok_k, d_k, route_k = _check_route_feasibility_fast(
                            new_rk, dist, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate, horizon, stations
                        )
                        if ok_i and ok_k and d_i + d_k < route_dists[i] + route_dists[k] - 1e-9:
                            routes[i] = route_i
                            routes[k] = route_k
                            route_dists[i] = d_i
                            route_dists[k] = d_k
                            improved = True
                            found = True
                            break
                if found:
                    break
            if improved:
                break
        if improved:
            continue
    
    # Final validation and return
    final_routes = []
    for route in routes:
        custs = extract_custs(route)
        ok, d, full = _check_route_feasibility_fast(
            custs, dist, demands, ready, due, service,
            capacity, battery_cap, speed, recharge_rate, horizon, stations
        )
        if ok:
            final_routes.append(full)
    
    return final_routes

def plan(instance):
    return _solve_instance(instance)
# EVOLVE-BLOCK-END
