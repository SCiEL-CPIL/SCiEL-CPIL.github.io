import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

    def check_route(route_nodes):
        """Check if a route (list of customer/station ids) is valid. Returns (valid, distance)."""
        if not route_nodes:
            return True, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0
            if node in customers:
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return False, 0.0
                t += service[node]
                load += demands[node]
                if load > capacity + _EPS:
                    return False, 0.0
            else:
                # Station: recharge
                # The problem says: "a station adds (battery - energy) * recharge_rate time and refills"
                # Wait, "refills" implies battery becomes full? Or just adds energy?
                # "refills" usually means to full capacity.
                # Let's re-read carefully: "a station adds (battery - energy) * recharge_rate time and refills"
                # This phrasing is ambiguous. "battery - energy" is the deficit.
                # If it refills to full, the time taken is (capacity - current_battery) / recharge_rate?
                # Or is it (deficit) * recharge_rate?
                # Usually recharge time = energy_added / rate.
                # If it refills to full, energy_added = battery_cap - b.
                # Time = (battery_cap - b) / recharge_rate.
                # The text says "(battery - energy) * recharge_rate". This looks like a typo in my understanding or the problem statement.
                # Let's look at the standard interpretation.
                # If recharge_rate is 0.5 (units of time per unit energy? or energy per time?)
                # If rate is 0.5, and it's "time = delta_energy * rate", then higher rate = slower?
                # Usually rate = power. Time = Energy / Power.
                # If the problem says "adds ... time", it's a direct formula.
                # Let's assume: time += (battery_cap - b) * recharge_rate.
                # And b = battery_cap.
                # Let's check the magnitude. Battery 300. Deficit 100. Rate 0.5. Time = 50.
                # If rate was power (1/0.5 = 2), time = 100/2 = 50. Same result.
                # So time = deficit * recharge_rate is consistent with rate being time/energy.
                deficit = battery_cap - b
                if deficit > 0:
                    t += deficit * recharge_rate
                b = battery_cap
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0
        if t > horizon + _EPS:
            return False, 0.0
        d_total += d
        return True, d_total

    def route_distance(route_nodes):
        """Calculate distance of a route, assuming it's valid."""
        if not route_nodes:
            return 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d_total += dist[prev, node]
            prev = node
        d_total += dist[prev, 0]
        return d_total

    def optimize_route(route_nodes, time_limit=1.0):
        """Optimize a single route using 2-opt and relocate. Returns optimized route."""
        if len(route_nodes) <= 1:
            return route_nodes

        current_route = route_nodes[:]
        current_valid, current_dist = check_route(current_route)
        if not current_valid:
            return current_route

        best_route = current_route[:]
        best_dist = current_dist

        # 2-opt
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            n = len(current_route)
            for i in range(n - 1):
                if time.time() - t0 > time_limit:
                    break
                for j in range(i + 1, n):
                    # Check if reversing segment i..j improves distance
                    # Old edges: (i-1, i) and (j, j+1)  [with 0 being depot for i=0 or j=n-1]
                    # New edges: (i-1, j) and (i, j+1)
                    u = current_route[i-1] if i > 0 else 0
                    v = current_route[i]
                    w = current_route[j]
                    x = current_route[j+1] if j + 1 < n else 0

                    old_d = dist[u, v] + dist[w, x]
                    new_d = dist[u, w] + dist[v, x]

                    if new_d < old_d - _EPS:
                        # Tentative reverse
                        current_route[i:j+1] = current_route[i:j+1][::-1]
                        valid, d = check_route(current_route)
                        if valid and d < best_dist:
                            best_dist = d
                            best_route = current_route[:]
                        else:
                            # Revert
                            current_route[i:j+1] = current_route[i:j+1][::-1]
                        improved = True

        # Relocate
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            n = len(current_route)
            for i in range(n):
                if time.time() - t0 > time_limit:
                    break
                c = current_route[i]
                # Remove c
                temp_route = current_route[:i] + current_route[i+1:]
                if not temp_route:
                    continue
                # Try inserting at every position
                for j in range(len(temp_route) + 1):
                    if j == i or (j > i and j == i + 1):
                        continue
                    new_route = temp_route[:j] + [c] + temp_route[j:]
                    valid, d = check_route(new_route)
                    if valid and d < best_dist - _EPS:
                        best_dist = d
                        best_route = new_route
                        current_route = new_route
                        improved = True
                        break

        return best_route

    def cluster_customers(k):
        """Cluster customers into k groups using k-means."""
        if k >= len(customers):
            return [[c] for c in customers]

        # Initialize centroids randomly from customers
        centroids = np.array([coords[c] for c in random.sample(customers, k)])

        for _ in range(50):
            # Assign customers to nearest centroid
            labels = np.zeros(len(customers), dtype=int)
            for idx, c in enumerate(customers):
                d_to_centroids = np.sqrt(np.sum((centroids - coords[c])**2, axis=1))
                labels[idx] = np.argmin(d_to_centroids)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)
            for idx, c in enumerate(customers):
                new_centroids[labels[idx]] += coords[c]
                counts[labels[idx]] += 1

            for i in range(k):
                if counts[i] > 0:
                    new_centroids[i] /= counts[i]
                else:
                    # Re-initialize empty cluster with a random customer
                    new_centroids[i] = coords[random.choice(customers)]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Form clusters
        clusters = [[] for _ in range(k)]
        for idx, c in enumerate(customers):
            clusters[labels[idx]].append(c)

        return [cl for cl in clusters if cl]

    def build_solution(clusters):
        """Build routes from clusters and optimize them."""
        routes = []
        for cl in clusters:
            if not cl:
                continue
            # Initial route: sort by angle from depot or just random
            # Simple heuristic: sort by x-coordinate or angle
            angles = [math.atan2(coords[c][1], coords[c][0]) for c in cl]
            cl_sorted = [c for _, c in sorted(zip(angles, cl))]

            route = cl_sorted[:]
            # Optimize
            route = optimize_route(route)

            valid, d = check_route(route)
            if valid:
                routes.append(route)
            else:
                # If route is invalid, try to split it or just add customers one by one
                # For now, fallback to single customer routes for this cluster
                for c in cl:
                    single_route = [c]
                    valid_s, _ = check_route(single_route)
                    if valid_s:
                        routes.append(single_route)
                    else:
                        # Even single customer is invalid? Should not happen.
                        routes.append(single_route)

        return routes

    # Main Strategy:
    # 1. Try different numbers of clusters (k)
    # 2. For each k, run k-means with multiple restarts
    # 3. Optimize routes within clusters
    # 4. Keep the best solution found

    best_routes = None
    best_cost = float('inf')

    # Determine range of k
    min_k = 3
    max_k = 8

    for k in range(min_k, max_k + 1):
        if time.time() > DEADLINE - 5:
            break

        # Multiple restarts for k-means
        for restart in range(5):
            if time.time() > DEADLINE - 2:
                break

            clusters = cluster_customers(k)
            routes = build_solution(clusters)

            # Check if all customers are covered
            served = set()
            for r in routes:
                for c in r:
                    served.add(c)

            if len(served) != len(customers):
                # Missing customers, add them as singles
                missing = set(customers) - served
                for c in missing:
                    routes.append([c])

            # Calculate cost
            total_dist = 0
            valid_all = True
            for r in routes:
                valid, d = check_route(r)
                if not valid:
                    valid_all = False
                    break
                total_dist += d

            if valid_all and total_dist < best_cost:
                best_cost = total_dist
                best_routes = routes[:]

    # If no valid solution found, fallback to single customer routes
    if best_routes is None:
        best_routes = [[c] for c in customers]

    # Final repair: ensure all customers are present and routes are valid
    final_routes = []
    served = set()
    for r in best_routes:
        valid, d = check_route(r)
        if valid:
            final_routes.append(r)
            for c in r:
                served.add(c)

    missing = set(customers) - served
    for c in missing:
        final_routes.append([c])

    return final_routes
