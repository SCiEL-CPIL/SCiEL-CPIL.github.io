import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

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

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

    # Helper to check if a route is valid and return its cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0, 0.0, 0.0
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
                return False, 0.0, t, d_total, load
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, t, d_total, load
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, t, d_total, load
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, t, d_total, load
        if t > horizon + _EPS:
            return False, 0.0, t, d_total, load
        d_total += d
        return True, d_total, t, d_total, load

    # Phase 1: Build initial routes using nearest neighbor + insertion
    # Sort customers by angle from depot for a good starting order
    depot = coords[0]
    angles = [math.atan2(coords[c][1] - depot[1], coords[c][0] - depot[0]) for c in customers]
    sorted_customers = [c for _, c in sorted(zip(angles, customers))]

    routes = []
    for c in sorted_customers:
        # Try to insert c into an existing route
        best_route_idx = -1
        best_pos = -1
        best_delta = float('inf')
        
        for i, r in enumerate(routes):
            for pos in range(len(r) + 1):
                trial = r[:pos] + [c] + r[pos:]
                valid, d, t, _, _ = check_route(trial)
                if valid:
                    # Compute delta distance
                    old_d = 0.0 if not r else check_route(r)[1]
                    delta = d - old_d
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = i
                        best_pos = pos
        
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_pos, c)
        else:
            routes.append([c])

    # Phase 2: Local Search with 2-opt, Or-opt, and Relocate
    def get_route_cost(route):
        if not route:
            return 0.0
        valid, d, t, _, _ = check_route(route)
        if not valid:
            return float('inf')
        return d

    def two_opt(route):
        n = len(route)
        improved = False
        for i in range(n - 1):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                valid, d, t, _, _ = check_route(new_route)
                if valid:
                    old_d = get_route_cost(route)
                    if d < old_d - _EPS:
                        route = new_route
                        improved = True
        return route, improved

    def or_opt(route):
        # Move a segment of 1, 2, or 3 nodes to another position
        improved = False
        n = len(route)
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                if time.time() > DEADLINE:
                    break
                seg = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                for j in range(len(rest) + 1):
                    if j >= i and j < i + seg_len:
                        continue  # Don't insert at same position
                    new_route = rest[:j] + seg + rest[j:]
                    if set(new_route) != set(route):
                        continue  # Skip if no change
                    valid, d, t, _, _ = check_route(new_route)
                    if valid:
                        old_d = get_route_cost(route)
                        if d < old_d - _EPS:
                            route = new_route
                            improved = True
                            break
        return route, improved

    def relocate_route(route, c, new_pos):
        # Remove c and insert at new_pos in the same route
        idx = route.index(c)
        route.pop(idx)
        route.insert(new_pos, c)
        valid, d, t, _, _ = check_route(route)
        if not valid:
            route.pop(new_pos)
            route.insert(idx, c)
            return False, 0.0
        return True, d

    # Iterative improvement
    for iteration in range(100):
        if time.time() > DEADLINE:
            break
        improved = False

        # 2-opt
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            new_r, imp = two_opt(r)
            if imp:
                routes[i] = new_r
                improved = True

        # Or-opt
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            new_r, imp = or_opt(r)
            if imp:
                routes[i] = new_r
                improved = True

        # Relocate within same route
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            for j, c in enumerate(r):
                if time.time() > DEADLINE:
                    break
                for k in range(len(r)):
                    if k == j or k == j + 1:
                        continue
                    insert_idx = k if k < j else k - 1
                    new_r = list(r)
                    new_r.pop(j)
                    new_r.insert(insert_idx, c)
                    valid, d, t, _, _ = check_route(new_r)
                    if valid:
                        old_d = get_route_cost(r)
                        if d < old_d - _EPS:
                            routes[i] = new_r
                            improved = True
                            break

        # Relocate to other routes
        for i, r in enumerate(routes):
            if time.time() > DEADLINE:
                break
            for j, c in enumerate(r):
                for m, other_r in enumerate(routes):
                    if m == i:
                        continue
                    # Try inserting c into other_r at every position
                    best_pos = -1
                    best_d = float('inf')
                    old_d_i = get_route_cost(r)
                    old_d_m = get_route_cost(other_r)
                    old_total = old_d_i + old_d_m

                    # Remove c from r
                    new_r = list(r)
                    new_r.pop(j)

                    for pos in range(len(other_r) + 1):
                        trial_other = other_r[:pos] + [c] + other_r[pos:]
                        valid_r, d_r, t_r, _, _ = check_route(new_r)
                        valid_o, d_o, t_o, _, _ = check_route(trial_other)
                        if valid_r and valid_o:
                            new_total = d_r + d_o
                            if new_total < best_d:
                                best_d = new_total
                                best_pos = pos

                    if best_pos != -1 and best_d < old_total - _EPS:
                        # Perform move
                        routes[i] = new_r
                        routes[m] = other_r[:best_pos] + [c] + other_r[best_pos:]
                        improved = True
                        break

        if not improved:
            break

    # Final Validation and Return
    final_routes = []
    for r in routes:
        if r:
            final_routes.append(r)

    # Check coverage
    served = set()
    for r in final_routes:
        for c in r:
            served.add(c)

    missing = set(customers) - served
    if missing:
        for c in missing:
            final_routes.append([c])

    return final_routes
