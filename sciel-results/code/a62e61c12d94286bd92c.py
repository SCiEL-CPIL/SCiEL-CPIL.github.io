"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _build_distance_matrix(coords):
    n = len(coords)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dmat[i, j] = _euclidean(coords[i], coords[j])
            dmat[j, i] = dmat[i, j]
    return dmat


def _check_route(route, instance, dmat):
    """Check if a route is feasible. Returns (feasible, total_distance, end_time, end_battery)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0, 0, 0

    time = 0.0
    battery = float(battery_max)
    dist = 0.0
    cur = 0  # depot

    for v in route:
        d = dmat[cur][v]
        battery -= d
        time += d / speed
        dist += d
        if battery < -1e-9:
            return False, 0, 0, 0

        if v in stations:
            # Charge: add (battery - energy) * recharge_rate... actually refills to full
            # The spec: "a station adds (battery - energy) * recharge_rate time and refills"
            # I interpret: time added = (battery_max - battery) * recharge_rate, then battery = battery_max
            time += (battery_max - battery) * recharge_rate
            battery = float(battery_max)
        else:
            # Customer
            time = max(time, ready[v])
            if time > due[v] + 1e-9:
                return False, 0, 0, 0
            time += service[v]

        cur = v

    # Return to depot
    d = dmat[cur][0]
    battery -= d
    time += d / speed
    dist += d
    if battery < -1e-9:
        return False, 0, 0, 0
    if time > horizon + 1e-9:
        return False, 0, 0, 0

    return True, dist, time, battery


def _route_distance(route, dmat):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = dmat[0][route[0]]
    for i in range(len(route) - 1):
        total += dmat[route[i]][route[i + 1]]
    total += dmat[route[-1]][0]
    return total


def _two_opt_improve(route, dmat, instance, max_iters=50):
    """Apply 2-opt improvement while maintaining feasibility."""
    if len(route) < 3:
        return route

    best = route[:]
    best_dist = _route_distance(best, dmat)
    feasible, _, _, _ = _check_route(best, instance, dmat)
    if not feasible:
        return route

    for _ in range(max_iters):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                # Try reversing segment i..j
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if candidate == best:
                    continue
                feasible, dist, _, _ = _check_route(candidate, instance, dmat)
                if feasible and dist < best_dist - 1e-6:
                    best = candidate
                    best_dist = dist
                    improved = True
        if not improved:
            break
    return best


def plan(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    dmat = _build_distance_matrix(coords)

    # Start with each customer as its own route
    routes = [[c] for c in customers]

    # Sort customers by due time for better initial ordering
    # Use savings heuristic: compute savings for merging routes
    # savings(i, j) = d(0,i) + d(0,j) - d(i,j)
    # We want to merge routes that give the largest savings

    # For each pair of routes, try to merge them
    # A route is a list of node ids. Merging route A and route B means
    # concatenating them in some order: A+B, B+A, A+B reversed, etc.
    # We need to check feasibility.

    # To make this efficient, we'll try all pairs and pick the best feasible merge
    while len(routes) > 1:
        best_saving = 0
        best_merge = None

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                A = routes[i]
                B = routes[j]

                # Try all 4 orderings: A+B, B+A, A_rev+B, A+B_rev, etc.
                # Actually the standard savings merges the end of one with the start of another.
                # Let's try: A + B, B + A, A[::-1] + B, B[::-1] + A
                candidates = [
                    A + B,
                    B + A,
                    A[::-1] + B,
                    B[::-1] + A,
                    A + B[::-1],
                    B + A[::-1],
                    A[::-1] + B[::-1],
                    B[::-1] + A[::-1],
                ]

                for cand in candidates:
                    feasible, dist, _, _ = _check_route(cand, instance, dmat)
                    if not feasible:
                        continue
                    # Compute saving: distance(A) + distance(B) - distance(cand)
                    dA = _route_distance(A, dmat)
                    dB = _route_distance(B, dmat)
                    saving = dA + dB - dist
                    if saving > best_saving:
                        best_saving = saving
                        best_merge = (i, j, cand)

        if best_merge is None:
            break

        i, j, merged = best_merge
        # Remove routes i and j, add merged
        # Be careful with indices: remove the higher index first
        if i > j:
            i, j = j, i
        # Remove j first, then i
        routes.pop(j)
        routes.pop(i)
        routes.append(merged)

    # Apply 2-opt improvement to each route
    improved_routes = []
    for r in routes:
        ir = _two_opt_improve(r, dmat, instance)
        improved_routes.append(ir)

    return improved_routes
# EVOLVE-BLOCK-END
