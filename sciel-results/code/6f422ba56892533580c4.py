"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_time_and_battery(route, D, coords, instance):
    """Return (total_time, total_dist, min_battery, feasible) for a route (list of nodes, depot implicit)."""
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)
    n = len(coords)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    t = 0.0
    bat = battery_cap
    dist = 0.0
    prev = 0
    min_bat = battery_cap
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, dist, bat, False
        min_bat = min(min_bat, bat)
        if v in stations:
            # recharge
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = battery_cap
        else:
            # customer
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, dist, bat, False
            t += service[v]
        prev = v
    # return to depot
    d = D[prev, 0]
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, dist, bat, False
    min_bat = min(min_bat, bat)
    if t > horizon + 1e-9:
        return t, dist, bat, False
    return t, dist, min_bat, True


def _route_demand(route, instance):
    return sum(instance["demands"][v] for v in route)


def _route_feasible(route, D, coords, instance):
    if not route:
        return True
    cap = instance.get("capacity", 100)
    if _route_demand(route, instance) > cap + 1e-9:
        return False
    _, _, _, feas = _route_time_and_battery(route, D, coords, instance)
    return feas


def _total_dist(routes, D):
    total = 0.0
    for r in routes:
        if not r:
            continue
        total += D[0, r[0]]
        for i in range(len(r) - 1):
            total += D[r[i], r[i + 1]]
        total += D[r[-1], 0]
    return total


def _construct_greedy(instance, D, coords, deadline):
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    # Sort by due time for initial ordering
    customers.sort(key=lambda v: instance["due"][v])

    routes = []
    unassigned = set(customers)

    # Build routes greedily: start a new route, add customers in order of nearest feasible
    while unassigned:
        if time.time() > deadline:
            break
        # Start new route from depot
        route = []
        prev = 0
        t = 0.0
        bat = instance.get("battery", 300)
        speed = instance.get("speed", 1)
        recharge_rate = instance.get("recharge_rate", 0.5)
        cap = instance.get("capacity", 100)
        demand = 0
        horizon = instance.get("horizon", 1000)
        stations = set(instance.get("stations", []))
        ready = instance["ready"]
        due = instance["due"]
        service = instance["service"]

        # Try to add customers to this route
        while unassigned:
            best_v = None
            best_cost = float('inf')
            for v in list(unassigned):
                d = D[prev, v]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                if demand + instance["demands"][v] > cap + 1e-9:
                    continue
                if v in stations:
                    new_t += (instance.get("battery", 300) - new_bat) * recharge_rate
                    new_bat = instance.get("battery", 300)
                else:
                    new_t = max(new_t, ready[v])
                    if new_t > due[v] + 1e-9:
                        continue
                    new_t += service[v]
                # Check return to depot
                ret_d = D[v, 0]
                ret_t = new_t + ret_d / speed
                ret_bat = new_bat - ret_d
                if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                    continue
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
            if best_v is None:
                break
            # Add best_v
            d = D[prev, best_v]
            t += d / speed
            bat -= d
            demand += instance["demands"][best_v]
            if best_v in stations:
                t += (instance.get("battery", 300) - bat) * recharge_rate
                bat = instance.get("battery", 300)
            else:
                t = max(t, ready[best_v])
                t += service[best_v]
            route.append(best_v)
            prev = best_v
            unassigned.discard(best_v)

        if route:
            routes.append(route)
        else:
            # Force assign one customer to avoid infinite loop
            v = min(unassigned, key=lambda x: instance["due"][x])
            routes.append([v])
            unassigned.discard(v)

    return routes


def _2opt(route, D, coords, instance, max_iter=100):
    """Apply 2-opt within a route, preserving feasibility."""
    if len(route) < 2:
        return route
    best = route[:]
    best_dist = 0.0
    if best:
        best_dist = D[0, best[0]]
        for i in range(len(best) - 1):
            best_dist += D[best[i], best[i + 1]]
        best_dist += D[best[-1], 0]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                # Quick distance check
                # Remove segments i-1->i, j->j+1, add i-1->j, i->j+1
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                if not _route_feasible(new_route, D, coords, instance):
                    continue
                best = new_route
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt(route, D, coords, instance, max_iter=50):
    """Apply Or-opt: move single nodes within route."""
    if len(route) < 3:
        return route
    best = route[:]
    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            for j in range(len(best)):
                if j == i or j == i - 1:
                    continue
                # Remove node i, insert at position j
                node = best[i]
                new_route = best[:i] + best[i + 1:]
                # Insert at j (adjust index if i < j)
                ins_j = j if i > j else j - 1
                if ins_j < 0:
                    ins_j = 0
                if ins_j > len(new_route):
                    ins_j = len(new_route)
                new_route = new_route[:ins_j] + [node] + new_route[ins_j:]
                if not _route_feasible(new_route, D, coords, instance):
                    continue
                # Check if actually different
                if new_route == best:
                    continue
                best = new_route
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _inter_route_swap(routes, D, coords, instance, max_iters=20):
    """Try swapping nodes between routes."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, coords, instance):
                            continue
                        if not _route_feasible(new_j, D, coords, instance):
                            continue
                        # Check total distance improvement
                        old_d = 0
                        for r in (routes[i], routes[j]):
                            if r:
                                old_d += D[0, r[0]]
                                for k in range(len(r) - 1):
                                    old_d += D[r[k], r[k + 1]]
                                old_d += D[r[-1], 0]
                        new_d = 0
                        for r in (new_i, new_j):
                            if r:
                                new_d += D[0, r[0]]
                                for k in range(len(r) - 1):
                                    new_d += D[r[k], r[k + 1]]
                                new_d += D[r[-1], 0]
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    coords = instance["coords"]
    D = _dist_matrix(coords)

    # Construct initial routes
    routes = _construct_greedy(instance, D, coords, deadline)

    if time.time() > deadline:
        return routes

    # Local search: 2-opt and Or-opt within routes
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, coords, instance)
        routes[r_idx] = _or_opt(routes[r_idx], D, coords, instance)

    if time.time() > deadline:
        return routes

    # Inter-route improvements
    routes = _inter_route_swap(routes, D, coords, instance)

    # Final 2-opt pass
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, coords, instance)

    return routes
# EVOLVE-BLOCK-END
