import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t_start = time.time()
    DEADLINE = 145.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_set = set(stations)

    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_set

    def check_route(route):
        """Returns (total_dist, valid, final_time, final_battery)."""
        if not route:
            return 0.0, True, 0.0, battery_max

        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0

        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False, 0.0, 0.0

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            if b < -_EPS:
                return float('inf'), False, 0.0, 0.0
            if is_station(node):
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            prev = node

        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
        if t > horizon + _EPS:
            return float('inf'), False, 0.0, 0.0

        return total_dist, True, t, b

    def route_cost(route):
        return check_route(route)[0]

    def fast_check(route):
        """Fast check, returns (cost, valid)."""
        if not route:
            return 0.0, True
        total_dist = 0.0
        t = 0.0
        b = battery_max
        prev = 0
        load = 0.0
        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False
        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            if b < -_EPS:
                return float('inf'), False
            if is_station(node):
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + _EPS:
                    return float('inf'), False
                t += service[node]
            prev = node
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        if b < -_EPS:
            return float('inf'), False
        if t > horizon + _EPS:
            return float('inf'), False
        return total_dist, True

    def try_insert_station(route, max_stations=2):
        total_dist, valid, _, _ = check_route(route)
        if valid:
            return route, total_dist
        best_route = None
        best_cost = float('inf')
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand
        if best_route:
            return best_route, best_cost
        if max_stations >= 2:
            for s1 in stations:
                for i1 in range(len(route) + 1):
                    cand1 = route[:i1] + [s1] + route[i1:]
                    for s2 in stations:
                        for i2 in range(len(cand1) + 1):
                            cand2 = cand1[:i2] + [s2] + cand1[i2:]
                            c, v, _, _ = check_route(cand2)
                            if v and c < best_cost:
                                best_cost = c
                                best_route = cand2
        if best_route:
            return best_route, best_cost
        return None, float('inf')

    def construct_nearest_neighbor():
        remaining = list(cust_ids)
        routes = []
        while remaining:
            seed = min(remaining, key=lambda x: due[x])
            route = [seed]
            remaining.remove(seed)
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                best_c = None
                best_pos = -1
                best_cost = float('inf')
                best_cand = None
                load = sum(demands[n] for n in route if is_customer(n))
                for c in remaining:
                    if load + demands[c] > capacity + _EPS:
                        continue
                    for pos in range(len(route) + 1):
                        cand = route[:pos] + [c] + route[pos:]
                        c_cost, v = fast_check(cand)
                        if not v:
                            rep, rc = try_insert_station(cand)
                            if rep:
                                c_cost = rc
                                v = True
                                cand = rep
                        if v and c_cost < best_cost:
                            best_cost = c_cost
                            best_c = c
                            best_pos = pos
                            best_cand = cand
                if best_c is not None:
                    route = best_cand
                    remaining.remove(best_c)
                    improved = True
            routes.append(route)
        return routes

    def construct_savings():
        routes = [[c] for c in cust_ids]
        savings = {}
        for i in cust_ids:
            for j in cust_ids:
                if i != j:
                    s = dist[0, i] + dist[0, j] - dist[i, j]
                    savings[(i, j)] = s
        pairs = sorted(savings.items(), key=lambda x: -x[1])
        route_of = {}
        for idx, r in enumerate(routes):
            for c in r:
                route_of[c] = idx
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for (i, j), s in pairs:
                if time.time() > DEADLINE:
                    break
                ri = route_of.get(i)
                rj = route_of.get(j)
                if ri is None or rj is None or ri == rj:
                    continue
                route_i = routes[ri]
                route_j = routes[rj]
                if route_i[-1] != i or route_j[0] != j:
                    continue
                cand = route_i + route_j
                c_cost, v = fast_check(cand)
                if not v:
                    rep, rc = try_insert_station(cand)
                    if rep:
                        c_cost = rc
                        v = True
                        cand = rep
                if v:
                    cand2 = route_j + route_i
                    c2, v2 = fast_check(cand2)
                    if not v2:
                        rep2, rc2 = try_insert_station(cand2)
                        if rep2:
                            c2 = rc2
                            v2 = True
                            cand2 = rep2
                    if v2 and c2 < c_cost:
                        cand = cand2
                    routes[ri] = cand
                    for c in route_j:
                        route_of[c] = ri
                    routes[rj] = []
                    changed = True
        return [r for r in routes if r]

    def construct_random_insertion():
        remaining = list(cust_ids)
        random.shuffle(remaining)
        routes = []
        for c in remaining:
            if time.time() > DEADLINE:
                break
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            best_cand = None
            for ri, route in enumerate(routes):
                load = sum(demands[n] for n in route if is_customer(n))
                if load + demands[c] > capacity + _EPS:
                    continue
                for pos in range(len(route) + 1):
                    cand = route[:pos] + [c] + route[pos:]
                    c_cost, v = fast_check(cand)
                    if not v:
                        rep, rc = try_insert_station(cand)
                        if rep:
                            c_cost = rc
                            v = True
                            cand = rep
                    if v and c_cost < best_cost:
                        best_cost = c_cost
                        best_route_idx = ri
                        best_pos = pos
                        best_cand = cand
            if best_route_idx >= 0:
                routes[best_route_idx] = best_cand
            else:
                routes.append([c])
        return routes

    def two_opt(route):
        if len(route) < 2:
            return route, route_cost(route)
        best_route = list(route)
        best_cost, _ = fast_check(best_route)
        n = len(best_route)
        improved = True
        max_iter = 30
        while improved and max_iter > 0 and time.time() < DEADLINE:
            improved = False
            max_iter -= 1
            for i in range(n - 1):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v = fast_check(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        return best_route, best_cost

    def or_opt(route):
        """Move a segment of length 1, 2, or 3 to another position."""
        best_route = list(route)
        best_cost, _ = fast_check(best_route)
        n = len(best_route)
        improved = True
        max_iter = 20
        while improved and max_iter > 0 and time.time() < DEADLINE:
            improved = False
            max_iter -= 1
            for seg_len in [1, 2, 3]:
                if n < seg_len:
                    continue
                for i in range(n - seg_len + 1):
                    if time.time() > DEADLINE:
                        break
                    seg = best_route[i:i+seg_len]
                    rest = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(rest) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        cand = rest[:j] + seg + rest[j:]
                        c, v = fast_check(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            improved = True
        return best_route, best_cost

    def alns_destroy(routes, destroy_fraction):
        all_customers = [c for r in routes for c in r if is_customer(c)]
        if not all_customers:
            return routes, []
        n_remove = max(1, int(len(all_customers) * destroy_fraction))
        to_remove = set(random.sample(all_customers, min(n_remove, len(all_customers))))
        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in to_remove]
            if new_r:
                new_routes.append(new_r)
        return new_routes, list(to_remove)

    def alns_repair(routes, customers_to_insert):
        for c in customers_to_insert:
            if time.time() > DEADLINE:
                break
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            best_cand = None
            for ri, route in enumerate(routes):
                load = sum(demands[n] for n in route if is_customer(n))
                if load + demands[c] > capacity + _EPS:
                    continue
                for pos in range(len(route) + 1):
                    cand = route[:pos] + [c] + route[pos:]
                    c_cost, v = fast_check(cand)
                    if not v:
                        rep, rc = try_insert_station(cand)
                        if rep:
                            c_cost = rc
                            v = True
                            cand = rep
                    if v and c_cost < best_cost:
                        best_cost = c_cost
                        best_route_idx = ri
                        best_pos = pos
                        best_cand = cand
            if best_route_idx >= 0:
                routes[best_route_idx] = best_cand
            else:
                routes.append([c])
        return routes

    def total_distance(routes):
        return sum(route_cost(r) for r in routes)

    # Build multiple initial solutions
    candidates = []

    r1 = construct_nearest_neighbor()
    if r1 and all(fast_check(r)[1] for r in r1):
        candidates.append(r1)

    if time.time() < DEADLINE - 5:
        r2 = construct_savings()
        if r2 and all(fast_check(r)[1] for r in r2):
            candidates.append(r2)

    for _ in range(3):
        if time.time() > DEADLINE - 5:
            break
        r3 = construct_random_insertion()
        if r3 and all(fast_check(r)[1] for r in r3):
            candidates.append(r3)

    if not candidates:
        candidates.append(construct_nearest_neighbor())

    best_sol = min(candidates, key=total_distance)
    best_cost = total_distance(best_sol)

    for i in range(len(best_sol)):
        if time.time() > DEADLINE - 10:
            break
        best_sol[i], _ = two_opt(best_sol[i])
    best_cost = total_distance(best_sol)

    current_routes = [list(r) for r in best_sol]
    current_cost = best_cost
    best_routes = [list(r) for r in best_sol]
    best_alns_cost = best_cost

    no_improve = 0
    max_no_improve = 50
    destroy_fraction = 0.3

    while time.time() < DEADLINE - 5:
        destroyed_routes, removed = alns_destroy(current_routes, destroy_fraction)
        if not removed:
            break

        repaired = alns_repair(destroyed_routes, removed)

        for i in range(len(repaired)):
            if time.time() > DEADLINE - 5:
                break
            repaired[i], _ = two_opt(repaired[i])
            repaired[i], _ = or_opt(repaired[i])

        if all(fast_check(r)[1] for r in repaired):
            new_cost = total_distance(repaired)
            if new_cost < current_cost - 1e-7:
                current_routes = repaired
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_alns_cost - 1e-7:
                    best_alns_cost = new_cost
                    best_routes = [list(r) for r in repaired]
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            current_routes = [list(r) for r in best_routes]
            current_cost = best_alns_cost
            no_improve = 0
            destroy_fraction = min(0.5, destroy_fraction * 1.2)
            if destroy_fraction > 0.5:
                break

    final_routes = best_routes

    for i in range(len(final_routes)):
        if time.time() > DEADLINE - 2:
            break
        final_routes[i], _ = two_opt(final_routes[i])
        final_routes[i], _ = or_opt(final_routes[i])

    served = set()
    valid = True
    for r in final_routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break

    if not valid or served != set(range(1, n_cust + 1)):
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        final_routes = [[v] for v in customers]
        for i in range(len(final_routes)):
            if not check_route(final_routes[i])[1]:
                rep, _ = try_insert_station(final_routes[i])
                if rep:
                    final_routes[i] = rep

    return final_routes
