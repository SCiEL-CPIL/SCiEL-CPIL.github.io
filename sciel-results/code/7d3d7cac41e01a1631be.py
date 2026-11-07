import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0  # Safety margin for 150s kill

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

    cx = coords[:, 0]
    cy = coords[:, 1]
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    is_cust_arr = np.zeros(n_nodes, dtype=bool)
    is_cust_arr[1:n_cust+1] = True
    is_stat_arr = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_stat_arr[s] = True

    def check_route(route):
        """Returns (total_dist, valid, final_time, final_battery)"""
        if not route:
            return 0.0, True, 0.0, battery_max

        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        dist_l = dist
        demands_l = demands
        ready_l = ready
        due_l = due
        service_l = service
        is_cust_l = is_cust_arr
        is_stat_l = is_stat_arr
        cap_l = capacity
        bmax_l = battery_max
        speed_l = speed
        rr_l = recharge_rate
        hor_l = horizon
        eps_l = _EPS

        for node in route:
            if is_cust_l[node]:
                load += demands_l[node]
        if load > cap_l + eps_l:
            return float('inf'), False, 0.0, 0.0

        for node in route:
            d = dist_l[prev, node]
            total_dist += d
            t += d / speed_l
            b -= d

            if b < -eps_l:
                return float('inf'), False, 0.0, 0.0

            if is_stat_l[node]:
                t += (bmax_l - b) * rr_l
                b = bmax_l
            elif is_cust_l[node]:
                if t < ready_l[node]:
                    t = ready_l[node]
                if t > due_l[node] + eps_l:
                    return float('inf'), False, 0.0, 0.0
                t += service_l[node]

            prev = node

        d = dist_l[prev, 0]
        total_dist += d
        t += d / speed_l
        b -= d

        if b < -eps_l:
            return float('inf'), False, 0.0, 0.0

        if t > hor_l + eps_l:
            return float('inf'), False, 0.0, 0.0

        return total_dist, True, t, b

    def route_load(route):
        return sum(demands[n] for n in route if is_cust_arr[n])

    def try_repair(route):
        c, v, _, _ = check_route(route)
        if v:
            return route, c

        best_route = None
        best_cost = float('inf')

        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c2, v2, _, _ = check_route(cand)
                if v2 and c2 < best_cost:
                    best_cost = c2
                    best_route = cand

        if best_route:
            return best_route, best_cost
        return None, float('inf')

    # --- Construction: Due-date-sorted greedy with nearest-feasible insertion ---
    def greedy_construct():
        """Build routes using due-date ordering and nearest-feasible customer insertion."""
        customers_sorted = sorted(cust_ids, key=lambda c: (due[c], ready[c]))
        routes = []
        remaining = set(cust_ids)

        while remaining:
            if time.time() > DEADLINE - 30.0:
                break
            best_route = None
            best_cost = float('inf')

            # Try each remaining customer as a potential start (limited to first 8 by due)
            start_candidates = [c for c in customers_sorted if c in remaining][:8]

            for start in start_candidates:
                route = [start]
                used = {start}
                cur_cost, cur_valid, _, _ = check_route(route)
                if not cur_valid:
                    continue

                # Greedily add nearest feasible customer
                while True:
                    best_next = None
                    best_metric = float('inf')

                    for c in remaining - used:
                        # Try inserting c at each position in current route
                        for pos in range(len(route) + 1):
                            cand = route[:pos] + [c] + route[pos:]
                            # Quick capacity check
                            if route_load(cand) > capacity + _EPS:
                                continue
                            c_cost, c_valid, _, _ = check_route(cand)
                            if c_valid:
                                # Metric: distance saving from insertion
                                if pos == 0:
                                    saving = dist[0, route[0]] + dist[0, c] - dist[0, c] - dist[c, route[0]]
                                elif pos == len(route):
                                    saving = dist[route[-1], 0] + dist[route[-1], c] - dist[route[-1], c] - dist[c, 0]
                                else:
                                    saving = dist[route[pos-1], route[pos]] - dist[route[pos-1], c] - dist[c, route[pos]]
                                if saving > best_metric:
                                    best_metric = saving
                                    best_next = (c, pos)

                    if best_next is None:
                        break

                    c_to_add, pos = best_next
                    route = route[:pos] + [c_to_add] + route[pos:]
                    used.add(c_to_add)
                    remaining.discard(c_to_add)

                c_final, v_final, _, _ = check_route(route)
                if v_final and c_final < best_cost:
                    best_cost = c_final
                    best_route = route[:]

            if best_route is not None:
                routes.append(best_route)
            else:
                # Fallback: single customer per route for remaining
                for c in list(remaining):
                    r = [c]
                    c_r, v_r, _, _ = check_route(r)
                    if v_r:
                        routes.append(r)
                        remaining.discard(c)
                    else:
                        rep, _ = try_repair(r)
                        if rep:
                            routes.append(rep)
                            remaining.discard(c)

        return routes

    # --- Alternative: savings-based construction (from current program) ---
    def savings_construct():
        """Savings-based construction: start with individual routes, merge best pairs."""
        routes = [[c] for c in cust_ids]
        active = [True] * len(routes)
        route_costs = [check_route(r)[0] for r in routes]

        changed = True
        while changed:
            changed = False
            best_saving = 1e-6
            best_i = -1
            best_j = -1
            best_merged = None
            best_cost = float('inf')

            for i in range(len(routes)):
                if not active[i] or route_costs[i] == float('inf'):
                    continue
                last_i = routes[i][-1]

                for j in range(i + 1, len(routes)):
                    if not active[j] or route_costs[j] == float('inf'):
                        continue
                    first_j = routes[j][0]

                    est_saving = dist[last_i, 0] + dist[0, first_j] - dist[last_i, first_j]
                    if est_saving < best_saving:
                        continue

                    merged1 = routes[i] + routes[j]
                    c1, v1, _, _ = check_route(merged1)
                    if v1:
                        saving = route_costs[i] + route_costs[j] - c1
                        if saving > best_saving:
                            best_saving = saving
                            best_i = i
                            best_j = j
                            best_merged = merged1
                            best_cost = c1

                    first_i = routes[i][0]
                    last_j = routes[j][-1]
                    est_saving2 = dist[last_j, 0] + dist[0, first_i] - dist[last_j, first_i]
                    if est_saving2 < best_saving:
                        continue

                    merged2 = routes[j] + routes[i]
                    c2, v2, _, _ = check_route(merged2)
                    if v2:
                        saving = route_costs[i] + route_costs[j] - c2
                        if saving > best_saving:
                            best_saving = saving
                            best_i = i
                            best_j = j
                            best_merged = merged2
                            best_cost = c2

            if best_i != -1:
                routes[best_i] = best_merged
                route_costs[best_i] = best_cost
                active[best_j] = False
                changed = True

        routes = [r for i, r in enumerate(routes) if active[i]]
        return routes

    # --- Local Search ---
    def optimize_route(route):
        """Apply 2-opt + or-opt to minimize distance while maintaining feasibility."""
        n = len(route)
        if n < 2:
            return route, check_route(route)[0]

        cur_cost, cur_valid, _, _ = check_route(route)
        if not cur_valid:
            rep, rcost = try_repair(route)
            if rep:
                route = rep
                cur_cost = rcost
                cur_valid = True
            else:
                return route, float('inf')

        best_route = list(route)
        best_cost = cur_cost

        # 2-opt
        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr - 1):
                if time.time() > DEADLINE - 5.0:
                    break
                for j in range(i + 1, n_curr):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True

        # Or-opt (move 1-3 consecutive nodes)
        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr):
                if time.time() > DEADLINE - 5.0:
                    break
                for k in range(1, 4):
                    if i + k > n_curr:
                        continue
                    segment = best_route[i:i+k]
                    rest = best_route[:i] + best_route[i+k:]
                    for j in range(len(rest) + 1):
                        cand = rest[:j] + segment + rest[j:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            improved = True

        return best_route, best_cost

    def local_search(routes, route_costs):
        """Full local search: intra-route + inter-route moves."""
        improved = True
        iteration = 0
        while improved and time.time() < DEADLINE - 10.0:
            improved = False
            iteration += 1

            # 1. Intra-route optimization
            for ri in range(len(routes)):
                if time.time() > DEADLINE - 10.0:
                    break
                route = routes[ri]
                new_route, new_cost = optimize_route(route)
                if new_cost < route_costs[ri] - 1e-7:
                    routes[ri] = new_route
                    route_costs[ri] = new_cost
                    improved = True

            # 2. Inter-route Relocate
            if time.time() > DEADLINE - 10.0:
                break
            relocate_done = False
            for ri in range(len(routes)):
                if time.time() > DEADLINE - 10.0 or relocate_done:
                    break
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE - 10.0 or relocate_done:
                        break
                    cust = routes[ri][ci]
                    if not is_cust_arr[cust]:
                        continue

                    temp_from = routes[ri][:]
                    del temp_from[ci]
                    c_from, v_from, _, _ = check_route(temp_from)
                    if not v_from:
                        continue

                    for rj in range(len(routes)):
                        if rj == ri or time.time() > DEADLINE - 10.0 or relocate_done:
                            continue
                        for cj in range(len(routes[rj]) + 1):
                            if time.time() > DEADLINE - 10.0 or relocate_done:
                                break
                            cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                            load_to = route_load(routes[rj]) + demands[cust]
                            if load_to > capacity + _EPS:
                                continue
                            c_to, v_to, _, _ = check_route(cand_to)
                            if not v_to:
                                continue
                            old_cost = route_costs[ri] + route_costs[rj]
                            new_cost = c_from + c_to
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = temp_from
                                routes[rj] = cand_to
                                route_costs[ri] = c_from
                                route_costs[rj] = c_to
                                improved = True
                                relocate_done = True
                                break
                        if relocate_done:
                            break
                    if relocate_done:
                        break
                if relocate_done:
                    break

            # Remove empty routes
            new_routes = []
            new_costs = []
            for i, r in enumerate(routes):
                if r:
                    new_routes.append(r)
                    new_costs.append(route_costs[i])
            if len(new_routes) != len(routes):
                routes = new_routes
                route_costs = new_costs

            # 3. Inter-route Swap
            if time.time() > DEADLINE - 10.0:
                break
            swap_done = False
            for ri in range(len(routes)):
                if time.time() > DEADLINE - 10.0 or swap_done:
                    break
                for rj in range(ri + 1, len(routes)):
                    if time.time() > DEADLINE - 10.0 or swap_done:
                        break
                    for ci in range(len(routes[ri])):
                        if time.time() > DEADLINE - 10.0 or swap_done:
                            break
                        cust1 = routes[ri][ci]
                        if not is_cust_arr[cust1]:
                            continue
                        for cj in range(len(routes[rj])):
                            if time.time() > DEADLINE - 10.0 or swap_done:
                                break
                            cust2 = routes[rj][cj]
                            if not is_cust_arr[cust2]:
                                continue

                            load_i = route_load(routes[ri])
                            load_j = route_load(routes[rj])
                            new_load_i = load_i - demands[cust1] + demands[cust2]
                            new_load_j = load_j - demands[cust2] + demands[cust1]

                            if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                                continue

                            new_route_i = routes[ri][:]
                            new_route_i[ci] = cust2
                            new_route_j = routes[rj][:]
                            new_route_j[cj] = cust1

                            c_i, v_i, _, _ = check_route(new_route_i)
                            c_j, v_j, _, _ = check_route(new_route_j)

                            if v_i and v_j:
                                old_c_i = route_costs[ri]
                                old_c_j = route_costs[rj]
                                if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                    routes[ri] = new_route_i
                                    routes[rj] = new_route_j
                                    route_costs[ri] = c_i
                                    route_costs[rj] = c_j
                                    improved = True
                                    swap_done = True
                                    break
                    if swap_done:
                        break
                if swap_done:
                    break

            # 4. Inter-route 2-opt (cross-route edge swap / suffix swap)
            if time.time() > DEADLINE - 10.0:
                break
            cross2opt_done = False
            for ri in range(len(routes)):
                if time.time() > DEADLINE - 10.0 or cross2opt_done:
                    break
                for rj in range(ri + 1, len(routes)):
                    if time.time() > DEADLINE - 10.0 or cross2opt_done:
                        break
                    for ci in range(len(routes[ri]) - 1):
                        if time.time() > DEADLINE - 10.0 or cross2opt_done:
                            break
                        node1 = routes[ri][ci]
                        if not is_cust_arr[node1]:
                            continue
                        for cj in range(len(routes[rj]) - 1):
                            if time.time() > DEADLINE - 10.0 or cross2opt_done:
                                break
                            node2 = routes[rj][cj]
                            if not is_cust_arr[node2]:
                                continue

                            new_route_i = routes[ri][:ci+1] + routes[rj][cj+1:]
                            new_route_j = routes[rj][:cj+1] + routes[ri][ci+1:]

                            load_i = route_load(new_route_i)
                            load_j = route_load(new_route_j)
                            if load_i > capacity + _EPS or load_j > capacity + _EPS:
                                continue

                            c_i, v_i, _, _ = check_route(new_route_i)
                            if not v_i:
                                continue
                            c_j, v_j, _, _ = check_route(new_route_j)
                            if not v_j:
                                continue

                            old_cost = route_costs[ri] + route_costs[rj]
                            new_cost = c_i + c_j
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                route_costs[ri] = c_i
                                route_costs[rj] = c_j
                                improved = True
                                cross2opt_done = True
                                break
                    if cross2opt_done:
                        break
                if cross2opt_done:
                    break

        return routes, route_costs

    # --- Main: Try multiple constructions, keep best ---
    best_routes = None
    best_total = float('inf')

    # Construction 1: Greedy due-date based
    if time.time() < DEADLINE - 40.0:
        r1 = greedy_construct()
        if r1:
            costs1 = [check_route(r)[0] for r in r1]
            total1 = sum(costs1)
            if total1 < best_total:
                best_total = total1
                best_routes = r1

    # Construction 2: Savings-based
    if time.time() < DEADLINE - 40.0:
        r2 = savings_construct()
        if r2:
            costs2 = [check_route(r)[0] for r in r2]
            total2 = sum(costs2)
            if total2 < best_total:
                best_total = total2
                best_routes = r2

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in cust_ids]
        for i in range(len(best_routes)):
            if not check_route(best_routes[i])[1]:
                rep, _ = try_repair(best_routes[i])
                if rep:
                    best_routes[i] = rep

    # Local search on best construction
    route_costs = [check_route(r)[0] for r in best_routes]
    best_routes, route_costs = local_search(best_routes, route_costs)

    # Final polish: tighter tolerance 2-opt
    if time.time() < DEADLINE - 3.0:
        for ri in range(len(best_routes)):
            if time.time() > DEADLINE - 3.0:
                break
            n = len(best_routes[ri])
            if n < 2:
                continue
            cur_cost, cur_valid, _, _ = check_route(best_routes[ri])
            if not cur_valid:
                continue
            best_route = list(best_routes[ri])
            best_cost = cur_cost
            improved = True
            while improved and time.time() < DEADLINE - 3.0:
                improved = False
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-9:
                            best_cost = c
                            best_route = cand
                            improved = True
            if best_cost < route_costs[ri] - 1e-9:
                best_routes[ri] = best_route
                route_costs[ri] = best_cost

    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in best_routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
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
        best_routes = [[v] for v in customers]
        for i in range(len(best_routes)):
            if not check_route(best_routes[i])[1]:
                rep, _ = try_repair(best_routes[i])
                if rep:
                    best_routes[i] = rep

    return best_routes
