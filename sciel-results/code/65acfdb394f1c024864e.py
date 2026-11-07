"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances using NumPy
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    dist_list = dist.tolist()

    station_set = set(stations)

    def simulate_route(route):
        """Simulate route, return (total_distance, is_valid)."""
        t = 0.0
        bat = float(battery_cap)
        load = 0
        total_d = 0.0
        prev = 0
        for node in route:
            d = dist_list[prev][node]
            if bat < d - 1e-9:
                return (float('inf'), False)
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return (float('inf'), False)
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return (float('inf'), False)
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                t += max(0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
            total_d += d
        d = dist_list[prev][0]
        if bat < d - 1e-9:
            return (float('inf'), False)
        t += d / speed
        if t > horizon + 1e-9:
            return (float('inf'), False)
        total_d += d
        return (total_d, True)

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist_list[prev][node]
            prev = node
        total += dist_list[prev][0]
        return total

    def validate_route(route):
        if not route:
            return True
        _, valid = simulate_route(route)
        return valid

    def build_route_from_sequence(cust_seq):
        """
        Given a sequence of customers, build a valid route by inserting stations as needed.
        Returns the route (list of node ids) or None if infeasible.
        """
        if not cust_seq:
            return []

        total_load = sum(demands[c] for c in cust_seq)
        if total_load > capacity:
            return None

        route = []
        t = 0.0
        bat = float(battery_cap)
        load = 0
        prev = 0

        for cust in cust_seq:
            d_direct = dist_list[prev][cust]
            best_t = float('inf')
            best_bat = -1
            best_st = -1
            best_direct = False

            # Option 1: Direct
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr <= due[cust] + 1e-9:
                    t_new = t_arr + service[cust]
                    bat_new = bat - d_direct
                    load_new = load + demands[cust]
                    if load_new <= capacity:
                        if t_new < best_t:
                            best_t = t_new
                            best_bat = bat_new
                            best_st = -1
                            best_direct = True

            # Option 2: Via Station
            for st in stations:
                d_to_st = dist_list[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_cust = dist_list[st][cust]
                if bat_st < d_st_to_cust - 1e-9:
                    continue
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                if t_cust > due[cust] + 1e-9:
                    continue
                t_new = t_cust + service[cust]
                bat_new = bat_st - d_st_to_cust
                load_new = load + demands[cust]
                if load_new <= capacity:
                    if t_new < best_t:
                        best_t = t_new
                        best_bat = bat_new
                        best_st = st
                        best_direct = False

            if best_st == -1 and not best_direct:
                return None

            if best_direct:
                route.append(cust)
            else:
                route.append(best_st)
                route.append(cust)

            t = best_t
            bat = best_bat
            load += demands[cust]
            prev = cust

        # Check return to depot
        d_return = dist_list[prev][0]
        best_t = float('inf')
        best_st = -1
        best_direct = False

        if bat >= d_return - 1e-9:
            t_return = t + d_return / speed
            if t_return <= horizon + 1e-9:
                best_t = t_return
                best_direct = True

        for st in stations:
            d_to_st = dist_list[prev][st]
            if bat < d_to_st - 1e-9:
                continue
            t_st = t + d_to_st / speed
            bat_at_st = bat - d_to_st
            deficit = battery_cap - bat_at_st
            recharge_time = deficit * recharge_rate
            t_st += recharge_time
            bat_st = battery_cap
            d_st_to_depot = dist_list[st][0]
            if bat_st < d_st_to_depot - 1e-9:
                continue
            t_depot = t_st + d_st_to_depot / speed
            if t_depot <= horizon + 1e-9:
                if t_depot < best_t:
                    best_t = t_depot
                    best_st = st
                    best_direct = False

        if best_st == -1 and not best_direct:
            return None

        if not best_direct:
            route.append(best_st)

        return route

    def get_cust_seq(route):
        return [c for c in route if 1 <= c <= n_customers]

    # --- Clarke-Wright Savings Construction ---
    def clarke_wright_savings(randomize=False):
        """Build initial routes using Clarke-Wright savings heuristic."""
        customers = list(range(1, n_customers + 1))

        # Initial: each customer in its own route
        routes = [[c] for c in customers]
        route_of = {c: i for i, c in enumerate(customers)}
        route_custs = {i: [c] for i, c in enumerate(customers)}
        route_loads = {i: demands[c] for i, c in enumerate(customers)}

        # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, n_customers + 1):
            for j in range(i + 1, n_customers + 1):
                s = dist_list[0][i] + dist_list[0][j] - dist_list[i][j]
                if s > 0:
                    savings.append((s, i, j))
        savings.sort(key=lambda x: -x[0])

        if randomize:
            random.shuffle(savings)
            # Re-sort but with some randomness for diversity
            savings.sort(key=lambda x: -x[0] + random.uniform(-0.5, 0.5) * max(s[0] for s in savings) if savings else 0)

        # Try to merge routes based on savings
        for s_val, i, j in savings:
            if time.time() > deadline - 30.0:
                break

            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue

            # Check if i and j are at the ends of their routes
            seq_i = route_custs[ri]
            seq_j = route_custs[rj]

            if len(seq_i) == 0 or len(seq_j) == 0:
                continue

            i_at_start = (seq_i[0] == i)
            i_at_end = (seq_i[-1] == i)
            j_at_start = (seq_j[0] == j)
            j_at_end = (seq_j[-1] == j)

            if not (i_at_start or i_at_end) or not (j_at_start or j_at_end):
                continue

            # Check capacity
            if route_loads[ri] + route_loads[rj] > capacity:
                continue

            # Try all valid merge combinations
            merged = None
            if i_at_start and j_at_end:
                merged = seq_j + seq_i
            elif i_at_end and j_at_start:
                merged = seq_i + seq_j
            elif i_at_start and j_at_start:
                # Both at start: reverse one
                merged = list(reversed(seq_j)) + seq_i
            elif i_at_end and j_at_end:
                # Both at end: reverse one
                merged = seq_i + list(reversed(seq_j))

            if merged is None:
                continue

            # Validate the merged route
            new_route = build_route_from_sequence(merged)
            if new_route is None:
                continue
            if not validate_route(new_route):
                continue

            # Perform the merge
            new_idx = ri  # keep smaller index
            route_custs[new_idx] = merged
            route_loads[new_idx] = route_loads[ri] + route_loads[rj]
            for c in merged:
                route_of[c] = new_idx

            # Remove the other route
            if rj < ri:
                new_idx = rj
                for c in merged:
                    route_of[c] = new_idx
                route_custs.pop(ri, None)
                route_loads.pop(ri, None)
            else:
                route_custs.pop(rj, None)
                route_loads.pop(rj, None)

        # Build final routes
        final_routes = []
        seen = set()
        for idx in sorted(route_custs.keys()):
            if idx in seen:
                continue
            seq = route_custs[idx]
            if not seq:
                continue
            r = build_route_from_sequence(seq)
            if r is not None and validate_route(r):
                final_routes.append(r)
                seen.add(idx)

        # Ensure all customers covered
        covered = set()
        for r in final_routes:
            for c in get_cust_seq(r):
                covered.add(c)

        for c in range(1, n_customers + 1):
            if c not in covered:
                single = build_route_from_sequence([c])
                if single is not None:
                    final_routes.append(single)
                else:
                    final_routes.append([c])

        return final_routes

    def construct_sweep(randomize=False):
        """Sweep heuristic: sort by angle, pack routes."""
        customers = list(range(1, n_customers + 1))
        depot = 0
        angles = []
        for c in customers:
            dx = coords_arr[c][0] - coords_arr[depot][0]
            dy = coords_arr[c][1] - coords_arr[depot][1]
            ang = math.atan2(dy, dx)
            if randomize:
                ang += random.uniform(-0.3, 0.3)
            angles.append((ang, c))
        angles.sort(key=lambda x: x[0])
        sorted_custs = [c for _, c in angles]

        routes = []
        unvisited = set(customers)

        for c in sorted_custs:
            if c not in unvisited:
                continue

            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            best_new_route = None

            for i, r in enumerate(routes):
                cust_seq = get_cust_seq(r)
                for pos in range(len(cust_seq) + 1):
                    new_seq = cust_seq[:pos] + [c] + cust_seq[pos:]
                    if sum(demands[x] for x in new_seq) > capacity:
                        continue
                    new_route = build_route_from_sequence(new_seq)
                    if new_route is None:
                        continue
                    old_cost = route_distance(r)
                    new_cost = route_distance(new_route)
                    delta = new_cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = i
                        best_pos = pos
                        best_new_route = new_route

            if best_route_idx != -1:
                routes[best_route_idx] = best_new_route
                unvisited.remove(c)
            else:
                single_route = build_route_from_sequence([c])
                if single_route is not None:
                    routes.append(single_route)
                else:
                    routes.append([c])
                unvisited.remove(c)

        for c in unvisited:
            single_route = build_route_from_sequence([c])
            if single_route is not None:
                routes.append(single_route)
            else:
                routes.append([c])

        return routes

    def construct_nearest_neighbor(randomize=False):
        """Nearest neighbor construction."""
        customers = list(range(1, n_customers + 1))
        routes = []
        unvisited = set(customers)

        while unvisited:
            if randomize:
                # Pick a random starting customer with some bias toward nearest
                candidates = list(unvisited)
                weights = [1.0 / (dist_list[0][c] + 1) for c in candidates]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                start = candidates[-1]
                for idx, c in enumerate(candidates):
                    cum += weights[idx]
                    if r <= cum:
                        start = c
                        break
            else:
                start = min(unvisited, key=lambda c: dist_list[0][c])

            route_seq = [start]
            unvisited.remove(start)
            current_load = demands[start]
            prev = start

            while True:
                best_next = None
                best_d = float('inf')
                for c in unvisited:
                    if current_load + demands[c] > capacity:
                        continue
                    d = dist_list[prev][c]
                    if d < best_d:
                        test_seq = route_seq + [c]
                        if build_route_from_sequence(test_seq) is not None:
                            best_d = d
                            best_next = c

                if best_next is None:
                    break

                route_seq.append(best_next)
                unvisited.remove(best_next)
                current_load += demands[best_next]
                prev = best_next

            final_route = build_route_from_sequence(route_seq)
            if final_route is not None:
                routes.append(final_route)
            else:
                routes.append(route_seq)

        return routes

    def local_search(routes, time_budget):
        """Apply 2-opt and relocate moves."""
        current_routes = [r[:] for r in routes]
        current_dist = sum(route_distance(r) for r in current_routes)
        improved = True

        while improved and time.time() < time_budget:
            improved = False

            # 1. 2-Opt within routes
            for i in range(len(current_routes)):
                if time.time() > time_budget:
                    break
                cust_seq = get_cust_seq(current_routes[i])
                if len(cust_seq) < 2:
                    continue

                best_seq = cust_seq[:]
                best_route = build_route_from_sequence(cust_seq)
                if best_route is None:
                    continue
                best_d = route_distance(best_route)

                for a in range(len(best_seq)):
                    if time.time() > time_budget:
                        break
                    for b in range(a + 1, len(best_seq)):
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        new_route = build_route_from_sequence(new_seq)
                        if new_route is not None and validate_route(new_route):
                            d = route_distance(new_route)
                            if d < best_d - 1e-6:
                                best_d = d
                                best_seq = new_seq[:]
                                best_route = new_route

                if best_route is not None and best_d < route_distance(current_routes[i]) - 1e-6:
                    current_routes[i] = best_route
                    current_dist = sum(route_distance(r) for r in current_routes)
                    improved = True
                    break

            # 2. Relocate between routes
            if time.time() < time_budget:
                for i in range(len(current_routes)):
                    if time.time() > time_budget:
                        break
                    cust_seq_i = get_cust_seq(current_routes[i])
                    if not cust_seq_i:
                        continue
                    for j_idx, cust in enumerate(cust_seq_i):
                        if time.time() > time_budget:
                            break

                        remaining_seq_i = [c for c in cust_seq_i if c != cust]
                        route_i_new = build_route_from_sequence(remaining_seq_i) if remaining_seq_i else []
                        if route_i_new is None:
                            continue
                        if not validate_route(route_i_new):
                            continue

                        for k in range(len(current_routes)):
                            if k == i:
                                continue
                            if time.time() > time_budget:
                                break
                            cust_seq_k = get_cust_seq(current_routes[k])
                            for pos in range(len(cust_seq_k) + 1):
                                new_seq_k = cust_seq_k[:pos] + [cust] + cust_seq_k[pos:]
                                if sum(demands[x] for x in new_seq_k) > capacity:
                                    continue
                                route_k_new = build_route_from_sequence(new_seq_k)
                                if route_k_new is None:
                                    continue
                                if not validate_route(route_k_new):
                                    continue

                                new_total = route_distance(route_i_new) + route_distance(route_k_new)
                                old_total = route_distance(current_routes[i]) + route_distance(current_routes[k])

                                if new_total < old_total - 1e-6:
                                    current_routes[i] = route_i_new
                                    current_routes[k] = route_k_new
                                    current_dist = sum(route_distance(r) for r in current_routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return current_routes

    # --- Multi-start with diverse constructions ---
    solutions = []

    # Deterministic constructions
    if time.time() < deadline - 40.0:
        sol = clarke_wright_savings(randomize=False)
        solutions.append(sol)

    if time.time() < deadline - 40.0:
        sol = construct_sweep(randomize=False)
        solutions.append(sol)

    if time.time() < deadline - 40.0:
        sol = construct_nearest_neighbor(randomize=False)
        solutions.append(sol)

    # Randomized constructions for diversity
    random.seed(42)
    n_random = 5
    for _ in range(n_random):
        if time.time() > deadline - 40.0:
            break
        sol = clarke_wright_savings(randomize=True)
        solutions.append(sol)

    if time.time() < deadline - 40.0:
        sol = construct_sweep(randomize=True)
        solutions.append(sol)

    if time.time() < deadline - 40.0:
        sol = construct_nearest_neighbor(randomize=True)
        solutions.append(sol)

    # Select best initial solution
    best_routes = None
    best_dist = float('inf')
    for sol in solutions:
        valid = True
        total_d = 0
        covered = set()
        for r in sol:
            if not validate_route(r):
                valid = False
                break
            total_d += route_distance(r)
            for c in get_cust_seq(r):
                if c in covered:
                    valid = False
                    break
                covered.add(c)
            if not valid:
                break

        if valid and covered == set(range(1, n_customers + 1)) and total_d < best_dist:
            best_dist = total_d
            best_routes = [r[:] for r in sol]

    if best_routes is None:
        # Fallback
        best_routes = []
        for c in range(1, n_customers + 1):
            best_routes.append([c])
        best_dist = sum(route_distance(r) for r in best_routes)

    # --- Local search on best solution ---
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        best_routes = local_search(best_routes, deadline - 5.0)
        best_dist = sum(route_distance(r) for r in best_routes)

    # --- Second round of local search if time permits ---
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        # Try local search on second-best initial solution too
        second_best = None
        second_dist = float('inf')
        for sol in solutions:
            if sol == best_routes:
                continue
            valid = True
            total_d = 0
            covered = set()
            for r in sol:
                if not validate_route(r):
                    valid = False
                    break
                total_d += route_distance(r)
                for c in get_cust_seq(r):
                    if c in covered:
                        valid = False
                        break
                    covered.add(c)
                if not valid:
                    break
            if valid and covered == set(range(1, n_customers + 1)) and total_d < second_dist and total_d > best_dist:
                second_dist = total_d
                second_best = [r[:] for r in sol]

        if second_best is not None:
            lb = local_search(second_best, deadline - 5.0)
            lb_dist = sum(route_distance(r) for r in lb)
            if lb_dist < best_dist:
                best_routes = lb
                best_dist = lb_dist

    # Final validation
    covered = set()
    final_routes = []
    for r in best_routes:
        if not validate_route(r):
            continue
        custs_in_r = get_cust_seq(r)
        if any(c in covered for c in custs_in_r):
            continue
        covered.update(custs_in_r)
        final_routes.append(r)

    for c in range(1, n_customers + 1):
        if c not in covered:
            single_route = build_route_from_sequence([c])
            if single_route is not None and validate_route(single_route):
                final_routes.append(single_route)
                covered.add(c)
            else:
                final_routes.append([c])
                covered.add(c)

    return final_routes
# EVOLVE-BLOCK-END
