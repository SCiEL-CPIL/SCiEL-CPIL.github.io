"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

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

    # Precompute pairwise distances using NumPy vectorization
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    station_set = set(stations)

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    def validate_route(route):
        if not route:
            return True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        for node in route:
            d = dist[prev][node]
            if bat < d - 1e-9:
                return False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return False
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                t += max(0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
        d = dist[prev][0]
        if bat < d - 1e-9:
            return False
        t += d / speed
        if t > horizon + 1e-9:
            return False
        return True

    def build_route_from_sequence(cust_seq):
        """
        Given a sequence of customers, build a valid route by inserting stations as needed.
        Tries ALL stations and picks the one minimizing total time+distance.
        Returns the route (list of node ids) or None if infeasible.
        """
        if not cust_seq:
            return []
        
        total_load = sum(demands[c] for c in cust_seq)
        if total_load > capacity:
            return None
        
        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        
        for cust in cust_seq:
            d_direct = dist[prev][cust]
            if bat >= d_direct:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust]:
                    return None
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                route.append(cust)
                prev = cust
            else:
                # Need to visit a station; try all and pick best
                best_st = None
                best_total_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0
                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    deficit = battery_cap - bat_at_st
                    recharge_time = deficit * recharge_rate
                    t_st += recharge_time
                    bat_st = battery_cap
                    d_st_to_cust = dist[st][cust]
                    if bat_st < d_st_to_cust:
                        continue
                    t_cust = t_st + d_st_to_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust]:
                        continue
                    # Cost metric: total distance detour + recharge time
                    cost = d_to_st + d_st_to_cust + recharge_time
                    if cost < best_total_cost:
                        best_total_cost = cost
                        best_st = st
                        best_t_cust = t_cust
                        best_bat_after = bat_st - d_st_to_cust
                if best_st is None:
                    return None
                t = best_t_cust + service[cust]
                bat = best_bat_after
                load += demands[cust]
                route.append(best_st)
                route.append(cust)
                prev = cust
        
        # Check return to depot
        d_return = dist[prev][0]
        if bat < d_return:
            found = False
            best_st = None
            best_cost = float('inf')
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_depot = dist[st][0]
                if bat_st < d_st_to_depot:
                    continue
                t_depot = t_st + d_st_to_depot / speed
                if t_depot > horizon:
                    continue
                cost = d_to_st + d_st_to_depot + recharge_time
                if cost < best_cost:
                    best_cost = cost
                    best_st = st
            if best_st is None:
                return None
            route.append(best_st)
        else:
            t_return = t + d_return / speed
            if t_return > horizon:
                return None
        
        return route

    # Fast feasibility check without building full route
    def quick_feasible(prev_node, t, bat, cust):
        """Quick check if we can reach cust from prev_node with given state."""
        d = dist[prev_node][cust]
        if bat >= d:
            t_arr = t + d / speed
            if t_arr < ready[cust]:
                t_arr = ready[cust]
            return t_arr <= due[cust]
        else:
            for st in stations:
                d_to_st = dist[prev_node][st]
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                d_st_to_cust = dist[st][cust]
                if battery_cap < d_st_to_cust:
                    continue
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                if t_cust <= due[cust]:
                    return True
        return False

    # Greedy construction with better heuristics
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    routes = []
    remaining = set(customers)

    while remaining and time.time() < deadline - 3.0:
        best_route = None
        best_dist = float('inf')
        best_used = None

        # Try multiple starting points
        start_candidates = sorted(remaining, key=lambda v: due[v])
        # Also try customers closest to depot
        start_candidates += sorted(remaining, key=lambda v: dist[0][v])
        # Remove duplicates while preserving order
        seen_starts = set()
        unique_starts = []
        for s in start_candidates:
            if s not in seen_starts:
                seen_starts.add(s)
                unique_starts.append(s)
        start_candidates = unique_starts[:8]

        for start_cust in start_candidates:
            route_seq = [start_cust]
            used = {start_cust}
            current_load = demands[start_cust]

            while True:
                best_next = None
                best_metric = float('inf')

                for c in remaining - used:
                    if current_load + demands[c] > capacity:
                        continue
                    # Quick feasibility check
                    last = route_seq[-1]
                    # Estimate state at end of current route
                    # We'll do a quick simulation
                    t_est = 0.0
                    bat_est = battery_cap
                    for nc in route_seq:
                        d_prev = dist[0 if nc == route_seq[0] else route_seq[route_seq.index(nc)-1]][nc] if route_seq.index(nc) > 0 else dist[0][nc]
                        if bat_est >= d_prev:
                            t_est += d_prev / speed
                            bat_est -= d_prev
                        else:
                            # Approximate: go through nearest station
                            st = min(stations, key=lambda s: dist[route_seq[route_seq.index(nc)-1]][s] if route_seq.index(nc) > 0 else dist[0][s])
                            d_st = dist[route_seq[route_seq.index(nc)-1]][st] if route_seq.index(nc) > 0 else dist[0][st]
                            t_est += d_st / speed
                            bat_at_st = bat_est - d_st
                            deficit = battery_cap - bat_at_st
                            t_est += deficit * recharge_rate
                            bat_est = battery_cap
                            d_to_nc = dist[st][nc]
                            t_est += d_to_nc / speed
                            bat_est -= d_to_nc
                        t_est = max(t_est, ready[nc])
                        t_est += service[nc]
                    
                    if t_est > due[c] + 50:  # Too late even approximately
                        continue
                    if not quick_feasible(last, 0, battery_cap, c):
                        # More detailed check needed; skip for now, try full build
                        pass

                    new_seq = route_seq + [c]
                    test_route = build_route_from_sequence(new_seq)
                    if test_route is not None:
                        d = route_distance(test_route)
                        if d < best_metric:
                            best_metric = d
                            best_next = c

                if best_next is None:
                    break
                route_seq.append(best_next)
                used.add(best_next)
                current_load += demands[best_next]

            final_route = build_route_from_sequence(route_seq)
            if final_route is not None and validate_route(final_route):
                d = route_distance(final_route)
                if d < best_dist:
                    best_dist = d
                    best_route = final_route
                    best_used = set(c for c in route_seq if 1 <= c <= n_customers)

        if best_route is None:
            for c in remaining:
                single_route = build_route_from_sequence([c])
                if single_route is not None and validate_route(single_route):
                    best_route = single_route
                    best_used = {c}
                    break
            if best_route is None:
                best_route = [list(remaining)[0]]
                best_used = {list(remaining)[0]}

        routes.append(best_route)
        remaining -= best_used

    for c in remaining:
        single_route = build_route_from_sequence([c])
        if single_route is not None:
            routes.append(single_route)
        else:
            routes.append([c])

    # Local search: relocate + 2-opt + Or-opt + LNS
    if time.time() < deadline - 5.0:
        route_dists = [route_distance(r) for r in routes]

        # Relocate moves
        improved = True
        max_iter = 20
        it = 0
        while improved and it < max_iter and time.time() < deadline - 3.0:
            improved = False
            it += 1
            for i in range(len(routes)):
                if time.time() > deadline - 3.0:
                    break
                custs_in_i = [c for c in routes[i] if 1 <= c <= n_customers]
                for cust in custs_in_i:
                    if time.time() > deadline - 3.0:
                        break
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 3.0:
                            continue
                        load_k = sum(demands[c] for c in routes[k] if 1 <= c <= n_customers)
                        if load_k + demands[cust] > capacity:
                            continue
                        custs_in_k = [c for c in routes[k] if 1 <= c <= n_customers]
                        for pos in range(len(custs_in_k) + 1):
                            if time.time() > deadline - 3.0:
                                break
                            new_seq_i = [c for c in custs_in_i if c != cust]
                            new_seq_k = custs_in_k[:pos] + [cust] + custs_in_k[pos:]
                            r_i = build_route_from_sequence(new_seq_i) if new_seq_i else []
                            r_k = build_route_from_sequence(new_seq_k)
                            if r_i is None or r_k is None:
                                continue
                            if not validate_route(r_i) or not validate_route(r_k):
                                continue
                            new_dist = route_distance(r_i) + route_distance(r_k)
                            if new_dist < route_dists[i] + route_dists[k] - 0.1:
                                routes[i] = r_i
                                routes[k] = r_k
                                route_dists[i] = route_distance(r_i)
                                route_dists[k] = route_distance(r_k)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        # 2-opt within routes
        if time.time() < deadline - 3.0:
            for i in range(len(routes)):
                if time.time() > deadline - 3.0:
                    break
                cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(cust_seq) < 2:
                    continue
                current_route = build_route_from_sequence(cust_seq)
                if current_route is None:
                    continue
                best_d = route_distance(current_route)
                best_seq = cust_seq[:]
                improved_route = False
                for a in range(len(best_seq)):
                    if time.time() > deadline - 3.0:
                        break
                    for b in range(a + 1, len(best_seq)):
                        if time.time() > deadline - 3.0:
                            break
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        new_route = build_route_from_sequence(new_seq)
                        if new_route is not None and validate_route(new_route):
                            d = route_distance(new_route)
                            if d < best_d - 0.1:
                                best_d = d
                                best_seq = new_seq[:]
                                improved_route = True
                if improved_route:
                    new_route = build_route_from_sequence(best_seq)
                    if new_route is not None and validate_route(new_route):
                        routes[i] = new_route
                        route_dists[i] = route_distance(new_route)

        # Or-opt moves: relocate a segment of 1-3 customers
        if time.time() < deadline - 3.0:
            for i in range(len(routes)):
                if time.time() > deadline - 3.0:
                    break
                custs_in_i = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(custs_in_i) < 2:
                    continue
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline - 3.0:
                        break
                    for pos in range(len(custs_in_i) - seg_len + 1):
                        if time.time() > deadline - 3.0:
                            break
                        segment = custs_in_i[pos:pos+seg_len]
                        remaining_in_i = custs_in_i[:pos] + custs_in_i[pos+seg_len:]
                        
                        # Try inserting into other routes
                        for k in range(len(routes)):
                            if k == i or time.time() > deadline - 3.0:
                                continue
                            load_k = sum(demands[c] for c in routes[k] if 1 <= c <= n_customers)
                            if load_k + sum(demands[c] for c in segment) > capacity:
                                continue
                            custs_in_k = [c for c in routes[k] if 1 <= c <= n_customers]
                            for insert_pos in range(len(custs_in_k) + 1):
                                if time.time() > deadline - 3.0:
                                    break
                                new_seq_i = remaining_in_i
                                new_seq_k = custs_in_k[:insert_pos] + segment + custs_in_k[insert_pos:]
                                r_i = build_route_from_sequence(new_seq_i) if new_seq_i else []
                                r_k = build_route_from_sequence(new_seq_k)
                                if r_i is None or r_k is None:
                                    continue
                                if not validate_route(r_i) or not validate_route(r_k):
                                    continue
                                new_dist = route_distance(r_i) + route_distance(r_k)
                                if new_dist < route_dists[i] + route_dists[k] - 0.1:
                                    routes[i] = r_i
                                    routes[k] = r_k
                                    route_dists[i] = route_distance(r_i)
                                    route_dists[k] = route_distance(r_k)
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

        # LNS: Large Neighborhood Search
        if time.time() < deadline - 5.0:
            lns_iter = 0
            max_lns_iter = 30
            while lns_iter < max_lns_iter and time.time() < deadline - 3.0:
                lns_iter += 1
                
                # Destroy: remove a random subset of customers from routes
                destroy_count = max(3, len(customers) // 5)
                destroy_count = min(destroy_count, len(customers) - 2)
                
                # Randomly select customers to destroy
                all_custs = [c for r in routes for c in r if 1 <= c <= n_customers]
                if len(all_custs) < destroy_count + 2:
                    break
                
                # Weighted random selection based on due times (tighter windows more likely)
                weights = [1.0 / max(1, due[c] - ready[c]) for c in all_custs]
                total_w = sum(weights)
                selected = set()
                for _ in range(destroy_count):
                    r = random.random() * total_w
                    cum = 0
                    for c, w in zip(all_custs, weights):
                        cum += w
                        if cum >= r:
                            selected.add(c)
                            break
                    else:
                        selected.add(all_custs[-1])
                
                # Remove selected customers from routes
                remaining_lns = set(selected)
                for i in range(len(routes)):
                    routes[i] = [c for c in routes[i] if c not in remaining_lns]
                    # Rebuild route if needed
                    cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
                    if cust_seq:
                        new_r = build_route_from_sequence(cust_seq)
                        if new_r is not None:
                            routes[i] = new_r
                        else:
                            routes[i] = []
                    else:
                        routes[i] = []
                
                # Remove empty routes
                routes = [r for r in routes if r]
                
                # Re-insert customers using regret-based insertion
                for cust in sorted(remaining_lns, key=lambda c: due[c] - ready[c]):
                    best_pos = None
                    best_route_idx = -1
                    best_cost = float('inf')
                    
                    for k in range(len(routes)):
                        load_k = sum(demands[c] for c in routes[k] if 1 <= c <= n_customers)
                        if load_k + demands[cust] > capacity:
                            continue
                        custs_in_k = [c for c in routes[k] if 1 <= c <= n_customers]
                        for pos in range(len(custs_in_k) + 1):
                            new_seq = custs_in_k[:pos] + [cust] + custs_in_k[pos:]
                            r = build_route_from_sequence(new_seq)
                            if r is None:
                                continue
                            if not validate_route(r):
                                continue
                            d = route_distance(r)
                            if d < best_cost:
                                best_cost = d
                                best_pos = pos
                                best_route_idx = k
                    
                    if best_route_idx >= 0:
                        custs_in_k = [c for c in routes[best_route_idx] if 1 <= c <= n_customers]
                        new_seq = custs_in_k[:best_pos] + [cust] + custs_in_k[best_pos:]
                        new_r = build_route_from_sequence(new_seq)
                        if new_r is not None:
                            routes[best_route_idx] = new_r
                        remaining_lns.discard(cust)
                    else:
                        # Create new route
                        single = build_route_from_sequence([cust])
                        if single is not None:
                            routes.append(single)
                        else:
                            routes.append([cust])
                        remaining_lns.discard(cust)
                
                # Accept if improved or same (to allow exploration)
                total_dist = sum(route_distance(r) for r in routes)
                if total_dist <= sum(route_dists) + 0.5:  # Allow slight worsening for exploration
                    route_dists = [route_distance(r) for r in routes]

    # Final validation
    valid_routes = []
    covered = set()
    for r in routes:
        if validate_route(r):
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    for cust in range(1, n_customers + 1):
        if cust not in covered:
            single_route = build_route_from_sequence([cust])
            if single_route is not None:
                valid_routes.append(single_route)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
# EVOLVE-BLOCK-END
