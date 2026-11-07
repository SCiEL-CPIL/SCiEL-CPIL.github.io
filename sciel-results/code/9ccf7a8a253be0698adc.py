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

    def eval_route(route):
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

    def fast_eval(route):
        """Fast feasibility + cost check."""
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

    def try_fix_with_stations(route, max_extra=2):
        """Try inserting up to max_extra stations to make route feasible."""
        _, valid, _, _ = eval_route(route)
        if valid:
            return route
        best = None
        best_cost = float('inf')
        # Try 1 station
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = eval_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best = cand
        if best:
            return best
        if max_extra >= 2:
            for s1 in stations:
                for i1 in range(len(route) + 1):
                    r1 = route[:i1] + [s1] + route[i1:]
                    for s2 in stations:
                        for i2 in range(len(r1) + 1):
                            cand = r1[:i2] + [s2] + r1[i2:]
                            c, v, _, _ = eval_route(cand)
                            if v and c < best_cost:
                                best_cost = c
                                best = cand
        return best

    def total_cost(routes):
        return sum(fast_eval(r)[0] for r in routes)

    def all_valid(routes):
        return all(fast_eval(r)[1] for r in routes)

    # --- Construction: sweep-based ---
    def construct_sweep():
        """Angular sweep from depot to partition customers into sectors, then build routes."""
        angles = []
        for c in cust_ids:
            dx = cx[c] - cx[0]
            dy = cy[c] - cy[0]
            a = math.atan2(dy, dx)
            if a < 0:
                a += 2 * math.pi
            angles.append((a, c))
        angles.sort()
        
        # Try different numbers of routes
        best_routes = None
        best_cost = float('inf')
        
        for n_routes in range(3, 9):
            if time.time() > DEADLINE:
                break
            # Partition by angle into n_routes groups
            sector_size = 2 * math.pi / n_routes
            groups = [[] for _ in range(n_routes)]
            for a, c in angles:
                idx = int(a / sector_size) % n_routes
                groups[idx].append(c)
            
            routes = []
            for g in groups:
                if not g:
                    continue
                # Sort within group by distance from depot
                g.sort(key=lambda x: dist[0, x])
                route = g[:]
                # Try to make feasible with stations
                fixed = try_fix_with_stations(route)
                if fixed:
                    routes.append(fixed)
                else:
                    # Split into individual routes
                    for c in g:
                        ind = [c]
                        fixed_ind = try_fix_with_stations(ind)
                        if fixed_ind:
                            routes.append(fixed_ind)
                        else:
                            routes.append(ind)
            
            if all_valid(routes):
                c = total_cost(routes)
                if c < best_cost:
                    best_cost = c
                    best_routes = [list(r) for r in routes]
        
        return best_routes

    # --- Construction: Clarke-Wright savings ---
    def construct_savings():
        routes = [[c] for c in cust_ids]
        savings = []
        for i in cust_ids:
            for j in cust_ids:
                if i != j:
                    s = dist[0, i] + dist[0, j] - dist[i, j]
                    savings.append((s, i, j))
        savings.sort(reverse=True)
        
        route_of = {}
        for idx, r in enumerate(routes):
            for c in r:
                route_of[c] = idx
        
        for s, i, j in savings:
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
            _, v = fast_eval(cand)
            if not v:
                fixed = try_fix_with_stations(cand)
                if fixed:
                    cand = fixed
                    _, v = fast_eval(cand)
            if v:
                routes[ri] = cand
                for c in route_j:
                    route_of[c] = ri
                routes[rj] = []
        
        return [r for r in routes if r]

    # --- Construction: nearest neighbor ---
    def construct_nn():
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
                        cc, v = fast_eval(cand)
                        if not v:
                            fixed = try_fix_with_stations(cand)
                            if fixed:
                                cc, v = fast_eval(fixed)
                                if v:
                                    cand = fixed
                        if v and cc < best_cost:
                            best_cost = cc
                            best_c = c
                            best_pos = pos
                            best_cand = cand
                if best_c is not None:
                    route = best_cand
                    remaining.remove(best_c)
                    improved = True
            routes.append(route)
        return routes

    # --- Local search: 2-opt intra-route ---
    def two_opt(route):
        if len(route) < 2:
            return route, fast_eval(route)[0]
        best_route = list(route)
        best_cost, _ = fast_eval(best_route)
        n = len(best_route)
        improved = True
        max_iter = 20
        while improved and max_iter > 0 and time.time() < DEADLINE:
            improved = False
            max_iter -= 1
            for i in range(n - 1):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v = fast_eval(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        return best_route, best_cost

    # --- Local search: relocate (move one customer to another position/route) ---
    def relocate(routes):
        improved_any = False
        for _ in range(10):
            if time.time() > DEADLINE:
                break
            improved = False
            for ri in range(len(routes)):
                for pos in range(len(routes[ri])):
                    if time.time() > DEADLINE:
                        break
                    node = routes[ri][pos]
                    if not is_customer(node):
                        continue
                    # Try moving to other positions in same route
                    for new_pos in range(len(routes[ri])):
                        if new_pos == pos:
                            continue
                        route = routes[ri]
                        new_route = route[:pos] + route[pos+1:]
                        new_route = new_route[:new_pos] + [node] + new_route[new_pos:]
                        c, v = fast_eval(new_route)
                        if v:
                            old_c, _ = fast_eval(route)
                            if c < old_c - 1e-7:
                                routes[ri] = new_route
                                improved = True
                                improved_any = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Try inter-route relocate
                for ri in range(len(routes)):
                    for pos in range(len(routes[ri])):
                        if time.time() > DEADLINE:
                            break
                        node = routes[ri][pos]
                        if not is_customer(node):
                            continue
                        for rj in range(len(routes)):
                            if rj == ri:
                                continue
                            load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
                            if load_j + demands[node] > capacity + _EPS:
                                continue
                            for new_pos in range(len(routes[rj]) + 1):
                                new_route_i = routes[ri][:pos] + routes[ri][pos+1:]
                                new_route_j = routes[rj][:new_pos] + [node] + routes[rj][new_pos:]
                                ci, vi = fast_eval(new_route_i)
                                cj, vj = fast_eval(new_route_j)
                                if vi and vj:
                                    old_ci, _ = fast_eval(routes[ri])
                                    old_cj, _ = fast_eval(routes[rj])
                                    if ci + cj < old_ci + old_cj - 1e-7:
                                        routes[ri] = new_route_i
                                        routes[rj] = new_route_j
                                        improved = True
                                        improved_any = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return routes, improved_any

    # --- Local search: swap (exchange two customers between positions) ---
    def swap_customers(routes):
        improved_any = False
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            improved = False
            for ri in range(len(routes)):
                for rj in range(len(routes)):
                    if rj <= ri:
                        continue
                    for pi in range(len(routes[ri])):
                        if time.time() > DEADLINE:
                            break
                        ni = routes[ri][pi]
                        if not is_customer(ni):
                            continue
                        for pj in range(len(routes[rj])):
                            nj = routes[rj][pj]
                            if not is_customer(nj):
                                continue
                            # Swap
                            new_ri = list(routes[ri])
                            new_rj = list(routes[rj])
                            new_ri[pi] = nj
                            new_rj[pj] = ni
                            ci, vi = fast_eval(new_ri)
                            cj, vj = fast_eval(new_rj)
                            if vi and vj:
                                old_ci, _ = fast_eval(routes[ri])
                                old_cj, _ = fast_eval(routes[rj])
                                if ci + cj < old_ci + old_cj - 1e-7:
                                    routes[ri] = new_ri
                                    routes[rj] = new_rj
                                    improved = True
                                    improved_any = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes, improved_any

    # --- Tabu search ---
    def tabu_search(initial_routes, tabu_tenure=7, max_iter=2000):
        current = [list(r) for r in initial_routes]
        current_cost = total_cost(current)
        best = [list(r) for r in current]
        best_cost = current_cost
        
        # Tabu list: (node, route_idx, pos) for relocate, (node1, node2) for swap
        tabu_relocate = {}  # node -> (iteration, route_idx, pos)
        tabu_swap = {}      # (node1, node2) -> iteration
        iter_count = 0
        
        while iter_count < max_iter and time.time() < DEADLINE:
            iter_count += 1
            best_move_cost = current_cost
            best_move = None
            best_move_type = None
            best_move_data = None
            
            # Generate candidate moves
            # 1. 2-opt within each route
            for ri in range(len(current)):
                route = current[ri]
                n = len(route)
                if n < 2:
                    continue
                for i in range(n - 1):
                    if time.time() > DEADLINE:
                        break
                    for j in range(i + 1, n):
                        cand = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        c, v = fast_eval(cand)
                        if v:
                            delta = c - fast_eval(route)[0]
                            if delta < best_move_cost - current_cost + 1e-7:
                                best_move_cost = c
                                best_move_type = '2opt'
                                best_move_data = (ri, i, j)
            
            # 2. Relocate: move customer from route ri to route rj
            for ri in range(len(current)):
                for pos in range(len(current[ri])):
                    if time.time() > DEADLINE:
                        break
                    node = current[ri][pos]
                    if not is_customer(node):
                        continue
                    # Tabu check
                    if node in tabu_relocate:
                        if iter_count - tabu_relocate[node][0] < tabu_tenure:
                            continue
                    for rj in range(len(current)):
                        if rj == ri:
                            continue
                        load_j = sum(demands[n] for n in current[rj] if is_customer(n))
                        if load_j + demands[node] > capacity + _EPS:
                            continue
                        for new_pos in range(len(current[rj]) + 1):
                            new_ri = current[ri][:pos] + current[ri][pos+1:]
                            new_rj = current[rj][:new_pos] + [node] + current[rj][new_pos:]
                            ci, vi = fast_eval(new_ri)
                            cj, vj = fast_eval(new_rj)
                            if vi and vj:
                                new_total = ci + cj
                                old_total = fast_eval(current[ri])[0] + fast_eval(current[rj])[0]
                                delta = new_total - old_total
                                if delta < best_move_cost - current_cost + 1e-7:
                                    best_move_cost = new_total
                                    best_move_type = 'relocate'
                                    best_move_data = (ri, pos, rj, new_pos)
            
            # 3. Swap: exchange customers between routes
            for ri in range(len(current)):
                for rj in range(len(current)):
                    if rj <= ri:
                        continue
                    for pi in range(len(current[ri])):
                        if time.time() > DEADLINE:
                            break
                        ni = current[ri][pi]
                        if not is_customer(ni):
                            continue
                        for pj in range(len(current[rj])):
                            nj = current[rj][pj]
                            if not is_customer(nj):
                                continue
                            key = (min(ni, nj), max(ni, nj))
                            if key in tabu_swap:
                                if iter_count - tabu_swap[key] < tabu_tenure:
                                    continue
                            new_ri = list(current[ri])
                            new_rj = list(current[rj])
                            new_ri[pi] = nj
                            new_rj[pj] = ni
                            ci, vi = fast_eval(new_ri)
                            cj, vj = fast_eval(new_rj)
                            if vi and vj:
                                new_total = ci + cj
                                old_total = fast_eval(current[ri])[0] + fast_eval(current[rj])[0]
                                delta = new_total - old_total
                                if delta < best_move_cost - current_cost + 1e-7:
                                    best_move_cost = new_total
                                    best_move_type = 'swap'
                                    best_move_data = (ri, pi, rj, pj)
            
            # Apply best move
            if best_move_type is None:
                break
            
            if best_move_type == '2opt':
                ri, i, j = best_move_data
                route = current[ri]
                current[ri] = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
            elif best_move_type == 'relocate':
                ri, pos, rj, new_pos = best_move_data
                node = current[ri][pos]
                current[ri] = current[ri][:pos] + current[ri][pos+1:]
                current[rj] = current[rj][:new_pos] + [node] + current[rj][new_pos:]
                tabu_relocate[node] = (iter_count, rj, new_pos)
            elif best_move_type == 'swap':
                ri, pi, rj, pj = best_move_data
                ni = current[ri][pi]
                nj = current[rj][pj]
                current[ri][pi] = nj
                current[rj][pj] = ni
                key = (min(ni, nj), max(ni, nj))
                tabu_swap[key] = iter_count
            
            current_cost = total_cost(current)
            if current_cost < best_cost - 1e-7:
                best_cost = current_cost
                best = [list(r) for r in current]
        
        return best, best_cost

    # --- Main ---
    # Build multiple initial solutions
    candidates = []
    
    r1 = construct_sweep()
    if r1 and all_valid(r1):
        candidates.append(r1)
    
    if time.time() < DEADLINE - 10:
        r2 = construct_savings()
        if r2 and all_valid(r2):
            candidates.append(r2)
    
    if time.time() < DEADLINE - 10:
        r3 = construct_nn()
        if r3 and all_valid(r3):
            candidates.append(r3)
    
    if not candidates:
        # Fallback
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        candidates.append([[v] for v in customers])
    
    # Pick best initial
    best_sol = min(candidates, key=total_cost)
    best_cost = total_cost(best_sol)
    
    # Local improvement on best
    for i in range(len(best_sol)):
        if time.time() > DEADLINE - 15:
            break
        best_sol[i], _ = two_opt(best_sol[i])
    best_cost = total_cost(best_sol)
    
    # Tabu search
    if time.time() < DEADLINE - 5:
        tabu_result, tabu_cost = tabu_search(best_sol, tabu_tenure=7, max_iter=1500)
        if tabu_cost < best_cost:
            best_sol = tabu_result
            best_cost = tabu_cost
    
    # Final local polish
    for i in range(len(best_sol)):
        if time.time() > DEADLINE - 2:
            break
        best_sol[i], _ = two_opt(best_sol[i])
    
    # Final validation
    served = set()
    valid = True
    for r in best_sol:
        c, v, _, _ = eval_route(r)
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
        best_sol = [[v] for v in customers]
        for i in range(len(best_sol)):
            if not eval_route(best_sol[i])[1]:
                fixed = try_fix_with_stations(best_sol[i])
                if fixed:
                    best_sol[i] = fixed
    
    return best_sol
