import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 149.0  # Safety margin for 150s kill

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

    # Node IDs: 0 depot, 1..n_cust customers, then stations
    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    station_set = station_ids

    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    # Vectorized distance computation
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_set

    # Precompute node type arrays for fast lookup
    is_cust_arr = np.zeros(n_nodes, dtype=bool)
    is_cust_arr[1:n_cust+1] = True
    is_stat_arr = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_stat_arr[s] = True

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        """
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
        
        # Check capacity
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
                # Recharge
                t += (bmax_l - b) * rr_l
                b = bmax_l
            elif is_cust_l[node]:
                if t < ready_l[node]:
                    t = ready_l[node]
                if t > due_l[node] + eps_l:
                    return float('inf'), False, 0.0, 0.0
                t += service_l[node]
            
            prev = node
            
        # Return to depot
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
        return sum(demands[n] for n in route if is_customer(n))

    # --- Construction: Greedy Insertion ---
    
    # Start with one customer per route
    routes = [[c] for c in cust_ids]
    
    active = [True] * len(routes)
    
    # Pre-compute costs for initial routes
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
            if not active[i]: continue
            if route_costs[i] == float('inf'): continue
            
            last_i = routes[i][-1]
            
            for j in range(i + 1, len(routes)):
                if not active[j]: continue
                if route_costs[j] == float('inf'): continue
                
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
    
    # Quick 2-opt on each route after construction
    for ri in range(len(routes)):
        if time.time() > DEADLINE: break
        n = len(routes[ri])
        if n < 2: continue
        cur_cost, cur_valid, _, _ = check_route(routes[ri])
        if not cur_valid: continue
        best_route = list(routes[ri])
        best_cost = cur_cost
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        if best_cost < route_costs[ri] - 1e-7:
            routes[ri] = best_route
            route_costs[ri] = best_cost
    
    # --- Local Search ---
    
    route_costs = [check_route(r)[0] for r in routes]

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

    def optimize_route(route):
        """Apply 2-opt to a single route to minimize distance while maintaining feasibility."""
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
        
        # First pass: standard 2-opt
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        
        # Second pass: Or-opt (move 1-3 consecutive nodes)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr):
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

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route optimization (2-opt + or-opt)
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            
            new_route, new_cost = optimize_route(route)
            
            if new_cost < route_costs[ri] - 1e-7:
                routes[ri] = new_route
                route_costs[ri] = new_cost
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        relocate_done = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or relocate_done: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE or relocate_done: break
                cust = routes[ri][ci]
                
                if not is_customer(cust):
                    continue
                    
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > DEADLINE or relocate_done: continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE or relocate_done: break
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
                    if relocate_done: break
                if relocate_done: break
            if relocate_done: break
        
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
        if time.time() > DEADLINE: break
        swap_done = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or swap_done: break
            for rj in range(ri + 1, len(routes)):
                if time.time() > DEADLINE or swap_done: break
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE or swap_done: break
                    cust1 = routes[ri][ci]
                    if not is_customer(cust1): continue
                    for cj in range(len(routes[rj])):
                        if time.time() > DEADLINE or swap_done: break
                        cust2 = routes[rj][cj]
                        if not is_customer(cust2): continue
                        
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
                if swap_done: break
            if swap_done: break
        
        # 4. Inter-route 2-opt (cross-route edge swap)
        if time.time() > DEADLINE: break
        cross2opt_done = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or cross2opt_done: break
            for rj in range(ri + 1, len(routes)):
                if time.time() > DEADLINE or cross2opt_done: break
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE or cross2opt_done: break
                    node1 = routes[ri][ci]
                    if not is_customer(node1): continue
                    for cj in range(len(routes[rj])):
                        if time.time() > DEADLINE or cross2opt_done: break
                        node2 = routes[rj][cj]
                        if not is_customer(node2): continue
                        
                        # Try swapping the suffixes
                        if ci >= len(routes[ri]) - 1 or cj >= len(routes[rj]) - 1:
                            continue
                            
                        new_route_i = routes[ri][:ci+1] + routes[rj][cj+1:]
                        new_route_j = routes[rj][:cj+1] + routes[ri][ci+1:]
                        
                        load_i = route_load(new_route_i)
                        load_j = route_load(new_route_j)
                        if load_i > capacity + _EPS or load_j > capacity + _EPS:
                            continue
                        
                        c_i, v_i, _, _ = check_route(new_route_i)
                        if not v_i: continue
                        c_j, v_j, _, _ = check_route(new_route_j)
                        if not v_j: continue
                        
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
                if cross2opt_done: break
            if cross2opt_done: break

    # LNS (Large Neighborhood Search) to escape local optima
    if time.time() < DEADLINE:
        import random as _rnd
        _rnd.seed(42)
        lns_iter = 0
        while time.time() < DEADLINE - 5.0 and lns_iter < 200:
            lns_iter += 1
            # Randomly destroy: remove k customers from random positions
            k = _rnd.randint(3, min(8, max(3, len(routes[0]) // 2)))
            removed = []
            for _ in range(k):
                ri = _rnd.randint(0, len(routes) - 1)
                if not routes[ri]:
                    continue
                ci = _rnd.randint(0, len(routes[ri]) - 1)
                cust = routes[ri][ci]
                if not is_customer(cust):
                    continue
                del routes[ri][ci]
                removed.append((cust, ri, ci))
            
            if not removed:
                continue
            
            # Reinsert removed customers using cheapest insertion
            for cust, _, _ in removed:
                best_cost_inf = float('inf')
                best_ri = -1
                best_pos = -1
                for ri in range(len(routes)):
                    for pos in range(len(routes[ri]) + 1):
                        cand = routes[ri][:pos] + [cust] + routes[ri][pos:]
                        load = route_load(routes[ri]) + demands[cust]
                        if load > capacity + _EPS:
                            continue
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost_inf:
                            best_cost_inf = c
                            best_ri = ri
                            best_pos = pos
                
                if best_ri == -1:
                    # Try inserting into a new route or with station help
                    for s in stations:
                        for pos in range(len(routes) + 1):
                            if pos < len(routes):
                                cand = routes[pos][:0] + [s, cust]
                            else:
                                cand = [s, cust]
                            c, v, _, _ = check_route(cand)
                            if v and c < best_cost_inf:
                                best_cost_inf = c
                                best_ri = pos if pos < len(routes) else len(routes)
                                best_pos = 0
                    
                    if best_ri == -1:
                        # Just add as new route
                        routes.append([cust])
                        route_costs.append(check_route([cust])[0])
                    else:
                        if best_ri < len(routes):
                            routes[best_ri] = routes[best_ri][:best_pos] + [cust] + routes[best_ri][best_pos:]
                        else:
                            routes.append([cust])
                        route_costs[best_ri if best_ri < len(routes) else len(routes)-1] = check_route(routes[best_ri if best_ri < len(routes) else len(routes)-1])[0]
                else:
                    routes[best_ri] = routes[best_ri][:best_pos] + [cust] + routes[best_ri][best_pos:]
                    route_costs[best_ri] = check_route(routes[best_ri])[0]
            
            # Remove empty routes
            new_routes = []
            new_costs = []
            for i, r in enumerate(routes):
                if r:
                    new_routes.append(r)
                    new_costs.append(route_costs[i])
            routes = new_routes
            route_costs = new_costs
            
            # Optimize after reinsertion
            for ri in range(len(routes)):
                if time.time() > DEADLINE - 5.0: break
                route = routes[ri]
                new_route, new_cost = optimize_route(route)
                if new_cost < route_costs[ri] - 1e-7:
                    routes[ri] = new_route
                    route_costs[ri] = new_cost
    
    # Final polish pass with tighter tolerance
    if time.time() < DEADLINE:
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            n = len(routes[ri])
            if n < 2: continue
            cur_cost, cur_valid, _, _ = check_route(routes[ri])
            if not cur_valid: continue
            best_route = list(routes[ri])
            best_cost = cur_cost
            improved = True
            while improved and time.time() < DEADLINE:
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
                routes[ri] = best_route
                route_costs[ri] = best_cost
    
    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
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
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_repair(routes[i])
                if rep:
                    routes[i] = rep

    return routes
