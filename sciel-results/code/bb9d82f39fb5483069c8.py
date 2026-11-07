import math
import time
import numpy as np

_EPS = 1e-9
_DEADLINE = 135.0  # Safety margin for 150s kill

def plan(instance):
    t0 = time.time()

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
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_set

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        Early exit on infeasibility for speed.
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity with early exit
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
                # Recharge
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            
            prev = node
            
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
            
        if t > horizon + _EPS:
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
        """Apply 2-opt (with nearest-neighbor search) + Or-opt to a single route."""
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
        
        # 2-opt with nearest-neighbor search for speed
        improved = True
        while improved:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr - 1):
                # Find best j for this i
                best_j = -1
                best_cand_cost = float('inf')
                for j in range(i + 1, n_curr):
                    # Compute cost delta for 2-opt reversal
                    # New edges: (i-1, j) and (i, j+1) instead of (i-1, i) and (j, j+1)
                    # where node -1 means depot (index 0), node n means depot
                    a = best_route[i - 1] if i > 0 else 0
                    b_node = best_route[i]
                    c_node = best_route[j]
                    d_node = best_route[j + 1] if j + 1 < n_curr else 0
                    
                    delta = dist[a, c_node] + dist[b_node, d_node] - dist[a, b_node] - dist[c_node, d_node]
                    
                    if delta < best_cand_cost - 1e-9:
                        # Quick feasibility check with partial route
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cand_cost = c
                            best_j = j
                
                if best_j != -1:
                    best_route = best_route[:i] + list(reversed(best_route[i:best_j+1])) + best_route[best_j+1:]
                    best_cost = best_cand_cost
                    improved = True
        
        # Or-opt (move 1-3 consecutive nodes)
        improved = True
        while improved:
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
    while improved and time.time() < _DEADLINE:
        improved = False
        
        # 1. Intra-route optimization (2-opt + or-opt)
        for ri in range(len(routes)):
            if time.time() > _DEADLINE: break
            route = routes[ri]
            
            new_route, new_cost = optimize_route(route)
            
            if new_cost < route_costs[ri] - 1e-7:
                routes[ri] = new_route
                route_costs[ri] = new_cost
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > _DEADLINE: break
        relocate_done = False
        for ri in range(len(routes)):
            if time.time() > _DEADLINE or relocate_done: break
            for ci in range(len(routes[ri])):
                if time.time() > _DEADLINE or relocate_done: break
                cust = routes[ri][ci]
                
                if not is_customer(cust):
                    continue
                    
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > _DEADLINE or relocate_done: continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > _DEADLINE or relocate_done: break
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
        if time.time() > _DEADLINE: break
        swap_done = False
        for ri in range(len(routes)):
            if time.time() > _DEADLINE or swap_done: break
            for rj in range(ri + 1, len(routes)):
                if time.time() > _DEADLINE or swap_done: break
                for ci in range(len(routes[ri])):
                    if time.time() > _DEADLINE or swap_done: break
                    cust1 = routes[ri][ci]
                    if not is_customer(cust1): continue
                    for cj in range(len(routes[rj])):
                        if time.time() > _DEADLINE or swap_done: break
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
