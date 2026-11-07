import math
import time
import random
import numpy as np

_EPS = 1e-9
_DEADLINE = 140.0  # Safety margin for 150s kill

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
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity
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

    # --- Construction: Multiple strategies ---
    
    def construct_greedy_insertion():
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
        route_costs = [check_route(r)[0] for r in routes]
        return routes, route_costs

    def construct_nearest_neighbor():
        """Build routes using nearest-neighbor heuristic with time windows."""
        remaining = set(cust_ids)
        routes = []
        
        while remaining:
            # Start from the customer with earliest due date
            start = min(remaining, key=lambda c: due[c])
            route = [start]
            remaining.remove(start)
            
            while remaining:
                last = route[-1]
                # Find nearest feasible customer
                best_c = None
                best_d = float('inf')
                for c in remaining:
                    d = dist[last, c]
                    if d < best_d:
                        # Quick feasibility check
                        cand = route + [c]
                        _, v, _, _ = check_route(cand)
                        if v:
                            best_c = c
                            best_d = d
                
                if best_c is None:
                    break
                    
                route.append(best_c)
                remaining.remove(best_c)
            
            if route:
                routes.append(route)
        
        route_costs = [check_route(r)[0] for r in routes]
        return routes, route_costs

    def construct_due_date():
        """Build routes sorted by due date."""
        sorted_custs = sorted(cust_ids, key=lambda c: due[c])
        routes = []
        current_route = []
        current_load = 0.0
        
        for c in sorted_custs:
            if current_load + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                current_route = []
                current_load = 0.0
            
            # Check if adding this customer keeps the route feasible
            cand = current_route + [c]
            _, v, _, _ = check_route(cand)
            if v:
                current_route.append(c)
                current_load += demands[c]
            else:
                # Try to start a new route
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_load = 0.0
                # Single customer route
                routes.append([c])
                current_route = []
                current_load = 0.0
        
        if current_route:
            routes.append(current_route)
        
        route_costs = [check_route(r)[0] for r in routes]
        return routes, route_costs

    # Try multiple constructions and pick the best
    constructions = [
        construct_greedy_insertion(),
        construct_nearest_neighbor(),
        construct_due_date(),
    ]
    
    best_routes = None
    best_total = float('inf')
    for routes, route_costs in constructions:
        # Verify all customers are served
        served = set()
        valid = True
        for r in routes:
            for node in r:
                if node in served:
                    valid = False
                    break
                served.add(node)
            if not valid:
                break
        if valid and served == set(range(1, n_cust + 1)):
            total = sum(route_costs)
            if total < best_total:
                best_total = total
                best_routes = routes
    
    if best_routes is None:
        best_routes, _ = construct_greedy_insertion()
    
    routes = best_routes
    route_costs = [check_route(r)[0] for r in routes]

    # --- Local Search ---

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
        """Apply 2-opt + or-opt to a single route to minimize distance while maintaining feasibility."""
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
        
        # 2-opt with early termination
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr - 1):
                if time.time() > _DEADLINE:
                    break
                for j in range(i + 1, n_curr):
                    # Quick distance check before full validation
                    if i == 0:
                        delta_d = dist[best_route[i+1], best_route[j]] + dist[best_route[i], best_route[j+1] if j+1 < n_curr else 0] - dist[best_route[i], best_route[i+1]] - dist[best_route[j], best_route[j+1] if j+1 < n_curr else 0]
                    else:
                        delta_d = dist[best_route[i-1], best_route[j]] + dist[best_route[i], best_route[j+1] if j+1 < n_curr else 0] - dist[best_route[i-1], best_route[i]] - dist[best_route[j], best_route[j+1] if j+1 < n_curr else 0]
                    
                    if delta_d >= 0:
                        continue
                    
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        
        # Or-opt (move 1-3 consecutive nodes)
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr):
                if time.time() > _DEADLINE:
                    break
                for k in range(1, 4):
                    if i + k > n_curr:
                        continue
                    segment = best_route[i:i+k]
                    rest = best_route[:i] + best_route[i+k:]
                    for j in range(len(rest) + 1):
                        if j == i and j <= len(rest):
                            continue  # Skip no-op
                        cand = rest[:j] + segment + rest[j:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            improved = True
                            break
        
        return best_route, best_cost

    # Main local search loop
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
        
        # 2. Inter-route Relocate (try all moves, keep best)
        if time.time() > _DEADLINE: break
        best_relocate = None
        best_relocate_cost = float('inf')
        
        for ri in range(len(routes)):
            if time.time() > _DEADLINE: break
            for ci in range(len(routes[ri])):
                if time.time() > _DEADLINE: break
                cust = routes[ri][ci]
                
                if not is_customer(cust):
                    continue
                    
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > _DEADLINE: continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > _DEADLINE: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        
                        load_to = route_load(routes[rj]) + demands[cust]
                        if load_to > capacity + _EPS:
                            continue
                            
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        old_cost = route_costs[ri] + route_costs[rj]
                        new_cost = c_from + c_to
                        if new_cost < best_relocate_cost - 1e-7:
                            best_relocate_cost = new_cost
                            best_relocate = (ri, ci, rj, cj, temp_from, cand_to, c_from, c_to)
        
        if best_relocate is not None:
            ri, ci, rj, cj, new_from, new_to, c_from, c_to = best_relocate
            routes[ri] = new_from
            routes[rj] = new_to
            route_costs[ri] = c_from
            route_costs[rj] = c_to
            improved = True
        
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
            
        # 3. Inter-route Swap (try all swaps, keep best)
        if time.time() > _DEADLINE: break
        best_swap = None
        best_swap_cost = float('inf')
        
        for ri in range(len(routes)):
            if time.time() > _DEADLINE: break
            for rj in range(ri + 1, len(routes)):
                if time.time() > _DEADLINE: break
                for ci in range(len(routes[ri])):
                    if time.time() > _DEADLINE: break
                    cust1 = routes[ri][ci]
                    if not is_customer(cust1): continue
                    for cj in range(len(routes[rj])):
                        if time.time() > _DEADLINE: break
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
                            if old_c_i + old_c_j > best_swap_cost + 1e-7:
                                best_swap_cost = c_i + c_j
                                best_swap = (ri, rj, new_route_i, new_route_j, c_i, c_j)
        
        if best_swap is not None:
            ri, rj, new_i, new_j, c_i, c_j = best_swap
            routes[ri] = new_i
            routes[rj] = new_j
            route_costs[ri] = c_i
            route_costs[rj] = c_j
            improved = True

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
