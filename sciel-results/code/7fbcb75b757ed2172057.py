import math
import time
import numpy as np
import heapq

_EPS = 1e-9

def plan(instance):
    t_start = time.time()
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

    # Node IDs: 0 depot, 1..n_cust customers, then stations
    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    
    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        Optimized for speed.
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Quick capacity check
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
                # Recharge to full
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
    
    # Start with empty routes
    routes = []
    route_loads = []
    route_costs = []
    
    # Sort customers by tightness (due - ready) or distance from depot to help construction
    # Using a simple heuristic: sort by distance from depot to spread them out?
    # Or just process in ID order? Let's try sorting by 'due' to handle tight windows early.
    cust_order = sorted(cust_ids, key=lambda c: due[c])
    
    unassigned = set(cust_order)
    
    while unassigned:
        # Find the best route and position to insert the next customer
        # If no routes, create a new one
        if not routes:
            c = unassigned.pop()
            routes.append([c])
            route_loads.append(demands[c])
            cost, valid, _, _ = check_route([c])
            if not valid:
                # Single customer route should always be valid unless battery/distance is huge
                # If invalid, we have a problem. But with 100x100 grid and battery 300, it should be fine.
                pass
            route_costs.append(cost)
            continue
            
        best_insertion_cost = float('inf')
        best_r_idx = -1
        best_pos = -1
        best_cust = -1
        
        # Try to insert each unassigned customer into each existing route
        # To save time, we can pick the "most constrained" customer first?
        # Or iterate all? N=25, so 25 * (num_routes * len_route) is small.
        
        for c in list(unassigned):
            for r_idx, route in enumerate(routes):
                # Check capacity
                if route_loads[r_idx] + demands[c] > capacity + _EPS:
                    continue
                
                # Try inserting at each position
                for pos in range(len(route) + 1):
                    # Quick distance check: dist[prev, c] + dist[c, next] - dist[prev, next]
                    prev_node = route[pos-1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0
                    
                    # Estimate extra distance
                    extra_dist = dist[prev_node, c] + dist[c, next_node] - dist[prev_node, next_node]
                    
                    # If estimated cost is already worse than best, skip detailed check
                    if route_costs[r_idx] + extra_dist > best_insertion_cost + 1e-7:
                        continue
                        
                    new_route = route[:pos] + [c] + route[pos:]
                    new_cost, valid, _, _ = check_route(new_route)
                    
                    if valid and new_cost < best_insertion_cost:
                        best_insertion_cost = new_cost
                        best_r_idx = r_idx
                        best_pos = pos
                        best_cust = c
                        
        if best_cust != -1:
            unassigned.remove(best_cust)
            routes[best_r_idx] = routes[best_r_idx][:best_pos] + [best_cust] + routes[best_r_idx][best_pos:]
            route_loads[best_r_idx] += demands[best_cust]
            route_costs[best_r_idx] = best_insertion_cost
        else:
            # Could not insert into existing routes, create new route
            # Pick the customer with the smallest distance to depot? Or just pop one.
            # To minimize routes, pick the one that fits best?
            # Since we failed to insert, we must create a new route.
            # Pick the customer that minimizes the cost of a new single-customer route?
            min_cost = float('inf')
            min_cust = None
            for c in unassigned:
                cost, valid, _, _ = check_route([c])
                if valid and cost < min_cost:
                    min_cost = cost
                    min_cust = c
            
            if min_cust:
                unassigned.remove(min_cust)
                routes.append([min_cust])
                route_loads.append(demands[min_cust])
                route_costs.append(min_cost)
            else:
                # Should not happen
                break

    # --- Local Search ---
    
    # Helper to check if we can remove a station
    def try_remove_station(route, station_idx):
        """
        Try to remove station at station_idx. Returns new route if feasible, else None.
        """
        if len(route) <= 1:
            return None
        if not is_station(route[station_idx]):
            return None
            
        new_route = route[:station_idx] + route[station_idx+1:]
        cost, valid, _, _ = check_route(new_route)
        if valid:
            return new_route
        return None

    # Helper to try inserting a station
    def try_insert_station(route, pos, station_id):
        new_route = route[:pos] + [station_id] + route[pos:]
        cost, valid, _, _ = check_route(new_route)
        if valid:
            return new_route, cost
        return None, float('inf')

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route 2-Opt and Or-Opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            n = len(route)
            if n < 2: continue
            
            cur_cost, cur_valid, _, _ = check_route(route)
            if not cur_valid:
                continue # Should not happen if construction is good, but skip if broken
            
            best_route = list(route)
            best_cost = cur_cost
            route_improved = True
            max_iter = 10 # Limit iterations per pass
            
            while route_improved and max_iter > 0 and time.time() < DEADLINE:
                route_improved = False
                max_iter -= 1
                
                # 2-Opt
                for i in range(n - 1):
                    if time.time() > DEADLINE: break
                    for j in range(i + 1, n):
                        # Skip if reversing segment includes depot? No, depot is implicit.
                        # But we must ensure we don't reverse the entire route in a way that breaks start/end logic?
                        # 2-opt on the sequence of nodes is fine.
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
                            
                # Or-Opt (move 1 or 2 nodes)
                # Move 1 node
                for i in range(n):
                    if time.time() > DEADLINE: break
                    node = best_route[i]
                    if i == 0 or i == n-1:
                        continue # Moving ends is effectively relocate to different position, handled by 2-opt? 
                        # Actually 2-opt covers reversals. Or-opt covers moves.
                        
                    # Try inserting at all other positions
                    # This is O(n^2) per node, so O(n^3) total. n~10, so 1000 ops. Fine.
                    for j in range(n):
                        if j == i or j == i+1: continue
                        # Remove i
                        temp = best_route[:i] + best_route[i+1:]
                        # Insert at j
                        # If j > i, index shifts
                        insert_idx = j
                        if j > i:
                            insert_idx = j - 1
                            
                        cand = temp[:insert_idx] + [node] + temp[insert_idx:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
                            # Break inner loop? No, continue to find better? 
                            # For simplicity, take first improvement and restart? 
                            # Let's just take best in this pass.
                            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                route_costs[ri] = best_cost
                improved = True
                
            # 2. Try removing stations
            for si in range(len(routes[ri]) - 1, -1, -1):
                if time.time() > DEADLINE: break
                new_route = try_remove_station(routes[ri], si)
                if new_route:
                    new_cost, valid, _, _ = check_route(new_route)
                    if valid and new_cost < route_costs[ri] - 1e-7:
                        routes[ri] = new_route
                        route_costs[ri] = new_cost
                        improved = True
                        break # Restart station removal for this route

        # 3. Inter-route Relocate
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
                    # If removing cust makes route infeasible (e.g. battery), we can't relocate it alone.
                    # We might need to move a station too. For now, skip.
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > DEADLINE or relocate_done: continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE or relocate_done: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        
                        # Capacity check
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
            
        # 4. Inter-route Swap
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
                        
                        # Capacity check
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

        # 5. Try inserting stations to enable better moves? 
        # This is complex. Instead, we rely on the fact that if a move is infeasible due to battery,
        # we might fix it by adding a station. 
        # Let's add a simple "station insertion" pass if we have time.
        if time.time() < DEADLINE - 10: # Only if we have plenty of time
            for ri in range(len(routes)):
                if time.time() > DEADLINE: break
                route = routes[ri]
                # Try inserting a station at each position
                for pos in range(len(route) + 1):
                    for s in stations:
                        if s in route: # Don't insert duplicate consecutive? Or allow?
                            # If s is already in route, inserting another might be redundant unless battery is low.
                            # check_route handles battery.
                            pass
                        cand = route[:pos] + [s] + route[pos:]
                        c, v, _, _ = check_route(cand)
                        if v:
                            # If this route is currently infeasible (shouldn't be), this fixes it.
                            # If it is feasible, does it allow a better 2-opt later? 
                            # For now, we only insert if it reduces cost? No, inserting station usually increases distance.
                            # We only want to insert if it *enables* a lower cost route via subsequent optimization.
                            # This is a "perturbation" or "repair". 
                            # Let's skip proactive insertion unless route is infeasible.
                            pass

    # Final validation
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
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        # Ensure validity of fallback
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                # Try to repair by adding station
                r = routes[i]
                for s in stations:
                    for p in range(len(r)+1):
                        cand = r[:p] + [s] + r[p:]
                        if check_route(cand)[1]:
                            routes[i] = cand
                            break
                    if check_route(routes[i])[1]:
                        break

    return routes
