import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s kill

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
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

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

    def try_insert_station(route, max_stations=2):
        """
        If route is infeasible due to battery, try inserting stations.
        Returns (new_route, new_cost) or (None, inf)
        """
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist

        best_route = None
        best_cost = float('inf')

        # Try inserting each station at each position (up to max_stations insertions)
        # For simplicity, we'll do a BFS/DFS up to depth max_stations
        # But for speed, let's just try single insertions first, then pairs if needed.
        # Given battery 300 and max distance ~141, single customer is always feasible.
        # Merging two customers might need 1 station. Merging 3 might need 1 or 2.
        
        # Strategy: Try all single insertions. If none work, try all pairs (if max_stations >= 2).
        
        # Level 1: Single station insertion
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
            # Level 2: Two station insertions
            # This is O(S^2 * N^2) which might be slow, but S=4, N is small.
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

    # --- Construction: Cluster + Savings ---
    
    # 1. Cluster customers based on proximity (K-Means like or simple nearest neighbor)
    # Let's use a simple greedy clustering: start with first customer, add nearest if capacity allows.
    # Actually, let's stick to the savings method but initialize with better clusters.
    
    # Let's try a different construction: 
    # 1. Sort customers by due time.
    # 2. Greedy insert into routes.
    
    routes = []
    remaining = list(cust_ids)
    
    # We will create routes one by one.
    # For each new route, we pick the customer with the earliest due time that is not yet served.
    # Then we greedily add the next best customer that fits (capacity, time, battery).
    
    while remaining:
        # Pick seed customer: earliest due
        remaining.sort(key=lambda x: due[x])
        seed = remaining[0]
        remaining.remove(seed)
        
        current_route = [seed]
        
        # Greedy insertion
        improved = True
        while improved:
            improved = False
            best_cand = None
            best_pos = -1
            best_cost = float('inf')
            
            for c in remaining:
                for pos in range(len(current_route) + 1):
                    cand = current_route[:pos] + [c] + current_route[pos:]
                    # Check feasibility
                    c_cost, v, _, _ = check_route(cand)
                    if v:
                        # We want to minimize the increase in cost? Or just find any feasible?
                        # To keep routes compact, let's minimize total cost.
                        if c_cost < best_cost:
                            best_cost = c_cost
                            best_cand = c
                            best_pos = pos
            
            if best_cand is not None:
                # Insert
                current_route = current_route[:best_pos] + [best_cand] + current_route[best_pos:]
                remaining.remove(best_cand)
                improved = True
        
        routes.append(current_route)
    
    # Now we have a set of routes. Let's try to merge them using Savings.
    active = [True] * len(routes)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_merged = None
        
        for i in range(len(routes)):
            if not active[i]: continue
            for j in range(i + 1, len(routes)):
                if not active[j]: continue
                
                # Try merging i and j in both orders
                for order in [0, 1]:
                    if order == 0:
                        merged = routes[i] + routes[j]
                    else:
                        merged = routes[j] + routes[i]
                    
                    c_merged, v_merged, _, _ = check_route(merged)
                    if v_merged:
                        ci, _, _, _ = check_route(routes[i])
                        cj, _, _, _ = check_route(routes[j])
                        saving = ci + cj - c_merged
                        if saving > best_saving:
                            best_saving = saving
                            best_i = i
                            best_j = j
                            best_merged = merged
                    else:
                        # Try to repair with station
                        rep, c_rep = try_insert_station(merged, max_stations=1)
                        if rep:
                            ci, _, _, _ = check_route(routes[i])
                            cj, _, _, _ = check_route(routes[j])
                            saving = ci + cj - c_rep
                            if saving > best_saving:
                                best_saving = saving
                                best_i = i
                                best_j = j
                                best_merged = rep

        if best_i != -1:
            routes[best_i] = best_merged
            active[best_j] = False
            changed = True

    routes = [r for i, r in enumerate(routes) if active[i]]
    
    # --- Local Search ---
    
    def get_route_cost(route):
        return check_route(route)[0]
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route 2-Opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            n = len(route)
            if n < 2: continue
            
            cur_cost, cur_valid, _, _ = check_route(route)
            if not cur_valid:
                rep, rcost = try_insert_station(route)
                if rep:
                    routes[ri] = rep
                    cur_cost = rcost
                    cur_valid = True
                else:
                    continue
            
            best_route = list(route)
            best_cost = cur_cost
            route_improved = True
            max_iter = 5 # Limit iterations to save time
            while route_improved and max_iter > 0:
                route_improved = False
                max_iter -= 1
                for i in range(n - 1):
                    if time.time() > DEADLINE: break
                    for j in range(i + 1, n):
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        relocated = False
        for ri in range(len(routes)):
            if relocated or time.time() > DEADLINE: break
            for ci in range(len(routes[ri])):
                if relocated or time.time() > DEADLINE: break
                cust = routes[ri][ci]
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or relocated or time.time() > DEADLINE: continue
                    for cj in range(len(routes[rj]) + 1):
                        if relocated or time.time() > DEADLINE: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        # Capacity check
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue
                        
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            relocated = True
                            break
                    if relocated: break
                if relocated: break
            if relocated: break
        
        # 3. Inter-route Swap
        if time.time() > DEADLINE: break
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > DEADLINE: break
            for rj in range(ri + 1, len(routes)):
                if swapped or time.time() > DEADLINE: break
                for ci in range(len(routes[ri])):
                    if swapped or time.time() > DEADLINE: break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if swapped or time.time() > DEADLINE: break
                        cust2 = routes[rj][cj]
                        
                        # Capacity check
                        load_i = sum(demands[n] for n in routes[ri] if is_customer(n))
                        load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
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
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
                                break
                if swapped: break
            if swapped: break

        # Remove empty routes
        routes = [r for r in routes if r]

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
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_insert_station(routes[i])
                if rep:
                    routes[i] = rep

    return routes
