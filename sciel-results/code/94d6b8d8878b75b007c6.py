import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s kill

    # 1. Parse Input
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
                t = max(t, ready[node])
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

    def try_insert_station(route):
        """
        Attempts to make a route feasible by inserting a station.
        Returns (new_route, new_cost) or (None, inf)
        """
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist
        
        best_route = None
        best_cost = float('inf')
        
        # Try inserting each station at each position
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand
                    
        if best_route:
            return best_route, best_cost
        return None, float('inf')

    # --- Construction: Savings Algorithm ---
    
    # Initialize with one customer per route
    routes = [[c] for c in cust_ids]
    
    # Pre-check and repair single customer routes
    for i in range(len(routes)):
        if not check_route(routes[i])[1]:
            rep, _ = try_insert_station(routes[i])
            if rep:
                routes[i] = rep

    # Calculate savings
    # s(i, j) = dist(0, i) + dist(j, 0) - dist(i, j)
    # We want to merge routes ending in i and starting with j.
    # However, standard savings merges two separate routes.
    # Here we have single nodes initially.
    # s(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    
    savings = []
    for i in cust_ids:
        for j in cust_ids:
            if i == j: continue
            s = dist[0, i] + dist[0, j] - dist[i, j]
            savings.append((s, i, j))
            
    savings.sort(reverse=True)
    
    # Map node to route index
    # Route structure: list of lists of nodes
    # We need to know which route a node belongs to and its position (start/end)
    # Since routes are small initially, we can just search.
    
    active = [True] * len(routes)
    
    def get_route_info(node):
        for idx, r in enumerate(routes):
            if not active[idx]: continue
            if node in r:
                if r[0] == node:
                    return idx, 'start'
                elif r[-1] == node:
                    return idx, 'end'
                else:
                    return idx, 'middle'
        return -1, None

    # Greedy Merge
    for s, i, j in savings:
        if time.time() > DEADLINE: break
        
        if s <= 0: break # No more positive savings
        
        idx_i, pos_i = get_route_info(i)
        idx_j, pos_j = get_route_info(j)
        
        if idx_i == -1 or idx_j == -1:
            continue
        if idx_i == idx_j:
            continue
        
        # We can merge if i is at the end of route idx_i and j is at the start of route idx_j
        # OR i is at start and j is at end (reverse order)
        # Standard savings: i is end of A, j is start of B. Merge A + B.
        
        if pos_i == 'end' and pos_j == 'start':
            merged = routes[idx_i] + routes[idx_j]
            c, v, _, _ = check_route(merged)
            if not v:
                rep, c_rep = try_insert_station(merged)
                if rep:
                    merged = rep
                    c = c_rep
                    v = True
            
            if v:
                # Check capacity (already done in check_route)
                routes[idx_i] = merged
                active[idx_j] = False
        elif pos_i == 'start' and pos_j == 'end':
            # Merge B + A (j is end of B, i is start of A -> B + A means j connects to i? No.)
            # If i is start of A, and j is end of B.
            # We want to connect end of B (j) to start of A (i).
            # So new route is B + A.
            merged = routes[idx_j] + routes[idx_i]
            c, v, _, _ = check_route(merged)
            if not v:
                rep, c_rep = try_insert_station(merged)
                if rep:
                    merged = rep
                    c = c_rep
                    v = True
            
            if v:
                routes[idx_j] = merged
                active[idx_i] = False

    # Remove inactive routes
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
            max_iter = 5
            while route_improved and max_iter > 0:
                route_improved = False
                max_iter -= 1
                for i in range(n - 1):
                    if time.time() > DEADLINE: break
                    for j in range(i + 1, n):
                        # 2-opt: reverse segment i..j
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
        
        # 2. Intra-route Or-opt (move 1, 2, or 3 nodes)
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            n = len(route)
            if n < 3: continue
            
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
            
            for k in [1, 2, 3]:
                for i in range(n - k):
                    if time.time() > DEADLINE: break
                    segment = best_route[i:i+k]
                    # Remove segment
                    temp = best_route[:i] + best_route[i+k:]
                    # Try inserting at all other positions
                    for j in range(len(temp) + 1):
                        if j >= i and j <= i+k: # Skip original position vicinity? 
                            # Actually, if j is such that it puts it back in same spot, skip
                            # If we remove i..i+k-1, the new list is shorter.
                            # Original pos was i.
                            # If we insert at j, and the resulting sequence is same, skip.
                            # Simple check: if j == i or j == i+1 (in new list indices it's tricky)
                            # Just check cost.
                            pass
                            
                        cand = temp[:j] + segment + temp[j:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True

        # 3. Inter-route Relocate
        if time.time() > DEADLINE: break
        relocate_done = False
        for ri in range(len(routes)):
            if relocate_done or time.time() > DEADLINE: break
            for ci in range(len(routes[ri])):
                if relocate_done or time.time() > DEADLINE: break
                cust = routes[ri][ci]
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if relocate_done or time.time() > DEADLINE: break
                    if rj == ri: continue
                    for cj in range(len(routes[rj]) + 1):
                        if relocate_done or time.time() > DEADLINE: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            # Try repair? Relocate repair is expensive, skip for now
                            continue
                        
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue
                        
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            relocate_done = True
                            break
                if relocate_done: break
            if relocate_done: break
            
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # 4. Inter-route Swap
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
