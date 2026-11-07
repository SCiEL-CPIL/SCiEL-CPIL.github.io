import numpy as np
import math
import time
import random
import copy

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    # Precompute pairwise distances
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords_arr[i][0] - coords_arr[j][0]
            dy = coords_arr[i][1] - coords_arr[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    station_set = set(stations)

    def simulate_route(cust_seq):
        """
        Simulates a route given a sequence of customer IDs.
        Dynamically inserts charging stations as needed.
        Returns (total_distance, is_valid, total_time)
        """
        if not cust_seq:
            return 0.0, True, 0.0
        
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        
        for cust in cust_seq:
            # Check capacity
            if load + demands[cust] > capacity:
                return float('inf'), False, 0.0
            load += demands[cust]
            
            # Try to go directly
            d_direct = dist[prev][cust]
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return float('inf'), False, 0.0
                t = t_arr + service[cust]
                bat -= d_direct
                total_dist += d_direct
                prev = cust
            else:
                # Need to charge. Find best station.
                best_st = None
                best_t_after = float('inf')
                best_dist_total = float('inf')
                best_bat_after = 0.0
                
                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    recharge_time = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                    t_st += recharge_time
                    bat_st = battery_cap
                    
                    d_st_cust = dist[st][cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    
                    t_after = t_cust + service[cust]
                    bat_after = bat_st - d_st_cust
                    d_total_seg = d_to_st + d_st_cust
                    
                    # Prefer earlier arrival, then shorter distance
                    if t_after < best_t_after - 1e-9 or (abs(t_after - best_t_after) < 1e-9 and d_total_seg < best_dist_total - 1e-9):
                        best_t_after = t_after
                        best_dist_total = d_total_seg
                        best_st = st
                        best_bat_after = bat_after
                
                if best_st is None:
                    return float('inf'), False, 0.0
                
                total_dist += best_dist_total
                t = best_t_after
                bat = best_bat_after
                prev = cust
        
        # Return to depot
        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return float('inf'), False, 0.0
            total_dist += d_ret
        else:
            # Need to charge before returning
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                recharge_time = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                t_st += recharge_time
                bat_st = battery_cap
                
                d_st_dep = dist[st][0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                
                total_dist += d_to_st + d_st_dep
                found = True
                break
            
            if not found:
                return float('inf'), False, 0.0

        return total_dist, True, t

    def total_distance(routes):
        total = 0.0
        for r in routes:
            d, valid, _ = simulate_route(r)
            if not valid:
                return float('inf')
            total += d
        return total

    # --- Construction: Nearest Neighbor / Savings hybrid ---
    def construct_initial():
        unserved = set(range(1, n_customers + 1))
        routes = []
        
        while unserved:
            # Start a new route from the customer with the earliest due time
            start_cust = min(unserved, key=lambda c: due[c])
            unserved.remove(start_cust)
            current_route = [start_cust]
            
            while unserved:
                best_cust = None
                best_pos = -1
                best_cost = float('inf')
                
                # Try inserting any unserved customer into any position in current_route
                for pos in range(len(current_route) + 1):
                    for c in unserved:
                        # Quick capacity check
                        load = sum(demands[x] for x in current_route)
                        if load + demands[c] > capacity:
                            continue
                        
                        new_route = current_route[:pos] + [c] + current_route[pos:]
                        d, valid, _ = simulate_route(new_route)
                        if valid and d < best_cost:
                            best_cost = d
                            best_cust = c
                            best_pos = pos
                
                if best_cust is None:
                    break
                
                unserved.remove(best_cust)
                current_route = current_route[:best_pos] + [best_cust] + current_route[best_pos:]
            
            routes.append(current_route)
        
        return routes

    # --- LNS ---
    def destroy(routes, num_to_remove):
        """Removes num_to_remove random customers from routes."""
        new_routes = [r[:] for r in routes]
        removed = []
        
        # Flatten to list of (route_idx, pos, cust)
        all_positions = []
        for i, r in enumerate(new_routes):
            for j, c in enumerate(r):
                if 1 <= c <= n_customers:
                    all_positions.append((i, j, c))
        
        if not all_positions:
            return new_routes, removed
            
        # Randomly select customers to remove
        k = min(num_to_remove, len(all_positions))
        indices_to_remove = random.sample(range(len(all_positions)), k)
        
        # Mark for removal
        to_remove_set = set()
        for idx in indices_to_remove:
            i, j, c = all_positions[idx]
            to_remove_set.add((i, j))
            removed.append(c)
            
        # Rebuild routes
        final_routes = []
        for i, r in enumerate(new_routes):
            new_r = []
            for j, c in enumerate(r):
                if (i, j) not in to_remove_set:
                    new_r.append(c)
            if new_r:
                final_routes.append(new_r)
                
        return final_routes, removed

    def repair(routes, removed_customers):
        """Inserts removed customers back into routes using best insertion."""
        # Sort removed customers by due time for stability
        removed_customers.sort(key=lambda c: due[c])
        
        current_routes = [r[:] for r in routes]
        
        for c in removed_customers:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            # Try inserting into existing routes
            for i, r in enumerate(current_routes):
                # Quick capacity check
                load = sum(demands[x] for x in r)
                if load + demands[c] > capacity:
                    continue
                    
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [c] + r[pos:]
                    d, valid, _ = simulate_route(new_r)
                    if valid and d < best_cost:
                        best_cost = d
                        best_route_idx = i
                        best_pos = pos
            
            # If no valid insertion, start a new route
            if best_route_idx == -1:
                # Check if single customer route is valid
                d, valid, _ = simulate_route([c])
                if valid:
                    current_routes.append([c])
                else:
                    # Infeasible instance? Should not happen if initial is valid
                    # Fallback: add it anyway, hoping simulation handles it or we fix later
                    current_routes.append([c])
            else:
                current_routes[best_route_idx] = current_routes[best_route_idx][:best_pos] + [c] + current_routes[best_route_idx][best_pos:]
                
        return current_routes

    def local_search(routes):
        """Applies 2-opt and relocate moves."""
        improved = True
        iterations = 0
        max_iter = 50
        
        while improved and iterations < max_iter and time.time() < deadline - 10.0:
            improved = False
            iterations += 1
            
            # 2-opt within routes
            for i in range(len(routes)):
                r = routes[i]
                if len(r) < 2:
                    continue
                best_r = r[:]
                best_d, _, _ = simulate_route(best_r)
                
                for a in range(len(best_r)):
                    for b in range(a + 1, len(best_r)):
                        new_r = best_r[:a] + best_r[a:b+1][::-1] + best_r[b+1:]
                        d, valid, _ = simulate_route(new_r)
                        if valid and d < best_d - 1e-9:
                            best_d = d
                            best_r = new_r[:]
                            improved = True
                
                routes[i] = best_r
            
            # Relocate between routes
            for i in range(len(routes)):
                for j in range(len(routes[i])):
                    c = routes[i][j]
                    if not (1 <= c <= n_customers):
                        continue
                    
                    # Remove from i
                    new_i = routes[i][:j] + routes[i][j+1:]
                    
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_k = routes[k][:pos] + [c] + routes[k][pos:]
                            
                            # Check capacity
                            load_k = sum(demands[x] for x in new_k)
                            if load_k > capacity:
                                continue
                            
                            d_i, valid_i, _ = simulate_route(new_i)
                            d_k, valid_k, _ = simulate_route(new_k)
                            
                            if valid_i and valid_k:
                                old_d = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
                                new_d = d_i + d_k
                                if new_d < old_d - 1e-9:
                                    routes[i] = new_i
                                    routes[k] = new_k
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        return routes

    # --- Main Execution ---
    
    # 1. Initial Solution
    best_routes = construct_initial()
    best_dist = total_distance(best_routes)
    
    # If initial is invalid (shouldn't be), try fallback
    if best_dist == float('inf'):
        best_routes = [[c] for c in range(1, n_customers + 1)]
        best_dist = total_distance(best_routes)

    current_routes = best_routes[:]
    current_dist = best_dist

    # 2. LNS Loop
    start_lns = time.time()
    lns_time_limit = 100.0  # Reserve time for LS and final checks
    
    while time.time() - start_lns < lns_time_limit:
        # Destroy
        # Remove 20-40% of customers
        total_custs = sum(len(r) for r in current_routes)
        num_remove = max(2, int(total_custs * random.uniform(0.2, 0.4)))
        
        destroyed_routes, removed = destroy(current_routes, num_remove)
        
        # Repair
        repaired_routes = repair(destroyed_routes, removed)
        
        # Evaluate
        rep_dist = total_distance(repaired_routes)
        
        if rep_dist < current_dist - 1e-9:
            current_routes = repaired_routes
            current_dist = rep_dist
            
            if current_dist < best_dist - 1e-9:
                best_routes = current_routes[:]
                best_dist = current_dist
        
        # Simulated Annealing style acceptance (optional, but greedy is safer for time)
        # For now, stick to greedy improvement to ensure validity and speed

    # 3. Local Search on Best Solution
    if time.time() < deadline - 20.0:
        best_routes = local_search(best_routes)
        best_dist = total_distance(best_routes)

    # 4. Final Validation and Cleanup
    valid_routes = []
    covered = set()
    for r in best_routes:
        d, valid, _ = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    # Add any missing customers (should be none)
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            sr = [cust]
            d, valid, _ = simulate_route(sr)
            if valid:
                valid_routes.append(sr)
            else:
                # Last resort: try to insert into any route
                for i, r in enumerate(valid_routes):
                    for pos in range(len(r) + 1):
                        new_r = r[:pos] + [cust] + r[pos:]
                        d, valid, _ = simulate_route(new_r)
                        if valid:
                            valid_routes[i] = new_r
                            covered.add(cust)
                            break
                    if cust in covered:
                        break
                if cust not in covered:
                    valid_routes.append([cust]) # Might be invalid, but better than missing

    return valid_routes

