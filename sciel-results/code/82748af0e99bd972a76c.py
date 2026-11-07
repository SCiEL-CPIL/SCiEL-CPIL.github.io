import numpy as np
import time
import random
import math
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Extract data
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    
    station_set = set(stations)
    
    # Helper: Simulate a route (list of node ids, customers and stations)
    # Returns total_distance if feasible, else float('inf')
    def simulate(route):
        if not route:
            return 0.0
        
        current_node = 0
        current_time = 0.0
        current_battery = max_battery
        current_load = 0
        total_dist = 0.0
        
        for node in route:
            dist_leg = D[current_node, node]
            if dist_leg > current_battery + 1e-9:
                return float('inf')
            
            travel_time = dist_leg / speed
            current_time += travel_time
            current_battery -= dist_leg
            total_dist += dist_leg
            
            if node in station_set:
                # Recharge to full
                deficit = max_battery - current_battery
                if deficit > 0:
                    current_time += deficit * recharge_rate
                current_battery = max_battery
            else:
                # Customer
                current_load += demands[node]
                if current_load > capacity:
                    return float('inf')
                
                if current_time < ready[node]:
                    current_time = ready[node]
                if current_time > due[node] + 1e-9:
                    return float('inf')
                current_time += service[node]
            
            current_node = node
        
        # Return to depot
        dist_back = D[current_node, 0]
        if dist_back > current_battery + 1e-9:
            # Try to find a station to charge on the way back
            # For simplicity in simulation, we assume if we can't go back directly, 
            # we might need a station. But in our construction, we ensure this.
            # For strict simulation of a given route list, if we can't go back, it's infeasible.
            return float('inf')
        
        current_time += dist_back / speed
        total_dist += dist_back
        
        if current_time > horizon + 1e-9:
            return float('inf')
            
        return total_dist

    # Helper: Build a feasible route from a sequence of customers by inserting stations
    # Returns (route_list_with_stations, total_distance) or (None, inf)
    def build_route(cust_seq):
        if not cust_seq:
            return [], 0.0
        
        route = []
        t = 0.0
        bat = max_battery
        load = 0
        prev = 0
        total_dist = 0.0
        
        for cust in cust_seq:
            d_direct = D[prev, cust]
            
            if bat >= d_direct - 1e-9:
                # Direct travel possible
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None, float('inf')
                
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                
                if load > capacity:
                    return None, float('inf')
                
                total_dist += d_direct
                route.append(cust)
                prev = cust
            else:
                # Need to charge
                best_st = -1
                best_total_dist = float('inf')
                best_end_time = 0.0
                best_bat_after = 0.0
                
                for s in stations:
                    d1 = D[prev, s]
                    if d1 > bat + 1e-9:
                        continue
                    d2 = D[s, cust]
                    if d2 > max_battery + 1e-9:
                        continue
                    
                    t1 = t + d1 / speed
                    bat_at_s = bat - d1
                    recharge_time = (max_battery - bat_at_s) * recharge_rate
                    t2 = t1 + recharge_time
                    t3 = t2 + d2 / speed
                    
                    if t3 < ready[cust]:
                        t3 = ready[cust]
                    if t3 > due[cust] + 1e-9:
                        continue
                    
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_st = s
                        best_end_time = t3
                        best_bat_after = max_battery - d2
                
                if best_st == -1:
                    return None, float('inf')
                
                total_dist += best_total_dist
                t = best_end_time + service[cust]
                bat = best_bat_after
                load += demands[cust]
                
                if load > capacity:
                    return None, float('inf')
                
                route.append(best_st)
                route.append(cust)
                prev = cust
        
        # Return to depot
        dist_back = D[prev, 0]
        if dist_back <= bat + 1e-9:
            t_back = t + dist_back / speed
            if t_back > horizon + 1e-9:
                return None, float('inf')
            total_dist += dist_back
        else:
            # Need to charge on the way back
            best_st = -1
            best_total_dist = float('inf')
            
            for s in stations:
                d1 = D[prev, s]
                if d1 > bat + 1e-9:
                    continue
                d2 = D[s, 0]
                if d2 > max_battery + 1e-9:
                    continue
                
                t1 = t + d1 / speed
                bat_at_s = bat - d1
                recharge_time = (max_battery - bat_at_s) * recharge_rate
                t2 = t1 + recharge_time
                t3 = t2 + d2 / speed
                
                if t3 > horizon + 1e-9:
                    continue
                
                total_d = d1 + d2
                if total_d < best_total_dist:
                    best_total_dist = total_d
                    best_st = s
            
            if best_st == -1:
                return None, float('inf')
            
            total_dist += best_total_dist
            route.append(best_st)
        
        return route, total_dist

    # Initial Solution: Nearest Neighbor with capacity/time/battery checks
    def construct_initial():
        routes = []
        unvisited = set(customers)
        
        while unvisited:
            # Start a new route from depot
            current_route_custs = []
            prev = 0
            t = 0.0
            bat = max_battery
            load = 0
            
            while unvisited:
                # Find nearest feasible customer
                best_c = -1
                best_d = float('inf')
                
                # Sort candidates by distance for efficiency? No, just iterate.
                # 25 customers is small.
                for c in list(unvisited):
                    if load + demands[c] > capacity:
                        continue
                    
                    d = D[prev, c]
                    if d > bat + 1e-9:
                        # Check if we can go via a station
                        # For simplicity in construction, skip if direct is not possible?
                        # Or try to find a station.
                        # Let's just check direct first for speed in construction.
                        # If direct not possible, we might still be feasible via station.
                        # But for NN construction, let's prefer direct or nearest via station.
                        # Actually, build_route handles the station insertion.
                        # So we just need to check if c is reachable from prev given current state.
                        # We can use a simplified check: is there a path?
                        # For now, let's just pick the nearest c such that build_route(current_route_custs + [c]) is feasible.
                        # This is expensive. 
                        # Alternative: Just pick nearest c. If infeasible, try next.
                        pass
                
                # Better approach: Iterate through unvisited, find the one that minimizes distance 
                # AND is feasible to add to the current route prefix.
                feasible_candidates = []
                for c in unvisited:
                    if load + demands[c] > capacity:
                        continue
                    # Quick check: distance
                    d = D[prev, c]
                    feasible_candidates.append((d, c))
                
                if not feasible_candidates:
                    break
                    
                feasible_candidates.sort(key=lambda x: x[0])
                
                added = False
                for d, c in feasible_candidates:
                    test_seq = current_route_custs + [c]
                    test_route, test_cost = build_route(test_seq)
                    if test_route is not None:
                        # Accept this customer
                        current_route_custs.append(c)
                        unvisited.remove(c)
                        
                        # Update state to reflect this addition
                        # We need to re-simulate to get exact t, bat, prev
                        # Or we can just let build_route do it at the end?
                        # No, we need to know the state to pick the next customer.
                        # So we simulate the prefix.
                        # Since N is small, this is okay.
                        sim_route, _ = build_route(current_route_custs)
                        if sim_route is None:
                            # Should not happen if we just checked
                            unvisited.add(c)
                            current_route_custs.pop()
                            break
                        
                        # Extract final state from sim_route
                        # Re-simulate to get t, bat, prev
                        t = 0.0
                        bat = max_battery
                        load = 0
                        prev = 0
                        for node in sim_route:
                            dist_leg = D[prev, node]
                            t += dist_leg / speed
                            bat -= dist_leg
                            if node in station_set:
                                deficit = max_battery - bat
                                if deficit > 0:
                                    t += deficit * recharge_rate
                                bat = max_battery
                            else:
                                load += demands[node]
                                if t < ready[node]:
                                    t = ready[node]
                                t += service[node]
                            prev = node
                        
                        added = True
                        break
                
                if not added:
                    break
            
            if current_route_custs:
                final_route, _ = build_route(current_route_custs)
                if final_route is not None:
                    routes.append(final_route)
            
            if time.time() > deadline - 5:
                break
        
        # Check if any customers are left unvisited
        visited = set()
        for r in routes:
            for node in r:
                if node not in station_set:
                    visited.add(node)
        
        remaining = [c for c in customers if c not in visited]
        
        if remaining:
            # Fallback: One route per remaining customer
            for c in remaining:
                r, cost = build_route([c])
                if r is not None:
                    routes.append(r)
        
        return routes

    routes = construct_initial()
    
    # Calculate initial cost
    def get_total_cost(routes):
        cost = 0
        for r in routes:
            c = simulate(r)
            if c == float('inf'):
                return float('inf')
            cost += c
        return cost

    current_cost = get_total_cost(routes)
    if current_cost == float('inf'):
        # Fallback: 1 customer per route
        routes = []
        for c in customers:
            r, cost = build_route([c])
            if r is not None:
                routes.append(r)
        current_cost = get_total_cost(routes)
        
    best_routes = [r[:] for r in routes]
    best_cost = current_cost
    
    # Simulated Annealing with Relocate and Swap moves
    # Note: Moves operate on the list of routes (which include stations).
    # This is tricky because moving a customer might require re-inserting stations.
    # Better approach: Keep the representation as "customer sequences" for the search,
    # and only build the full route with stations for evaluation.
    
    # Let's restart the search with customer sequences.
    # Convert current routes back to customer sequences
    def route_to_custs(route):
        return [node for node in route if node not in station_set]
    
    cust_routes = [route_to_custs(r) for r in routes]
    
    def evaluate_cust_routes(cust_routes):
        total_cost = 0
        for cr in cust_routes:
            if not cr:
                continue
            r, cost = build_route(cr)
            if r is None:
                return float('inf')
            total_cost += cost
        return total_cost

    current_cost = evaluate_cust_routes(cust_routes)
    best_cost = current_cost
    best_cust_routes = [r[:] for r in cust_routes]
    
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    while T > min_T and time.time() < deadline - 5:
        move_type = random.choice(['relocate', 'swap'])
        
        new_cust_routes = [r[:] for r in cust_routes]
        valid_move = False
        
        if move_type == 'relocate':
            if len(cust_routes) < 2:
                continue
            src_idx = random.randint(0, len(cust_routes) - 1)
            if len(cust_routes[src_idx]) <= 1:
                continue
            
            cust_pos = random.randint(0, len(cust_routes[src_idx]) - 1)
            cust = cust_routes[src_idx][cust_pos]
            
            dest_idx = random.randint(0, len(cust_routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(cust_routes) - 1)
            
            ins_pos = random.randint(0, len(cust_routes[dest_idx]))
            
            cust_routes[src_idx].pop(cust_pos)
            cust_routes[dest_idx].insert(ins_pos, cust)
            valid_move = True
            
        elif move_type == 'swap':
            if len(cust_routes) < 2:
                continue
            src_idx = random.randint(0, len(cust_routes) - 1)
            dest_idx = random.randint(0, len(cust_routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(cust_routes) - 1)
            
            if len(cust_routes[src_idx]) == 0 or len(cust_routes[dest_idx]) == 0:
                continue
                
            c1_pos = random.randint(0, len(cust_routes[src_idx]) - 1)
            c2_pos = random.randint(0, len(cust_routes[dest_idx]) - 1)
            
            c1 = cust_routes[src_idx][c1_pos]
            c2 = cust_routes[dest_idx][c2_pos]
            
            cust_routes[src_idx][c1_pos] = c2
            cust_routes[dest_idx][c2_pos] = c1
            valid_move = True
            
        if not valid_move:
            continue
            
        new_cost = evaluate_cust_routes(cust_routes)
        
        if new_cost == float('inf'):
            cust_routes = new_cust_routes
            T *= cooling
            continue
            
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_cust_routes = [r[:] for r in cust_routes]
        else:
            cust_routes = new_cust_routes
            
        T *= cooling
        
    # Convert best customer routes back to full routes with stations
    final_routes = []
    for cr in best_cust_routes:
        if not cr:
            continue
        r, cost = build_route(cr)
        if r is not None:
            final_routes.append(r)
        else:
            # Fallback: single customer routes
            for c in cr:
                r, cost = build_route([c])
                if r is not None:
                    final_routes.append(r)
    
    return final_routes
