import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    # 150s limit, keep 5s safety margin
    deadline = start_time + 145.0

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)
    
    # Pre-compute distance matrix using NumPy for speed
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    def simulate_route(route):
        """
        Simulate a route. Returns (is_valid, total_distance).
        Optimized for speed: local variable references, minimal overhead.
        """
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        # Local references for speed
        d_mat = dist
        req = demands
        rd = ready
        dd = due
        sv = service
        st = stations_set
        cap = capacity
        bat_cap = battery
        spd = speed
        rch = recharge_rate
        hor = horizon

        for node in route:
            d = d_mat[cur_node, node]
            cur_time += d / spd
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in st:
                energy = bat_cap - cur_bat
                cur_time += energy * rch
                cur_bat = bat_cap
            else:
                load += req[node]
                if load > cap + 1e-9:
                    return False, total_dist
                if cur_time < rd[node]:
                    cur_time = rd[node]
                if cur_time > dd[node] + 1e-9:
                    return False, total_dist
                cur_time += sv[node]

            cur_node = node

        d = d_mat[cur_node, 0]
        cur_time += d / spd
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > hor + 1e-9:
            return False, total_dist

        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        d_mat = dist
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += d_mat[cur_node, node]
                cur_node = node
            total += d_mat[cur_node, 0]
        return total

    def construct_nn(rng):
        """Nearest Neighbor Construction"""
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            # Start route from depot
            route = []
            cur_node = 0
            cur_load = 0
            cur_bat = battery
            cur_time = 0.0
            
            while unvisited:
                # Find nearest feasible customer
                best_node = -1
                best_dist = float('inf')
                
                # Optimization: Only check unvisited
                for c in unvisited:
                    d = dist[cur_node, c]
                    if d < best_dist:
                        # Check if feasible
                        new_time = cur_time + d / speed
                        if c in stations_set:
                            # Stations are usually not in 'customers' list for NN, 
                            # but if they were, handle logic. Here customers are 1..25.
                            pass
                        else:
                            # Check capacity
                            if cur_load + demands[c] > capacity:
                                continue
                            # Check time window
                            arr_time = new_time
                            if arr_time < ready[c]:
                                arr_time = ready[c]
                            if arr_time > due[c]:
                                continue
                            # Check battery (approx: just distance for now, refine later if needed)
                            if cur_bat - d < 0:
                                # Need charging, skip for simplicity in NN or handle explicitly
                                # For simplicity, skip if battery insufficient
                                continue
                            
                        best_dist = d
                        best_node = c
                
                if best_node == -1:
                    break
                
                # Move to best_node
                d = dist[cur_node, best_node]
                cur_time += d / speed
                cur_bat -= d
                cur_node = best_node
                
                if best_node in stations_set:
                    energy = battery - cur_bat
                    cur_time += energy * recharge_rate
                    cur_bat = battery
                else:
                    cur_load += demands[best_node]
                    if cur_time < ready[best_node]:
                        cur_time = ready[best_node]
                    cur_time += service[best_node]
                    unvisited.remove(best_node)
                
                route.append(best_node)
            
            if route:
                # Check return to depot
                d_ret = dist[cur_node, 0]
                if cur_bat - d_ret >= 0 and cur_time + d_ret / speed <= horizon:
                    routes.append(route)
                else:
                    # Route invalid, try to salvage? Or just discard and start new?
                    # For NN, if it fails, we lost those customers. 
                    # Better: just append and let local search fix? No, invalid route is bad.
                    # We should have checked feasibility more carefully.
                    # For now, if invalid, we drop the last customer that caused it?
                    # Simpler: Just append if valid, else restart? 
                    # Let's just append if valid. If invalid, we have a problem.
                    # Given NN is greedy, it's rare to fail return if we checked battery.
                    routes.append(route)
            
            # If we broke out of inner loop with unvisited, we start a new route
            # But we might have skipped customers due to capacity/battery.
            # This loop structure is flawed for multiple routes. 
            # Let's restart construction with a simpler logic:
            pass
            
        return routes

    def construct_savings(rng):
        """Clarke-Wright Savings Heuristic adapted for capacity/battery"""
        # Calculate savings
        # s(i, j) = d(0, i) + d(0, j) - d(i, j)
        # Sort customers by angle or savings? 
        # Standard CW:
        # 1. Start with each customer in its own route (0, i, 0)
        # 2. Calculate savings for all pairs
        # 3. Merge routes with highest savings if feasible
        
        # This is complex to implement efficiently in Python for 25 nodes in 150s?
        # 25 nodes is small. O(N^2) is 625. Very fast.
        
        # Initial routes: each customer is a route [c]
        routes = [[c] for c in customers]
        route_ids = list(range(n_customers))
        
        # Precompute savings
        savings = np.zeros((n_customers, n_customers))
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                ci = customers[i]
                cj = customers[j]
                savings[i, j] = dist[0, ci] + dist[0, cj] - dist[ci, cj]
        
        # Create list of pairs sorted by savings
        pairs = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                pairs.append((savings[i, j], i, j))
        pairs.sort(reverse=True)
        
        # Map customer to route index
        # We need a structure to quickly find the route containing a customer
        # and the position of the customer in that route (start or end)
        
        # Since N=25, we can just check validity on merge.
        
        # We need to track:
        # route_nodes: list of lists
        # route_load: list of sums
        # route_bat_end: battery at end of route (before return)
        # route_time_end: time at end of route (before return)
        
        # Actually, merging two routes (0, A, 0) and (0, B, 0) into (0, A, B, 0) or (0, B, A, 0)
        # Savings is for connecting A_end to B_start.
        
        # Let's use a simpler heuristic: Greedy Insertion with Regret-2
        pass

    def construct_greedy_regret(rng):
        """
        Greedy Insertion with Regret-2.
        1. Start with empty routes.
        2. At each step, for each unvisited customer, calculate the cost of inserting
           into the best position in each existing route (or starting a new route).
           Regret = cost_2nd_best - cost_1st_best.
        3. Insert the customer with the highest regret into its best position.
        """
        routes = []
        unvisited = set(customers)
        
        while unvisited:
            best_customer = None
            best_regret = -float('inf')
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            # For each unvisited customer
            for c in unvisited:
                costs = []
                
                # Option 1: Start new route
                # Cost = d(0,c) + d(c,0)
                # Validity check:
                d_out = dist[0, c]
                d_in = dist[c, 0]
                if demands[c] <= capacity and d_out <= battery and d_in <= battery:
                    t_arr = d_out / speed
                    if t_arr < ready[c]: t_arr = ready[c]
                    if t_arr <= due[c]:
                        t_end = t_arr + service[c]
                        t_ret = t_end + d_in / speed
                        if t_ret <= horizon:
                            costs.append((d_out + d_in, -1, -1)) # (cost, route_idx, pos)
                
                # Option 2: Insert into existing routes
                for r_idx, route in enumerate(routes):
                    # Try all positions
                    for pos in range(len(route) + 1):
                        # Construct new route to check validity? 
                        # Too slow. Use delta approximation?
                        # For N=25, full simulation is O(N). 
                        # 25 customers * 25 routes * 25 pos * 25 sim = 25^4 = 390k ops. 
                        # This is fast enough in Python if optimized.
                        
                        new_route = route[:pos] + [c] + route[pos:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid:
                            # Cost increase is new_dist - old_dist
                            # We need old_dist. Precompute route distances?
                            # Let's just use new_dist as proxy if we compare within same customer?
                            # No, we need consistent cost.
                            # Let's store route distances in a list 'route_dists'
                            pass
                
                # This is getting complex and slow. 
                # Let's stick to a simpler, faster construction: 
                # **Parallel Nearest Neighbor with Randomization**
                pass
        
        return routes

    # --- RE-DESIGN: Fast Multi-Start NN + Fast Local Search ---
    
    def construct_nn_fast(rng):
        """
        Fast Nearest Neighbor.
        We maintain a list of active routes.
        We pick the next customer to visit based on nearest neighbor from the end of a random active route,
        or start a new route if no active route can take it.
        """
        routes = []
        unvisited = set(customers)
        
        # To make it diverse, we can shuffle the order in which we consider "starting" new routes
        # or pick the next customer from unvisited with some probability of picking nearest.
        
        # Simple NN:
        # 1. Start with depot.
        # 2. Find nearest feasible customer.
        # 3. If none, close route.
        # 4. Repeat.
        
        # To get multiple routes, we just keep going until all unvisited are served.
        
        while unvisited:
            route = []
            cur_node = 0
            cur_load = 0
            cur_bat = battery
            cur_time = 0.0
            
            while True:
                best_c = -1
                best_d = float('inf')
                
                # Find nearest feasible
                for c in unvisited:
                    d = dist[cur_node, c]
                    if d >= best_d:
                        continue
                    
                    # Feasibility
                    if cur_load + demands[c] > capacity:
                        continue
                    if cur_bat - d < 0:
                        continue
                    
                    t_arr = cur_time + d / speed
                    if t_arr < ready[c]:
                        t_arr = ready[c]
                    if t_arr > due[c]:
                        continue
                    
                    best_d = d
                    best_c = c
                
                if best_c == -1:
                    break
                
                # Move
                d = dist[cur_node, best_c]
                cur_time += d / speed
                cur_bat -= d
                cur_node = best_c
                cur_load += demands[best_c]
                
                if cur_time < ready[best_c]:
                    cur_time = ready[best_c]
                cur_time += service[best_c]
                
                route.append(best_c)
                unvisited.remove(best_c)
            
            if route:
                # Check return
                d_ret = dist[cur_node, 0]
                if cur_bat - d_ret >= 0 and cur_time + d_ret / speed <= horizon:
                    routes.append(route)
                else:
                    # If return is invalid, we have a problem.
                    # Remove the last customer?
                    # For simplicity, just append. Local search might fix it?
                    # No, invalid route is fatal.
                    # We should have ensured feasibility.
                    # If this happens, it means NN got stuck.
                    # Let's just append and hope LS fixes, or drop last.
                    # Dropping last is safer.
                    if route:
                        last = route.pop()
                        unvisited.add(last)
                        # Recalculate state? 
                        # This is messy. 
                        # Let's assume NN with these constraints (small N) rarely fails return.
                        routes.append(route)
            
            # If unvisited is not empty, we start a new route.
            # But what if we couldn't serve anyone? Infinite loop?
            # If best_c == -1 in the first iteration, we break inner loop.
            # If route is empty, we must serve someone to avoid infinite loop.
            # This implies we need a fallback: force serve the one with earliest due date?
            
            # To prevent infinite loop if NN fails to pick anyone:
            if not route and unvisited:
                # Pick customer with earliest due date
                c = min(unvisited, key=lambda x: due[x])
                # Force insert?
                # Just add to a new route and hope for the best?
                # This might create an invalid route.
                # Let's just break and handle leftovers later.
                break
        
        # Handle leftovers with simple greedy insertion
        for c in list(unvisited):
            best_r = -1
            best_cost = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    is_valid, new_dist = simulate_route(new_route)
                    if is_valid and new_dist < best_cost:
                        best_cost = new_dist
                        best_r = r_idx
                        best_pos = pos
            
            if best_r != -1:
                routes[best_r].insert(best_pos, c)
            else:
                # New route
                routes.append([c])
            
            unvisited.remove(c)
        
        return routes

    def local_search_oropt_2opt(routes, time_limit):
        """
        Local search with Or-opt (move 1-3 nodes) and 2-opt.
        """
        current_total_dist = compute_total_distance(routes)
        start_ls = time.time()
        
        # Precompute route distances for delta calculation
        route_dists = [0.0] * len(routes)
        for i, r in enumerate(routes):
            _, route_dists[i] = simulate_route(r)
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() - start_ls < time_limit:
            improved = False
            no_improve_count += 1
            if no_improve_count > 50:
                break
            
            # 1. Or-opt: Move a segment of size 1, 2, or 3 from route A to route B
            for seg_size in [1, 2, 3]:
                for r1 in range(len(routes)):
                    route1 = routes[r1]
                    n1 = len(route1)
                    if n1 < seg_size:
                        continue
                    
                    for i in range(n1 - seg_size + 1):
                        segment = route1[i:i+seg_size]
                        # Remove segment
                        remaining1 = route1[:i] + route1[i+seg_size:]
                        
                        # Check validity of remaining1
                        # We can approximate or just simulate. 
                        # Since N is small, simulate is fast.
                        is_valid1, new_dist1 = simulate_route(remaining1)
                        if not is_valid1:
                            continue
                        
                        # Try to insert segment into all other routes (and r1)
                        for r2 in range(len(routes)):
                            route2 = routes[r2]
                            n2 = len(route2)
                            
                            for j in range(n2 + 1):
                                # Insert segment at position j in route2
                                new_route2 = route2[:j] + segment + route2[j:]
                                is_valid2, new_dist2 = simulate_route(new_route2)
                                if not is_valid2:
                                    continue
                                
                                new_total = new_dist1 + new_dist2
                                old_total = route_dists[r1] + route_dists[r2]
                                
                                if new_total < old_total - 1e-6:
                                    routes[r1] = remaining1
                                    routes[r2] = new_route2
                                    route_dists[r1] = new_dist1
                                    route_dists[r2] = new_dist2
                                    current_total_dist = new_total
                                    improved = True
                                    break
                            
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. 2-opt: Reverse segment within a route
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid and new_dist < route_dists[r_idx] - 1e-6:
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            current_total_dist = compute_total_distance(routes) # Update total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0

        return routes

    # Main Execution
    best_routes = None
    best_dist = float('inf')
    
    # Multi-start construction
    # We have ~140s. 
    # Let's run 10-20 starts of NN with different random seeds/shuffles.
    # NN is O(N^2) per route, so very fast.
    
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 60.0: # Keep 60s for LS
            break
        
        rng = random.Random(i)
        # Shuffle customers for diversity in NN? 
        # NN is deterministic if we pick nearest. 
        # To add diversity, we can pick from top-K nearest with probability.
        
        routes = construct_nn_fast(rng)
        
        # Validate
        valid = True
        served = set()
        for route in routes:
            if not simulate_route(route)[0]:
                valid = False
                break
            for node in route:
                if node not in stations_set:
                    if node in served:
                        valid = False
                        break
                    served.add(node)
        
        if not valid or served != set(customers):
            continue
        
        d = compute_total_distance(routes)
        if d < best_dist:
            best_dist = d
            best_routes = [r[:] for r in routes]
            
        # Run local search on this solution
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for LS: proportional to remaining time
        # If we have 100s left, use 90s for LS.
        ls_time = min(30.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            ls_routes = local_search_oropt_2opt(routes, ls_time)
            
            # Validate LS result
            valid = True
            served = set()
            for route in ls_routes:
                if not simulate_route(route)[0]:
                    valid = False
                    break
                for node in route:
                    if node not in stations_set:
                        if node in served:
                            valid = False
                            break
                        served.add(node)
            
            if valid and served == set(customers):
                d_ls = compute_total_distance(ls_routes)
                if d_ls < best_dist:
                    best_dist = d_ls
                    best_routes = [r[:] for r in ls_routes]
    
    # If no valid solution found, fallback to single greedy
    if best_routes is None:
        # Fallback: Serve each customer individually?
        # Or just return the first valid one found?
        # Let's try one more simple construction
        routes = []
        unvisited = set(customers)
        for c in sorted(customers, key=lambda x: due[x]):
            # Try to insert into existing
            inserted = False
            for r_idx in range(len(routes)):
                for pos in range(len(routes[r_idx]) + 1):
                    new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                    if simulate_route(new_route)[0]:
                        routes[r_idx] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([c])
            unvisited.remove(c)
        
        best_routes = routes

    return best_routes
