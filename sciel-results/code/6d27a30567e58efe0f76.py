import numpy as np
import time
import random
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Extract data
    coords = np.array(instance["coords"], dtype=float)
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

    def build_route(cust_seq):
        """
        Given a sequence of customer IDs, insert charging stops as needed.
        Returns (route_list, total_distance) or (None, inf) if infeasible.
        """
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
                best_st = None
                best_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0

                for st in stations:
                    d_to_st = D[prev, st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    rt = max(0.0, (max_battery - bat_at_st) * recharge_rate)
                    t_st += rt
                    bat_st = max_battery
                    d_st_cust = D[st, cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    extra_dist = d_to_st + d_st_cust
                    if extra_dist < best_cost - 1e-9:
                        best_cost = extra_dist
                        best_st = st
                        best_t_cust = t_cust + service[cust]
                        best_bat_after = bat_st - d_st_cust

                if best_st is None:
                    return None, float('inf')

                route.append(best_st)
                route.append(cust)
                t = best_t_cust
                bat = best_bat_after
                load += demands[cust]
                total_dist += best_cost
                prev = cust

        # Return to depot
        d_ret = D[prev, 0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None, float('inf')
            total_dist += d_ret
        else:
            found = False
            for st in stations:
                d_to_st = D[prev, st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                rt = max(0.0, (max_battery - bat_at_st) * recharge_rate)
                t_st += rt
                bat_st = max_battery
                d_st_dep = D[st, 0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                total_dist += d_to_st + d_st_dep
                route.append(st)
                found = True
                break
            if not found:
                return None, float('inf')

        return route, total_dist

    def simulate_route(route):
        """Simulate a route, return total_distance if valid, else inf."""
        if not route:
            return 0.0
        t = 0.0
        bat = max_battery
        load = 0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = D[prev, node]
            total_dist += d
            if bat < d - 1e-9:
                return float('inf')
            t += d / speed
            bat -= d
            
            if node in station_set:
                deficit = max_battery - bat
                if deficit > 0:
                    t += deficit * recharge_rate
                bat = max_battery
            else:
                load += demands[node]
                if load > capacity:
                    return float('inf')
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return float('inf')
                t += service[node]
            
            prev = node
        
        d_ret = D[prev, 0]
        total_dist += d_ret
        if bat < d_ret - 1e-9:
            return float('inf')
        t += d_ret / speed
        if t > horizon + 1e-9:
            return float('inf')
        return total_dist

    def route_to_custs(route):
        return [node for node in route if node not in station_set]

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

    # 1. Construction: Sweep + Nearest Neighbor Hybrid
    def construct_initial():
        # Sort customers by angle from depot for sweep
        depot = 0
        angles = []
        for c in customers:
            dx = coords[c][0] - coords[depot][0]
            dy = coords[c][1] - coords[depot][1]
            ang = math.atan2(dy, dx)
            angles.append((ang, c))
        angles.sort(key=lambda x: x[0])
        
        # Try multiple starting points in the sorted list to find a good partition
        # But simpler: just use the sorted order and greedily pack routes
        # Or use the standard sweep: start at each customer in sorted order?
        # Let's stick to a robust heuristic: 
        # We will iterate through the sorted customers and try to add them to existing routes.
        # If it doesn't fit, open a new route.
        
        # However, simple sweep can be bad if capacity is tight.
        # Let's use a "Sweep then Optimize" approach.
        
        sorted_custs = [c for _, c in angles]
        
        routes = []
        unvisited = set(customers)
        
        # First pass: Greedy packing based on sweep order
        # This ensures spatial locality.
        for c in sorted_custs:
            if c not in unvisited:
                continue
            # Try to insert c into an existing route
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            
            # For speed, we limit insertion checks to a few best candidates
            # Or just check all if routes are few.
            
            for i, r in enumerate(routes):
                if not r:
                    continue
                # Try inserting at each position
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [c] + r[pos:]
                    # Quick check: capacity
                    if sum(demands[x] for x in new_r) > capacity:
                        continue
                    # Full check
                    nr, cost = build_route(new_r)
                    if nr is None:
                        continue
                    # Calculate delta cost
                    # We need the old cost of r. We can store it or recompute.
                    # For simplicity in construction, recompute is ok if N is small.
                    # But we can approximate.
                    old_r, old_cost = build_route(r)
                    if old_r is None:
                        continue
                    delta = cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = i
                        best_pos = pos
            
            if best_route_idx != -1:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
                unvisited.remove(c)
            else:
                # Start new route
                routes.append([c])
                unvisited.remove(c)
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        return routes

    initial_cust_routes = construct_initial()
    
    # Fallback if construction failed or is invalid
    current_cost = evaluate_cust_routes(initial_cust_routes)
    if current_cost == float('inf') or not initial_cust_routes:
        # Fallback: 1 customer per route
        initial_cust_routes = [[c] for c in customers]
        current_cost = evaluate_cust_routes(initial_cust_routes)
        
    best_cust_routes = [r[:] for r in initial_cust_routes]
    best_cost = current_cost
    current_cust_routes = [r[:] for r in initial_cust_routes]

    # 2. Local Search: Simulated Annealing
    # Moves: Relocate, Swap, 2-opt (within route), Or-opt (move block of 2-3)
    
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    # Precompute neighbors for faster 2-opt? No, N is small.
    
    while T > min_T and time.time() < deadline - 5:
        move_type = random.choice(['relocate', 'swap', '2opt', 'oropt'])
        
        new_cust_routes = [r[:] for r in current_cust_routes]
        valid_move = False
        
        if move_type == 'relocate':
            if len(new_cust_routes) < 2:
                continue
            # Pick a route with > 1 customer
            non_empty = [i for i, r in enumerate(new_cust_routes) if len(r) > 1]
            if not non_empty:
                continue
            src_idx = random.choice(non_empty)
            cust_pos = random.randint(0, len(new_cust_routes[src_idx]) - 1)
            cust = new_cust_routes[src_idx][cust_pos]
            
            dest_idx = random.randint(0, len(new_cust_routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(new_cust_routes) - 1)
            
            ins_pos = random.randint(0, len(new_cust_routes[dest_idx]))
            
            new_cust_routes[src_idx].pop(cust_pos)
            new_cust_routes[dest_idx].insert(ins_pos, cust)
            valid_move = True
            
        elif move_type == 'swap':
            if len(new_cust_routes) < 2:
                continue
            non_empty = [i for i, r in enumerate(new_cust_routes) if len(r) > 0]
            if len(non_empty) < 2:
                continue
            src_idx = random.choice(non_empty)
            dest_idx = random.choice(non_empty)
            while dest_idx == src_idx:
                dest_idx = random.choice(non_empty)
            
            c1_pos = random.randint(0, len(new_cust_routes[src_idx]) - 1)
            c2_pos = random.randint(0, len(new_cust_routes[dest_idx]) - 1)
            
            c1 = new_cust_routes[src_idx][c1_pos]
            c2 = new_cust_routes[dest_idx][c2_pos]
            
            new_cust_routes[src_idx][c1_pos] = c2
            new_cust_routes[dest_idx][c2_pos] = c1
            valid_move = True
            
        elif move_type == '2opt':
            # 2-opt within a single route
            non_empty = [i for i, r in enumerate(new_cust_routes) if len(r) > 2]
            if not non_empty:
                continue
            r_idx = random.choice(non_empty)
            r = new_cust_routes[r_idx]
            i = random.randint(0, len(r) - 2)
            j = random.randint(i + 1, len(r) - 1)
            
            # Reverse segment i..j
            # r[i..j] becomes r[j], r[j-1], ..., r[i]
            segment = r[i:j+1]
            segment.reverse()
            new_r = r[:i] + segment + r[j+1:]
            new_cust_routes[r_idx] = new_r
            valid_move = True
            
        elif move_type == 'oropt':
            # Move a block of size 1, 2, or 3 within the same route or to another
            # For simplicity, let's do intra-route Or-opt first
            non_empty = [i for i, r in enumerate(new_cust_routes) if len(r) > 3]
            if not non_empty:
                continue
            r_idx = random.choice(non_empty)
            r = new_cust_routes[r_idx]
            size = random.choice([1, 2, 3])
            if len(r) <= size:
                continue
            
            # Pick start position
            start = random.randint(0, len(r) - size)
            block = r[start:start+size]
            
            # Remove block
            new_r = r[:start] + r[start+size:]
            
            # Pick insertion position (different from original)
            ins_pos = random.randint(0, len(new_r))
            if ins_pos == start:
                # If we insert right back, no change
                ins_pos = (ins_pos + 1) % (len(new_r) + 1)
            
            new_r = new_r[:ins_pos] + block + new_r[ins_pos:]
            new_cust_routes[r_idx] = new_r
            valid_move = True
            
        if not valid_move:
            continue
            
        new_cost = evaluate_cust_routes(new_cust_routes)
        
        if new_cost == float('inf'):
            # Revert
            T *= cooling
            continue
            
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cust_routes = new_cust_routes
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_cust_routes = [r[:] for r in current_cust_routes]
        else:
            # Revert
            pass
            
        T *= cooling
        
        # Occasionally do a more aggressive restart or perturbation?
        # No, SA is fine.

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
    
    # Final validation
    # Ensure all customers are covered
    covered = set()
    for r in final_routes:
        for node in r:
            if node not in station_set:
                covered.add(node)
                
    missing = [c for c in customers if c not in covered]
    if missing:
        # This shouldn't happen, but if it does, add them individually
        for c in missing:
            r, cost = build_route([c])
            if r is not None:
                final_routes.append(r)
    
    return final_routes
