import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    customers = list(range(1, n_customers + 1))
    
    def simulate_route(customers_seq):
        """
        Simulates a route of customers.
        Returns (total_distance, total_time, is_valid, node_sequence).
        node_sequence includes stations inserted for charging.
        """
        if not customers_seq:
            return 0.0, 0.0, True, []
            
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        last = 0
        nodes = []
        
        for c in customers_seq:
            d_direct = dist[last, c]
            
            if bat < d_direct:
                # Need to charge
                best_s = -1
                best_cost = float('inf')
                
                # Check all stations
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, c]
                    # Can we reach station?
                    if bat < d1:
                        continue
                    # Can we reach customer from station with full charge?
                    if battery_cap < d2:
                        continue
                    
                    cost = d1 + d2 - d_direct
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                        
                if best_s == -1:
                    return 0.0, 0.0, False, []
                
                # Go to station
                d1 = dist[last, best_s]
                total_dist += d1
                bat -= d1
                t += d1 / speed
                
                # Charge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = best_s
                nodes.append(best_s)
                
                # Go to customer
                d2 = dist[last, c]
                total_dist += d2
                bat -= d2
                t += d2 / speed
                last = c
            else:
                # Direct
                total_dist += d_direct
                bat -= d_direct
                t += d_direct / speed
                last = c
            
            nodes.append(c)
            
            # Time window check
            t_start = max(t, ready[c])
            if t_start > due[c]:
                return 0.0, 0.0, False, []
            t = t_start + service[c]
            
        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[last, s]
                d2 = dist[s, 0]
                if bat < d1 or battery_cap < d2:
                    continue
                cost = d1 + d2 - d_back
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            
            if best_s == -1:
                return 0.0, 0.0, False, []
            
            d1 = dist[last, best_s]
            total_dist += d1
            bat -= d1
            t += d1 / speed
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = best_s
            nodes.append(best_s)
            
            d2 = dist[last, 0]
            total_dist += d2
            bat -= d2
            t += d2 / speed
        else:
            total_dist += d_back
            bat -= d_back
            t += d_back / speed
            
        if t > horizon:
            return 0.0, 0.0, False, []
            
        return total_dist, t, True, nodes

    def evaluate_solution(routes):
        """
        Evaluates a list of customer lists.
        Returns (total_distance, is_valid).
        """
        total_dist = 0.0
        for route in routes:
            d, t, valid, _ = simulate_route(route)
            if not valid:
                return 0.0, False
            total_dist += d
        return total_dist, True

    # 1. Clarke-Wright Savings Construction
    def construct_initial():
        # Start with each customer in its own route
        routes = [[c] for c in customers]
        
        # Calculate savings: s(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
        # We want to merge routes if saving is positive and feasible
        # To simplify, we pre-compute savings for all pairs
        # But we need to check feasibility (capacity, battery, time)
        
        # Sort customers by angle? Or just use savings.
        # Standard CW:
        savings = []
        for i in customers:
            for j in customers:
                if i != j:
                    s = dist[0, i] + dist[0, j] - dist[i, j]
                    savings.append((s, i, j))
        
        savings.sort(reverse=True)
        
        # Map customer to route index
        cust_to_route = {c: i for i, route in enumerate(routes) for c in route}
        
        for s, i, j in savings:
            if s <= 0:
                break
            if i == j:
                continue
            
            ri = cust_to_route[i]
            rj = cust_to_route[j]
            
            if ri == rj:
                continue
            
            # Check if i is at the end of route ri and j is at the start of route rj
            # Or i at start, j at end?
            # CW usually merges end of i-route with start of j-route
            route_i = routes[ri]
            route_j = routes[rj]
            
            # We can only merge if i is the last node of route_i and j is the first node of route_j
            # Or vice versa?
            # Let's check both orientations if needed, but standard CW assumes fixed direction.
            # Let's try: append route_j to route_i if i is last of ri and j is first of rj
            if route_i[-1] == i and route_j[0] == j:
                new_route = route_i + route_j
                # Check capacity
                if sum(demands[c] for c in new_route) <= capacity:
                    d, t, valid, _ = simulate_route(new_route)
                    if valid:
                        # Merge
                        routes[ri] = new_route
                        routes[rj] = []
                        for c in new_route:
                            cust_to_route[c] = ri
                        continue
            
            # Try reverse: append route_i to route_j if j is last of rj and i is first of ri
            if route_j[-1] == j and route_i[0] == i:
                new_route = route_j + route_i
                if sum(demands[c] for c in new_route) <= capacity:
                    d, t, valid, _ = simulate_route(new_route)
                    if valid:
                        routes[rj] = new_route
                        routes[ri] = []
                        for c in new_route:
                            cust_to_route[c] = rj
                        continue
        
        # Clean up empty routes
        routes = [r for r in routes if r]
        
        # Ensure all customers are covered (should be, but just in case)
        covered = set(c for r in routes for c in r)
        missing = [c for c in customers if c not in covered]
        for c in missing:
            routes.append([c])
            
        return routes

    # 2. Local Search (VNS)
    def local_search(routes):
        best_routes = routes[:]
        best_dist, _ = evaluate_solution(best_routes)
        
        # Neighborhood 1: Intra-route 2-Opt
        # Neighborhood 2: Inter-route Relocate
        # Neighborhood 3: Inter-route Exchange (2-opt between routes)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Intra-route 2-Opt
            for i in range(len(best_routes)):
                route = best_routes[i]
                n = len(route)
                if n < 3:
                    continue
                
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        # Reverse segment a..b
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        
                        # Quick check: capacity is same.
                        # Check feasibility
                        d_new, t_new, valid_new, _ = simulate_route(new_route)
                        
                        # Calculate old distance for this route specifically?
                        # We have best_dist for total. We need to compare marginal cost.
                        # To be safe and simple, evaluate whole solution if promising?
                        # No, that's expensive.
                        # Let's compute old route dist.
                        d_old, t_old, valid_old, _ = simulate_route(route)
                        
                        if valid_new and d_new < d_old - 1e-9:
                            # Check if total solution is still valid (it should be, since only this route changed)
                            # But wait, simulate_route checks time windows relative to start time 0.
                            # If we change one route, others are unaffected.
                            best_routes[i] = new_route
                            best_dist = best_dist - d_old + d_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
                
            # 2. Inter-route Relocate
            for i in range(len(best_routes)):
                for j in range(len(best_routes)):
                    if i == j:
                        continue
                    for k in range(len(best_routes[i])):
                        c = best_routes[i][k]
                        # Remove c from i
                        route_i_without = best_routes[i][:k] + best_routes[i][k+1:]
                        d_i_old, _, valid_i_old, _ = simulate_route(best_routes[i])
                        d_i_new, _, valid_i_new, _ = simulate_route(route_i_without)
                        
                        if not valid_i_new:
                            continue
                            
                        # Try to insert c into route j at all positions
                        route_j = best_routes[j]
                        d_j_old, _, valid_j_old, _ = simulate_route(route_j)
                        
                        best_pos = -1
                        best_d_j_new = float('inf')
                        
                        for pos in range(len(route_j) + 1):
                            new_route_j = route_j[:pos] + [c] + route_j[pos:]
                            # Check capacity
                            if sum(demands[x] for x in new_route_j) > capacity:
                                continue
                            d, t, valid, _ = simulate_route(new_route_j)
                            if valid and d < best_d_j_new:
                                best_d_j_new = d
                                best_pos = pos
                                
                        if best_pos != -1:
                            total_new_dist = best_dist - d_i_old - d_j_old + d_i_new + best_d_j_new
                            if total_new_dist < best_dist - 1e-9:
                                best_routes[i] = route_i_without
                                best_routes[j] = best_routes[j][:best_pos] + [c] + best_routes[j][best_pos:]
                                best_dist = total_new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 3. Inter-route Exchange (Swap 2 nodes)
            # This is more complex. Let's try simple 1-for-1 swap first.
            for i in range(len(best_routes)):
                for j in range(i+1, len(best_routes)):
                    for k in range(len(best_routes[i])):
                        for l in range(len(best_routes[j])):
                            c1 = best_routes[i][k]
                            c2 = best_routes[j][l]
                            
                            new_i = best_routes[i][:]
                            new_j = best_routes[j][:]
                            new_i[k] = c2
                            new_j[l] = c1
                            
                            # Check capacity
                            if sum(demands[x] for x in new_i) > capacity or sum(demands[x] for x in new_j) > capacity:
                                continue
                            
                            d_i_old, _, _, _ = simulate_route(best_routes[i])
                            d_j_old, _, _, _ = simulate_route(best_routes[j])
                            
                            d_i_new, _, valid_i, _ = simulate_route(new_i)
                            d_j_new, _, valid_j, _ = simulate_route(new_j)
                            
                            if valid_i and valid_j:
                                total_new = best_dist - d_i_old - d_j_old + d_i_new + d_j_new
                                if total_new < best_dist - 1e-9:
                                    best_routes[i] = new_i
                                    best_routes[j] = new_j
                                    best_dist = total_new
                                    improved = True
                                    break
                        if improved: break
                    if improved: break
                if improved: break
            
        return best_routes, best_dist

    # Main Logic
    # Generate multiple initial solutions using CW with randomization or different starting points?
    # CW is deterministic. Let's run it once.
    # Then run Local Search.
    
    routes = construct_initial()
    
    # Verify initial solution
    d, valid = evaluate_solution(routes)
    if not valid:
        # Fallback: single customer per route
        routes = [[c] for c in customers]
        d, valid = evaluate_solution(routes)
        if not valid:
             # Should not happen if single customer is always feasible
             return [[c] for c in customers]

    # Run Local Search
    routes, d = local_search(routes)
    
    # Final Check and Conversion
    final_routes = []
    for r in routes:
        d, t, valid, nodes = simulate_route(r)
        if valid:
            final_routes.append(nodes)
        else:
            # Should not happen if local search maintained validity
            final_routes.append(r)
            
    return final_routes
