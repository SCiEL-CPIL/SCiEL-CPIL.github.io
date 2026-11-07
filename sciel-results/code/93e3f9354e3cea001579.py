import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dist = cdist(coords, coords)
    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route_nodes(nodes_seq):
        """
        Simulates a route given a sequence of node IDs (customers and stations).
        Returns (total_distance, final_time, is_valid, final_battery).
        """
        if not nodes_seq:
            return 0.0, 0.0, True, battery_cap

        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        last = 0

        for node in nodes_seq:
            d = dist[last, node]
            
            if bat < d:
                return 0.0, 0.0, False, bat

            total_dist += d
            bat -= d
            t += d / speed
            last = node

            if node in station_set:
                # Charge to full
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                # Customer
                if t > due[node]:
                    return 0.0, 0.0, False, bat
                t = max(t, ready[node]) + service[node]

        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            return 0.0, 0.0, False, bat
        
        total_dist += d_back
        bat -= d_back
        t += d_back / speed

        if t > horizon:
            return 0.0, 0.0, False, bat

        return total_dist, t, True, bat

    def get_best_station_insertion(last_node, next_node, current_battery, current_time):
        """
        Given we are at last_node and need to go to next_node, but battery is insufficient,
        find the best station to visit in between.
        Returns (station_id, extra_dist, extra_time) or None if impossible.
        """
        d_direct = dist[last_node, next_node]
        best_s = -1
        best_extra_dist = float('inf')
        best_extra_time = 0.0

        for s in stations:
            d1 = dist[last_node, s]
            d2 = dist[s, next_node]
            
            if current_battery < d1:
                continue
            if battery_cap < d2:
                continue
            
            # Check time feasibility roughly (arrival at s + charge + travel to next)
            t_arr_s = current_time + d1 / speed
            t_charge = (battery_cap - (current_battery - d1)) * recharge_rate
            t_arr_next = t_arr_s + t_charge + d2 / speed
            
            # We must arrive at next_node by due[next_node]
            # Note: if next_node is a customer, we wait until ready. If station, no wait.
            # For simplicity in insertion, we just check if we can physically get there.
            # The full simulation will check time windows.
            
            extra_dist = d1 + d2 - d_direct
            if extra_dist < best_extra_dist:
                best_extra_dist = extra_dist
                best_extra_time = t_arr_next - current_time
                best_s = s

        if best_s == -1:
            return None
        return (best_s, best_extra_dist, best_extra_time)

    def build_route_from_customers(customers_seq):
        """
        Takes a sequence of customers and inserts necessary stations to make it feasible.
        Returns (list_of_nodes, total_dist, is_valid).
        """
        if not customers_seq:
            return [], 0.0, True
        
        # Check capacity
        total_demand = sum(demands[c] for c in customers_seq)
        if total_demand > capacity:
            return [], 0.0, False

        nodes = []
        t = 0.0
        bat = battery_cap
        last = 0
        total_dist = 0.0

        for c in customers_seq:
            d_direct = dist[last, c]
            
            if bat < d_direct:
                # Need to charge
                res = get_best_station_insertion(last, c, bat, t)
                if res is None:
                    return [], 0.0, False
                
                s, extra_d, extra_t = res
                nodes.append(s)
                
                # Update state to station
                d1 = dist[last, s]
                total_dist += d1
                bat -= d1
                t += d1 / speed
                
                # Charge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = s
                
                # Go to customer
                d2 = dist[last, c]
                total_dist += d2
                bat -= d2
                t += d2 / speed
                last = c
            else:
                total_dist += d_direct
                bat -= d_direct
                t += d_direct / speed
                last = c
            
            nodes.append(c)
            
            # Time window check
            if t > due[c]:
                return [], 0.0, False
            t = max(t, ready[c]) + service[c]

        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            res = get_best_station_insertion(last, 0, bat, t)
            if res is None:
                return [], 0.0, False
            
            s, extra_d, extra_t = res
            nodes.append(s)
            
            d1 = dist[last, s]
            total_dist += d1
            bat -= d1
            t += d1 / speed
            
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = s
            
            d2 = dist[last, 0]
            total_dist += d2
            bat -= d2
            t += d2 / speed
        else:
            total_dist += d_back
            bat -= d_back
            t += d_back / speed

        if t > horizon:
            return [], 0.0, False

        return nodes, total_dist, True

    def evaluate_solution(routes_customers):
        """
        Takes a list of customer lists (routes), expands them with stations,
        and returns (total_distance, list_of_node_routes, is_valid).
        """
        total_dist = 0.0
        final_routes = []
        
        for r in routes_customers:
            nodes, d, valid = build_route_from_customers(r)
            if not valid:
                return 0.0, [], False
            total_dist += d
            final_routes.append(nodes)
            
        return total_dist, final_routes, True

    def split_permutation(perm):
        """
        Splits a permutation of customers into routes based on capacity and feasibility.
        Greedy first-fit decreasing style but respecting order for now?
        No, let's do a simple greedy split:
        Start a route. Add customers from perm. If capacity exceeded or infeasible, close route.
        """
        routes = []
        current_route = []
        current_demand = 0
        
        for c in perm:
            if current_demand + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_demand = 0
            
            # Tentatively add c
            test_route = current_route + [c]
            # Quick check: is it even possible?
            # We don't want to run full simulation for every single addition if it's slow.
            # But for 25 customers, it's fast enough.
            _, _, valid = build_route_from_customers(test_route)
            
            if valid:
                current_route = test_route
                current_demand += demands[c]
            else:
                # Cannot add c to current route.
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_demand = 0
                
                # Start new route with c
                # Check if single customer is feasible
                _, _, valid_single = build_route_from_customers([c])
                if not valid_single:
                    # This shouldn't happen for a single customer unless depot->c->depot is infeasible
                    # which is unlikely with battery 300 and coords 100x100.
                    # But if it does, we have a problem.
                    pass 
                current_route = [c]
                current_demand = demands[c]
        
        if current_route:
            routes.append(current_route)
            
        return routes

    # --- Initial Solution Generation ---
    # 1. Sweep / Angle based
    def angle_sort_customers():
        # Sort by angle from depot
        angles = [math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]) for c in customers]
        sorted_indices = sorted(range(len(angles)), key=lambda k: angles[k])
        return [customers[i] for i in sorted_indices]

    # 2. Random permutations
    def random_perm():
        p = customers[:]
        random.shuffle(p)
        return p

    # 3. Nearest Neighbor
    def nearest_neighbor():
        remaining = set(customers)
        route = []
        current = 0
        while remaining:
            # Find nearest customer that fits capacity? 
            # Actually, let's just find nearest. Capacity is handled by split.
            # But we want to build routes.
            # Let's do a simple NN tour first, then split.
            best_c = -1
            best_d = float('inf')
            for c in remaining:
                d = dist[current, c]
                if d < best_d:
                    best_d = d
                    best_c = c
            if best_c == -1: break
            route.append(best_c)
            remaining.remove(best_c)
            current = best_c
        return route

    # Generate initial population
    pop_size = 100
    population = []
    
    # Add some structured solutions
    population.append(angle_sort_customers())
    population.append(nearest_neighbor())
    
    for _ in range(pop_size - 2):
        population.append(random_perm())

    best_perm = None
    best_dist = float('inf')
    best_routes_nodes = None
    best_routes_customers = None

    def update_best(perm):
        nonlocal best_perm, best_dist, best_routes_nodes, best_routes_customers
        routes_cust = split_permutation(perm)
        d, routes_nodes, valid = evaluate_solution(routes_cust)
        if valid and d < best_dist:
            best_dist = d
            best_perm = perm[:]
            best_routes_nodes = routes_nodes
            best_routes_customers = routes_cust

    # Evaluate initial population
    for p in population:
        update_best(p)

    if best_perm is None:
        # Fallback
        return [[c] for c in customers]

    # --- Local Search ---
    # We operate on the permutation.
    # Moves:
    # 1. Swap two customers.
    # 2. Or-opt: Move a sequence of 1-3 customers to a different position.
    # 3. Inter-route move: Move a customer from one route to another?
    #    Since we split based on the permutation, moving a customer in the perm changes the routes.
    #    This is a "reordering" search.
    #    To explicitly change route assignments, we can try to swap customers that are far apart in the perm.
    
    def local_search(perm):
        improved = True
        iterations = 0
        max_iterations = 2000 # Cap iterations to prevent infinite loops
        
        while improved and iterations < max_iterations and time.time() < deadline - 10:
            improved = False
            iterations += 1
            old_dist = best_dist
            old_perm = perm[:]
            
            # Try random moves
            n_moves = 50
            for _ in range(n_moves):
                new_perm = perm[:]
                move_type = random.random()
                
                if move_type < 0.5:
                    # Swap
                    i, j = random.sample(range(len(new_perm)), 2)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                elif move_type < 0.8:
                    # Or-opt (move 1-3 items)
                    length = random.randint(1, 3)
                    i = random.randint(0, len(new_perm) - length)
                    j = random.randint(0, len(new_perm) - length)
                    if i == j: continue
                    segment = new_perm[i:i+length]
                    # Remove
                    del new_perm[i:i+length]
                    # Insert
                    # Adjust j if i < j
                    if i < j:
                        j -= length
                    new_perm[j:j] = segment
                else:
                    # Reverse segment (2-opt like)
                    i, j = random.sample(range(len(new_perm)), 2)
                    if i > j: i, j = j, i
                    new_perm[i:j+1] = new_perm[i:j+1][::-1]
                
                update_best(new_perm)
                if best_dist < old_dist - 1e-6:
                    perm = new_perm
                    improved = True
                    break # Restart search from new best
                    
        return perm

    # Run local search
    best_perm = local_search(best_perm)
    
    # If time remains, do more rounds or larger population
    if time.time() < deadline - 30:
        # Generate more random solutions and try to improve them
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            p = random_perm()
            # Quick local search
            p = local_search(p)
            update_best(p)
            
    # Final evaluation to ensure we have the best nodes
    if best_routes_nodes is None:
        routes_cust = split_permutation(best_perm)
        d, routes_nodes, valid = evaluate_solution(routes_cust)
        if valid:
            best_routes_nodes = routes_nodes
        else:
            # Fallback
            best_routes_nodes = [[c] for c in customers]
            
    return best_routes_nodes
