"""
Strategy: Simulated Annealing over a global customer permutation.
Representation: A single list of 25 customers.
Decoding: Greedy split into feasible routes respecting battery, capacity, and time windows.
Operators: 2-opt (inversion), Or-opt (segment move), Cross (recombination of two permutations).
This addresses the structural change requirement by moving from a direct multi-route representation
to a single-permutation representation with a stochastic global search method.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 145.0 

    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))

    # Fast feasibility check and distance calculation for a list of node IDs
    # Returns (is_feasible, total_distance)
    def evaluate_route(route):
        if not route:
            return True, 0.0
        
        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0
        
        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d
            
            if bat < -1e-9:
                return False, 0.0
            
            # Handle station recharge
            if v in stations:
                missing = battery - bat
                if missing < 0:
                    missing = 0
                t += missing * recharge_rate
                bat = battery
            else:
                # Customer
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, 0.0
                t += service[v]
            
            prev = v
        
        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d
        
        if bat < -1e-9:
            return False, 0.0
        if t > horizon + 1e-9:
            return False, 0.0
        
        return True, total_dist

    # Split a permutation into feasible routes
    # Returns (routes, total_distance) or (None, inf) if infeasible
    def decode(permutation):
        routes = []
        current_route = []
        current_dist = 0.0
        current_load = 0
        current_t = 0.0
        current_bat = battery
        prev = 0
        
        # We simulate the route construction.
        # If a customer doesn't fit, we close the current route and start a new one.
        # Note: This greedy split is not optimal, but it's fast.
        # To make it better, we could try to insert the customer in the middle, but that's complex.
        # For now, simple append. If it fails, we try to start a new route.
        
        # Actually, a simple append might fail even if the permutation is good, 
        # because the order in the permutation dictates the order in the route.
        # So we just follow the permutation order.
        
        for v in permutation:
            # Try to add v to current_route
            trial_route = current_route + [v]
            feasible, d = evaluate_route(trial_route)
            
            if feasible:
                # Update state efficiently? 
                # evaluate_route recalculates everything. For 25 nodes, it's fast enough.
                current_route = trial_route
                # We don't strictly need to track current_t, current_bat separately 
                # if we rely on evaluate_route for the final check, 
                # but we need to know if it fits.
                pass
            else:
                # Close current route
                if current_route:
                    # Check if current_route is feasible on its own (it should be, as we built it up)
                    # Actually, we only add if feasible, so current_route is always feasible.
                    routes.append(current_route)
                    current_dist += d # Wait, d is for trial_route which failed. 
                                      # We need the distance of the closed route.
                    # Let's re-evaluate the closed route to get its distance.
                    _, closed_d = evaluate_route(current_route)
                    current_dist += closed_d
                    current_route = []
                
                # Start new route with v
                # Check if single node route is feasible
                single_feasible, single_d = evaluate_route([v])
                if not single_feasible:
                    return None, float('inf')
                
                current_route = [v]
                # We don't add single_d to current_dist yet, only when closed.
                # Or we can track it. Let's track it.
                # But we need to be careful: current_dist accumulates completed routes.
                # Let's just accumulate at the end or track properly.
                # Simpler: just store routes, calculate distance at the end.
                # But we need distance for SA cost function.
                # Let's calculate total distance at the end of decode.
        
        # Close last route
        if current_route:
            routes.append(current_route)
        
        # Calculate total distance
        total_dist = 0.0
        for r in routes:
            _, d = evaluate_route(r)
            total_dist += d
            
        return routes, total_dist

    # Initial Solution: Sort by due date (common heuristic)
    initial_perm = sorted(customers, key=lambda x: due[x])
    best_routes, best_dist = decode(initial_perm)
    if best_routes is None:
        # Fallback: One customer per route
        best_routes = [[v] for v in customers]
        best_dist = sum(evaluate_route([v])[1] for v in customers)
    
    current_perm = initial_perm[:]
    current_routes, current_dist = decode(current_perm)
    
    # If initial is infeasible, try random permutations until we find a feasible one
    attempts = 0
    while current_routes is None and attempts < 100:
        current_perm = customers[:]
        random.shuffle(current_perm)
        current_routes, current_dist = decode(current_perm)
        attempts += 1
        
    if current_routes is None:
        # Last resort: one per route
        current_perm = customers[:]
        current_routes = [[v] for v in current_perm]
        current_dist = sum(evaluate_route([v])[1] for v in current_perm)
        
    best_perm = current_perm[:]
    best_dist = current_dist
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Second best permutation for Cross operator
    # We need a population or at least a second good solution.
    # Let's maintain a simple population of size 1 (current) and 1 (best) 
    # and occasionally generate a new random one to cross with.
    # Or just use the best solution as the other parent for Cross.
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        
        # Choose operator
        r = random.random()
        new_perm = current_perm[:]
        
        if r < 0.5:
            # 2-opt
            i, j = sorted(random.sample(range(len(current_perm)), 2))
            new_perm[i:j+1] = reversed(new_perm[i:j+1])
        elif r < 0.8:
            # Or-opt (move 1-3 nodes)
            k = random.randint(1, 3)
            i = random.randint(0, len(current_perm) - k)
            segment = new_perm[i:i+k]
            del new_perm[i:i+k]
            j = random.randint(0, len(new_perm))
            new_perm[j:j] = segment
        else:
            # Cross
            # Take prefix from best_perm, suffix from current_perm?
            # Or random crossover.
            # Let's do a uniform crossover with current and best.
            # This explores the space between current and best.
            crossover_point = random.randint(1, len(current_perm) - 1)
            prefix = best_perm[:crossover_point]
            # The rest must be a permutation of the remaining elements
            remaining = [x for x in current_perm if x not in prefix]
            # To preserve some order from current, we can just append remaining in current order
            # But that might be too similar. Let's shuffle remaining slightly?
            # No, keep it simple: prefix from best, suffix from current (filtered).
            new_perm = prefix + remaining
            
        # Evaluate new permutation
        new_routes, new_dist = decode(new_perm)
        
        if new_routes is None:
            # Infeasible: reject (or accept with penalty, but reject is safer for validity)
            # We can try to repair, but for now reject.
            continue
            
        # Acceptance criterion
        delta = new_dist - current_dist
        if delta < 0:
            current_perm = new_perm
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_perm = current_perm[:]
                best_dist = current_dist
                best_routes = current_routes
        else:
            # Metropolis criterion
            if random.random() < math.exp(-delta / T):
                current_perm = new_perm
                current_routes = new_routes
                current_dist = new_dist
        
        T *= cooling_rate
        
        # Periodic restart or diversification?
        # If stuck for a while, maybe perturb best solution?
        # For now, standard SA.
        
    return best_routes
# EVOLVE-BLOCK-END
