"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    # Helper to check if a route (list of node ids) is feasible
    # Returns (feasible, total_distance, total_time)
    def check_route(route):
        if not route:
            return True, 0.0, 0.0
            
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        prev = 0
        
        for node in route:
            d = dist[prev, node]
            total_dist += d
            bat -= d
            if bat < -1e-9:
                return False, total_dist, t
            t += d / speed
            
            if node in stations:
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                # Customer
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return False, total_dist, t
                t += service[node]
            
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        bat -= d
        if bat < -1e-9:
            return False, total_dist, t
        t += d / speed
        if t > horizon + 1e-9:
            return False, total_dist, t
            
        return True, total_dist, t

    def get_distance(route):
        if not route:
            return 0.0
        d = dist[0, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i+1]]
        d += dist[route[-1], 0]
        return d

    def get_demand(route):
        return sum(demands[n] for n in route if n not in stations)

    # Strategy: Simulated Annealing on a set of routes.
    # Representation: A list of lists. Each inner list is a route (list of customer/station ids).
    # We start with a greedy construction, then perform SA moves:
    # 1. Swap two customers in the same route.
    # 2. Move a customer from one route to another.
    # 3. 2-opt within a route.
    # 4. Or-opt within a route.
    # 5. Insert a station into a route (if needed for feasibility or to allow more customers? No, stations are only for charging. We can insert stations if battery is low).
    
    # Actually, inserting stations is a complex move because it changes the route structure.
    # Let's stick to moves that preserve the set of customers in each route, or move customers between routes.
    # For stations, we can have a separate move: "Insert station at position i in route r".
    
    # Initial Construction: Greedy Nearest Neighbor with constraints.
    # This is a different starting point than CW Savings.
    
    def construct_initial():
        unvisited = list(range(1, n_customers + 1))
        routes = []
        
        while unvisited:
            # Start a new route from depot
            route = []
            curr_node = 0
            curr_time = 0.0
            curr_bat = battery_cap
            curr_demand = 0
            
            # Try to add customers
            while unvisited:
                # Find nearest feasible customer
                best_c = -1
                best_dist = float('inf')
                
                for c in unvisited:
                    # Check if c can be added
                    d = dist[curr_node, c]
                    if curr_bat - d < 0:
                        continue
                    new_time = curr_time + d / speed
                    if new_time > due[c]:
                        continue
                    if curr_demand + demands[c] > capacity:
                        continue
                    
                    # Check if we can return to depot after adding c?
                    # Not strictly necessary for intermediate steps, but good for feasibility.
                    # We will check full feasibility at the end or during SA.
                    
                    if d < best_dist:
                        best_dist = d
                        best_c = c
                
                if best_c == -1:
                    break
                
                # Add best_c
                d = dist[curr_node, best_c]
                curr_time += d / speed
                curr_bat -= d
                curr_time = max(curr_time, ready[best_c])
                curr_time += service[best_c]
                curr_demand += demands[best_c]
                
                route.append(best_c)
                unvisited.remove(best_c)
                curr_node = best_c
                
            if route:
                routes.append(route)
            else:
                # Fallback: force add one customer if possible?
                # If no customer could be added, we have a problem.
                # But usually at least one is feasible.
                # If not, we just break and leave it? No, must serve all.
                # Let's just take the first unvisited and hope for the best, or handle in SA.
                # For now, break to avoid infinite loop, SA will fix.
                break
                
        return routes

    # Initial routes
    routes = construct_initial()
    
    # Ensure all customers are covered. If construct_initial failed, we need a fallback.
    # Let's check coverage.
    covered = set()
    for r in routes:
        for n in r:
            if n not in stations:
                covered.add(n)
    missing = [c for c in range(1, n_customers + 1) if c not in covered]
    if missing:
        # Create new routes for missing customers
        for c in missing:
            routes.append([c])
        
    # Evaluate initial solution
    def evaluate(routes):
        total_dist = 0
        valid = True
        for r in routes:
            feas, d, t = check_route(r)
            if not feas:
                valid = False
                # Penalty
                total_dist += 10000
            else:
                total_dist += d
            if get_demand(r) > capacity:
                valid = False
                total_dist += 10000
        return total_dist, valid

    curr_dist, curr_valid = evaluate(routes)
    best_routes = [list(r) for r in routes]
    best_dist = curr_dist
    best_valid = curr_valid
    
    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.995
    k = 0  # Iteration count
    
    # Helper to get a random route index
    def rand_route():
        return random.randint(0, len(routes) - 1)

    # Helper to get a random position in a route
    def rand_pos(route):
        return random.randint(0, len(route) - 1)

    # Moves:
    # 1. Swap two customers in same route
    # 2. Move customer from route i to route j
    # 3. 2-opt in a route
    # 4. Or-opt in a route
    # 5. Insert station in a route (if battery is low? Or just try random insertion)
    
    while T > T_min and time.time() < deadline:
        # Pick a random move
        move_type = random.randint(0, 4)
        
        new_routes = [list(r) for r in routes]
        
        if move_type == 0:
            # Swap
            if len(routes) > 0:
                r_idx = rand_route()
                r = new_routes[r_idx]
                if len(r) >= 2:
                    i, j = random.sample(range(len(r)), 2)
                    r[i], r[j] = r[j], r[i]
                    
        elif move_type == 1:
            # Move
            if len(routes) >= 2:
                r1_idx = rand_route()
                r2_idx = rand_route()
                while r2_idx == r1_idx:
                    r2_idx = rand_route()
                
                r1 = new_routes[r1_idx]
                r2 = new_routes[r2_idx]
                
                if len(r1) > 0:
                    i = rand_pos(r1)
                    node = r1[i]
                    # Only move customers, not stations?
                    # If we move a station, it might break charging.
                    # Let's only move customers.
                    if node in stations:
                        continue
                    r1.pop(i)
                    j = rand_pos(r2) if len(r2) > 0 else 0
                    r2.insert(j, node)
                    
        elif move_type == 2:
            # 2-opt
            if len(routes) > 0:
                r_idx = rand_route()
                r = new_routes[r_idx]
                if len(r) >= 2:
                    i, j = random.sample(range(len(r)), 2)
                    if i > j:
                        i, j = j, i
                    # Reverse segment i..j
                    r[i:j+1] = r[i:j+1][::-1]
                    
        elif move_type == 3:
            # Or-opt (move 1, 2, or 3 nodes)
            if len(routes) > 0:
                r_idx = rand_route()
                r = new_routes[r_idx]
                if len(r) >= 2:
                    length = random.randint(1, min(3, len(r)))
                    i = random.randint(0, len(r) - length)
                    segment = r[i:i+length]
                    r = r[:i] + r[i+length:]
                    j = random.randint(0, len(r))
                    r = r[:j] + segment + r[j:]
                    new_routes[r_idx] = r
                    
        elif move_type == 4:
            # Insert Station
            # Find a random route and a random position
            # Check if inserting a station here improves feasibility or distance?
            # For now, just try inserting a random station at a random position.
            # This is risky but might help with battery.
            if len(routes) > 0 and stations:
                r_idx = rand_route()
                r = new_routes[r_idx]
                s = random.choice(stations)
                # Don't insert if already present?
                if s not in r:
                    pos = random.randint(0, len(r))
                    r.insert(pos, s)
        
        new_dist, new_valid = evaluate(new_routes)
        
        # Acceptance
        delta = new_dist - curr_dist
        if new_valid and (not curr_valid or delta < 0):
            # Always accept improvement or recovery from invalid
            curr_dist = new_dist
            curr_valid = new_valid
            routes = new_routes
            if curr_valid and curr_dist < best_dist:
                best_dist = curr_dist
                best_routes = [list(r) for r in routes]
                best_valid = True
        elif random.random() < math.exp(-delta / T):
            # Accept with probability
            curr_dist = new_dist
            curr_valid = new_valid
            routes = new_routes
            if curr_valid and curr_dist < best_dist:
                best_dist = curr_dist
                best_routes = [list(r) for r in routes]
                best_valid = True
        
        T *= alpha
        k += 1

    # Final Local Search on best solution
    # Run 2-opt and Or-opt on best_routes to polish
    # We will do a few iterations of local search on the best solution found.
    
    def local_search(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 2-opt
            for i in range(len(routes)):
                r = routes[i]
                n = len(r)
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        if new_r == r:
                            continue
                        feas, d_new, _ = check_route(new_r)
                        if not feas:
                            continue
                        d_old = get_distance(r)
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            improved = True
            # Or-opt
            for i in range(len(routes)):
                r = routes[i]
                n = len(r)
                for length in [1, 2, 3]:
                    for a in range(n - length + 1):
                        segment = r[a:a+length]
                        rest = r[:a] + r[a+length:]
                        for b in range(len(rest) + 1):
                            new_r = rest[:b] + segment + rest[b:]
                            if new_r == r:
                                continue
                            feas, d_new, _ = check_route(new_r)
                            if not feas:
                                continue
                            d_old = get_distance(r)
                            if d_new < d_old - 1e-9:
                                routes[i] = new_r
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes

    if best_valid:
        # Copy best_routes to a local variable for LS
        ls_routes = [list(r) for r in best_routes]
        ls_routes = local_search(ls_routes)
        # Verify LS result
        ls_dist, ls_valid = evaluate(ls_routes)
        if ls_valid and ls_dist < best_dist:
            best_routes = ls_routes
            best_dist = ls_dist

    return best_routes
# EVOLVE-BLOCK-END
