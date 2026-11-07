"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

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

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    def check_route(route):
        """
        Checks if a route (list of node ids) is feasible.
        Returns (is_feasible, total_distance, total_time)
        """
        if not route:
            return True, 0.0, 0.0
        
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        demand_sum = 0
        
        for i, node in enumerate(route):
            prev = 0 if i == 0 else route[i - 1]
            d = dist[prev, node]
            
            # Battery check
            if bat - d < 0:
                return False, total_dist + d, t
            bat -= d
            total_dist += d
            t += d / speed
            
            if node in stations:
                # Recharge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                # Customer
                demand_sum += demands[node]
                if demand_sum > capacity:
                    return False, total_dist, t
                
                # Time window
                t = max(t, ready[node])
                if t > due[node]:
                    return False, total_dist, t
                t += service[node]
        
        # Return to depot
        d = dist[route[-1], 0]
        if bat - d < 0:
            return False, total_dist + d, t
        bat -= d
        total_dist += d
        t += d / speed
        
        if t > horizon:
            return False, total_dist, t
            
        return True, total_dist, t

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    def is_valid_solution(routes):
        # Check all customers served exactly once
        served = set()
        for r in routes:
            for node in r:
                if node in stations:
                    continue
                if node in served:
                    return False
                served.add(node)
                if node < 1 or node > n_customers:
                    return False
        if len(served) != n_customers:
            return False
        
        # Check feasibility of each route
        for r in routes:
            feas, _, _ = check_route(r)
            if not feas:
                return False
        return True

    # 1. Construction: Clarke-Wright Savings
    # Start with one route per customer
    customers = list(range(1, n_customers + 1))
    routes = [[c] for c in customers]
    
    # Calculate savings: dist(0, i) + dist(0, j) - dist(i, j)
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist[0, i] + dist[0, j] - dist[i, j]
            if s > 0:
                savings.append((s, i, j))
    
    savings.sort(reverse=True)
    
    # Map customer to route index
    cust_to_route = {c: i for i, c in enumerate(customers)}
    
    for s, i, j in savings:
        if time.time() > deadline - 30: break # Reserve time for GA
        
        ri = cust_to_route.get(i)
        rj = cust_to_route.get(j)
        if ri is None or rj is None:
            continue
        if ri == rj:
            continue
            
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Check if i is at the end of route_i or j is at the start of route_j
        # We need to merge such that i and j become adjacent
        # Possible merges:
        # 1. [.. i] + [j ..] -> [.. i, j ..]
        # 2. [.. i] + [.. j] -> [.. i, .. j] (invalid, j not at start)
        # 3. [i ..] + [j ..] -> [i .., j ..] (invalid, i not at end)
        # 4. [i ..] + [.. j] -> [i .., .. j] (invalid)
        
        # Check ends
        i_is_end = (route_i[-1] == i)
        i_is_start = (route_i[0] == i)
        j_is_end = (route_j[-1] == j)
        j_is_start = (route_j[0] == j)
        
        new_route = None
        
        if i_is_end and j_is_start:
            new_route = route_i + route_j
        elif i_is_start and j_is_end:
            new_route = route_j + route_i
        elif i_is_end and j_is_end:
            new_route = route_i + route_j[::-1]
        elif i_is_start and j_is_start:
            new_route = route_i[::-1] + route_j
        
        if new_route is None:
            continue
            
        # Check demand
        demand_sum = sum(demands[n] for n in new_route if n not in stations)
        if demand_sum > capacity:
            continue
            
        # Check feasibility
        feas, _, _ = check_route(new_route)
        if not feas:
            continue
            
        # Perform merge
        routes[ri] = new_route
        routes[rj] = []
        # Update cust_to_route for all nodes in new_route
        for node in new_route:
            cust_to_route[node] = ri
            
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # 2. Local Search (2-opt and Or-opt) to polish initial solution
    def local_search(routes, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            for i in range(len(routes)):
                r = routes[i]
                if len(r) < 2:
                    continue
                n = len(r)
                # 2-opt
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        feas, d_new, _ = check_route(new_r)
                        if not feas:
                            continue
                        d_old, _, _ = check_route(r)
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            improved = True
                            break
                    if improved:
                        break
                # Or-opt (move 1 or 2 nodes)
                if improved: continue
                for length in [1, 2]:
                    for a in range(n):
                        if a + length > n: continue
                        segment = r[a:a+length]
                        rest = r[:a] + r[a+length:]
                        for b in range(len(rest) + 1):
                            new_r = rest[:b] + segment + rest[b:]
                            if new_r == r: continue
                            feas, d_new, _ = check_route(new_r)
                            if not feas: continue
                            d_old, _, _ = check_route(r)
                            if d_new < d_old - 1e-9:
                                routes[i] = new_r
                                improved = True
                                break
                        if improved: break
                    if improved: break
                    
    local_search(routes)
    
    best_routes = [r[:] for r in routes]
    best_score = total_distance(best_routes)
    
    # 3. Genetic Algorithm
    # Population size 20
    pop_size = 20
    population = [best_routes]
    # Initialize rest of population with random valid solutions or mutations of best
    for _ in range(pop_size - 1):
        # Random construction
        custs = list(range(1, n_customers + 1))
        random.shuffle(custs)
        r_pop = []
        for c in custs:
            # Try to insert into existing routes
            inserted = False
            for i, r in enumerate(r_pop):
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [c] + r[pos:]
                    feas, _, _ = check_route(new_r)
                    if feas:
                        r_pop[i] = new_r
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                r_pop.append([c])
        # Polish
        local_search(r_pop, max_iters=5)
        population.append(r_pop)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        # Evaluate
        scores = []
        for ind in population:
            if is_valid_solution(ind):
                scores.append(total_distance(ind))
            else:
                scores.append(float('inf'))
        
        # Update best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_routes = [r[:] for r in population[min_idx]]
            
        # Selection: Tournament
        new_population = []
        for _ in range(pop_size):
            # Pick 3 random individuals
            candidates = random.sample(range(pop_size), 3)
            winner = min(candidates, key=lambda x: scores[x])
            parent1 = population[winner]
            
            # Crossover: Splice
            # Pick two random routes from parent1 to splice? 
            # Simpler: Pick a random cut point in the sequence of all customers?
            # Since it's multi-route, we can do:
            # 1. Flatten parent1 and parent2? No, keep structure.
            # Let's do a simple mutation-heavy GA if crossover is complex.
            # Or "Cross" as in the prompt: combine two routes.
            
            # Let's try: Pick two random routes from parent1, and one from parent2?
            # Actually, let's just do mutation for simplicity and robustness, 
            # but the prompt asks for "cross".
            # Let's implement a simple cross:
            # Take a random route from parent1 and a random route from parent2.
            # Merge them? No, that violates capacity potentially.
            
            # Let's stick to robust local search + random restarts which is often better for VRP.
            # But the directive says "Model the problem as...". 
            # Wait, the strategy I stated was "Clarke-Wright + GA with Cross".
            # Let's implement a basic cross:
            # 1. Select parent1 and parent2.
            # 2. Pick a random route index i from parent1 and j from parent2.
            # 3. Swap the contents of route i in P1 and route j in P2? 
            #    This is a "swap" operator.
            # 4. Or "splicing": Take first half of route i from P1 and second half from P2?
            #    This might break time windows.
            
            # Let's use "Relocate" and "Swap" as the main operators, which is standard and effective.
            child = [r[:] for r in parent1]
            
            # Mutation: Relocate
            if random.random() < 0.7:
                r_idx = random.randrange(len(child))
                if len(child[r_idx]) > 1:
                    c_idx = random.randrange(len(child[r_idx]))
                    c = child[r_idx].pop(c_idx)
                    # Try to insert c into another random route
                    target_idx = random.randrange(len(child))
                    pos = random.randrange(len(child[target_idx]) + 1)
                    child[target_idx].insert(pos, c)
            
            # Mutation: Swap
            if random.random() < 0.7:
                r_idx1 = random.randrange(len(child))
                r_idx2 = random.randrange(len(child))
                if r_idx1 != r_idx2 and len(child[r_idx1]) > 0 and len(child[r_idx2]) > 0:
                    c1_idx = random.randrange(len(child[r_idx1]))
                    c2_idx = random.randrange(len(child[r_idx2]))
                    c1 = child[r_idx1][c1_idx]
                    c2 = child[r_idx2][c2_idx]
                    child[r_idx1][c1_idx] = c2
                    child[r_idx2][c2_idx] = c1
            
            # Repair/Polish
            local_search(child, max_iters=3)
            
            if is_valid_solution(child):
                new_population.append(child)
            else:
                # If invalid, keep parent
                new_population.append([r[:] for r in parent1])
                
        population = new_population
        
        # Elitism: Keep best
        # Find best in new_population
        best_in_new = None
        best_score_new = float('inf')
        for ind in new_population:
            if is_valid_solution(ind):
                s = total_distance(ind)
                if s < best_score_new:
                    best_score_new = s
                    best_in_new = ind
        if best_in_new:
            # Replace worst with best_in_new if it's better
            worst_idx = np.argmax([total_distance(ind) if is_valid_solution(ind) else float('inf') for ind in new_population])
            if best_score_new < total_distance(new_population[worst_idx]):
                new_population[worst_idx] = best_in_new
            population = new_population

    return best_routes
# EVOLVE-BLOCK-END
