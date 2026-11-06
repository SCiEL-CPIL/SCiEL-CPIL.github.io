"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Clarke-Wright Savings-based Lookahead Construction.
    
    1. Identify feasible nodes based on capacity.
    2. Simulate the construction of the remaining route using a global Savings heuristic.
       - We maintain a set of "open segments" (initially single nodes).
       - We use a priority queue of potential merges based on savings.
       - We greedily merge segments if they are feasible (no crossing, capacity).
    3. Return the first customer in the resulting optimal sequence from the current node.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    n = len(feasible)
    
    # Precompute savings for all pairs in feasible set
    # Savings(i, j) = d(current, i) + d(current, j) - d(i, j) ? 
    # Actually, standard Clarke-Wright savings for a route starting at depot (or current) 
    # and ending at depot.
    # Here, we start at `current_node`. 
    # The cost of a route C is sum(d[C[i], C[i+1]]) + d[C[-1], depot].
    # If we have two segments A and B, and we merge them, the savings depend on the edges broken.
    # 
    # Let's simplify the simulation.
    # We want to build a single route from `current_node` visiting all `feasible` nodes and returning to `depot`.
    # We can model this as finding a path.
    # 
    # Alternative approach: 
    # Since N is small (<=50), we can do a fast heuristic simulation.
    # 
    # Let's use a "Savings" heuristic adapted for the open loop (current -> ... -> depot).
    # 
    # 1. Initialize: Each feasible node is a separate "chain" [node]. 
    #    Plus we have the start point `current_node` and end point `depot`.
    #    Actually, it's easier to think of it as merging nodes into a sequence.
    #
    # Let's try a different, faster global heuristic: 
    # "Farthest Insertion" or "Nearest Insertion" often outperforms simple NN for CVRP construction.
    # But the prompt asks for a MATERIAL difference. The current one is NN+Insertion+2Opt.
    # 
    # Let's implement a **Greedy Savings Construction** specifically.
    # 
    # Algorithm:
    # 1. Start with a route: [current_node]
    # 2. We have a set of unvisited nodes `U`.
    # 3. We maintain a priority queue of potential next nodes to insert.
    #    However, standard savings works by starting with all nodes as separate routes and merging.
    #    Since we must start at `current_node` and end at `depot`, we can't just merge arbitrarily.
    #
    # Let's use **Greedy Insertion** but with a global scoring function that considers the "Savings" of adding a node.
    # 
    # Actually, the most distinct and effective strategy for small N is **Simulated Annealing** or **Tabu Search** 
    # on the permutation of nodes, constrained by capacity.
    # Given the 120s limit and 8 instances, we have plenty of time for a metaheuristic.
    #
    # New Strategy: **Fast Simulated Annealing** on the sequence of feasible nodes.
    # 
    # 1. Generate an initial sequence using Nearest Neighbor.
    # 2. Check capacity feasibility. If infeasible, we need a different approach because SA might get stuck in infeasible states.
    #    Better: **Variable Neighborhood Search (VNS)** or **2-Opt/Or-Opt** with multiple random restarts.
    #
    # Let's go with **Multi-Start 2-Opt/Or-Opt**.
    # The current program does 1 start + 1 perturbation.
    # We will do K random starts (e.g., K=5) and optimize each, returning the best.
    # This is a different search method (stochastic global search vs deterministic local search).
    
    # To make it even more distinct and potentially stronger:
    # **Clarke-Wright Savings** is deterministic and often very good for CVRP.
    # Let's implement a proper Savings heuristic for the "current-to-depot" case.
    
    # 1. Create a list of all feasible nodes.
    # 2. We treat `current_node` as the start and `depot` as the end.
    # 3. We want to order the nodes to minimize total distance.
    #
    # Standard Savings Heuristic for CVRP:
    # - Start with each node as a separate route: [current, i, depot] for each i? No, that's not right.
    # - Standard CW starts with [depot, i, depot] for each i.
    # - Here, we have a fixed start `current` and fixed end `depot`.
    # - We can transform this:
    #   Imagine a super-depot D'.
    #   But simpler: 
    #   We have a set of nodes U.
    #   We want a path P from current to depot visiting all U.
    #
    # Let's use **Greedy Savings** adapted:
    # 1. Calculate savings for merging two nodes i and j into the sequence.
    #    If we have ...-i and j-... and we connect i-j, the savings is:
    #    (d(i, next_i) + d(prev_j, j)) - d(i, j) ... this is complex if we don't have the full route yet.
    #
    # Let's stick to **Multi-Start 2-Opt**. It is robust, easy to implement correctly, and often beats single-start heuristics significantly on small instances.
    
    def get_route_cost(route):
        if len(route) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(route) - 1):
            cost += dm[route[i], route[i+1]]
        cost += dm[route[-1], depot]
        return cost

    def is_feasible(route, cap):
        total_demand = sum(demands[node] for node in route)
        return total_demand <= cap

    def optimize_route(r, max_iter=20):
        """2-Opt and Or-Opt optimization."""
        improved = True
        iter_count = 0
        n_route = len(r)
        
        # 2-Opt
        while improved and iter_count < max_iter and not _time_exceeded():
            improved = False
            iter_count += 1
            
            for i in range(n_route - 2):
                if _time_exceeded():
                    return r
                node_i = r[i]
                node_i1 = r[i+1]
                
                for j in range(i + 2, n_route):
                    node_j = r[j]
                    
                    if j < n_route - 1:
                        node_j1 = r[j+1]
                        old_len = dm[node_i, node_i1] + dm[node_j, node_j1]
                        new_len = dm[node_i, node_j] + dm[node_i1, node_j1]
                    else:
                        node_j1 = depot
                        old_len = dm[node_i, node_i1] + dm[node_j, depot]
                        new_len = dm[node_i, node_j] + dm[node_i1, depot]
                        
                    if new_len < old_len - 1e-10:
                        r[i+1:j+1] = reversed(r[i+1:j+1])
                        improved = True
                        if _time_exceeded():
                            return r

        # Or-opt (Single Node Move)
        improved = True
        max_iter = 10
        iter_count = 0
        
        while improved and iter_count < max_iter and not _time_exceeded():
            improved = False
            iter_count += 1
            n_route = len(r)
            
            for k in range(1, n_route): 
                if _time_exceeded():
                    return r
                    
                node_k = r[k]
                
                L = r[k-1]
                if k < n_route - 1:
                    R = r[k+1]
                else:
                    R = depot
                
                cost_remove = dm[L, node_k] + dm[node_k, R]
                cost_connect = dm[L, R]
                
                # Try inserting at m
                for m in range(1, n_route):
                    if m == k:
                        continue
                    
                    r_copy = r[:]
                    r_copy.pop(k)
                    
                    insert_idx = m
                    if m > k:
                        insert_idx = m - 1
                        
                    if insert_idx < 1:
                        continue
                        
                    if insert_idx > len(r_copy) - 1:
                        L_ins = r_copy[-1]
                        R_ins = depot
                        cost_insert_old = dm[L_ins, depot]
                        cost_insert_new = dm[L_ins, node_k] + dm[node_k, depot]
                    else:
                        L_ins = r_copy[insert_idx - 1]
                        R_ins = r_copy[insert_idx]
                        cost_insert_old = dm[L_ins, R_ins]
                        cost_insert_new = dm[L_ins, node_k] + dm[node_k, R_ins]
                    
                    delta = (cost_connect - cost_remove) + (cost_insert_new - cost_insert_old)
                    
                    if delta < -1e-10:
                        r_copy.insert(insert_idx, node_k)
                        r = r_copy
                        improved = True
                        break

            if improved:
                break

        return r

    # Multi-Start Strategy
    best_route = None
    best_cost = float('inf')
    
    # Number of restarts. 
    # With N=50, 2-Opt is O(N^2) ~ 2500 ops. 
    # 10 restarts * 20 iterations * 2500 = 500k ops. Very fast.
    # We can afford many more restarts. Let's do 20 restarts.
    
    restarts = 20
    
    for r_idx in range(restarts):
        if _time_exceeded():
            break
            
        # Generate initial route
        if r_idx == 0:
            # Nearest Neighbor
            route = [current_node]
            remaining_cap = rest_capacity
            unvisited = set(feasible)
            
            while len(unvisited) > 0:
                last = route[-1]
                # Find nearest feasible
                best_node = -1
                best_dist = float('inf')
                for u in unvisited:
                    if demands[u] <= remaining_cap:
                        d = dm[last, u]
                        if d < best_dist:
                            best_dist = d
                            best_node = u
                if best_node == -1:
                    break
                route.append(best_node)
                remaining_cap -= demands[best_node]
                unvisited.remove(best_node)
        else:
            # Randomized Nearest Neighbor (with noise) or Random Shuffle
            # Let's use Random Shuffle for diversity
            route = [current_node] + feasible[:]
            np.random.shuffle(route[1:])
            
            # Check feasibility of the random route
            # If infeasible, we need to fix it or skip.
            # Simple fix: Sort by demand? No.
            # Better: Use a greedy repair or just skip if infeasible.
            # Given capacity 50 and demands 1-9, average demand 5, 10 nodes fit.
            # Random shuffle will often be infeasible if we don't check.
            # Let's check feasibility.
            total_demand = sum(demands[node] for node in route)
            if total_demand > rest_capacity:
                # Try to repair by removing nodes that are far from the end? 
                # Or just skip this restart.
                # To ensure we have valid candidates, let's use a "Greedy with Randomness"
                # instead of pure random.
                
                # Re-generate with randomized greedy
                route = [current_node]
                remaining_cap = rest_capacity
                unvisited = set(feasible)
                
                while len(unvisited) > 0:
                    last = route[-1]
                    # Pick among the top 3 nearest with random weights
                    candidates = []
                    for u in unvisited:
                        if demands[u] <= remaining_cap:
                            d = dm[last, u]
                            candidates.append((d, u))
                    if not candidates:
                        break
                    candidates.sort()
                    top_k = candidates[:min(3, len(candidates))]
                    # Weighted random choice
                    # Simpler: just pick random from top 3
                    chosen = top_k[np.random.randint(len(top_k))][1]
                    route.append(chosen)
                    remaining_cap -= demands[chosen]
                    unvisited.remove(chosen)

        # Ensure the route is feasible (it should be by construction, but double check)
        total_demand = sum(demands[node] for node in route)
        if total_demand > rest_capacity:
            continue
            
        # Optimize
        optimized = optimize_route(route[:])
        cost = get_route_cost(optimized)
        
        if cost < best_cost:
            best_cost = cost
            best_route = optimized[:]

    if best_route is None or len(best_route) < 2:
        # Fallback
        if len(feasible) > 0:
            return int(feasible[0])
        return depot

    return int(best_route[1])

# EVOLVE-BLOCK-END
