import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Step 1: Filter candidates based on diagonal constraint (2*a not square)
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    
    # Build adjacency list for the conflict graph
    # conflict[a] is a set of b such that a+b is a square
    conflict = {a: set() for a in candidates}
    
    # Efficiently build conflicts
    # For each a, check all b > a
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
                
    def greedy_min_degree(pool):
        """
        Greedy algorithm: repeatedly pick the node with minimum degree in the remaining pool,
        add to solution, remove it and its neighbors from pool.
        """
        pool = set(pool)
        selected = []
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in pool:
                # Count conflicts with other nodes in the pool
                deg = len(conflict[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
                        
            if min_deg_node is None:
                break
                
            selected.append(min_deg_node)
            # Remove node and its neighbors from pool
            pool.discard(min_deg_node)
            pool.difference_update(conflict[min_deg_node])
            
        return selected

    def greedy_randomized(pool, seed=None):
        """
        Randomized greedy: pick random node from nodes with minimum degree.
        """
        if seed is not None:
            random.seed(seed)
        pool = set(pool)
        selected = []
        while pool:
            degrees = {node: len(conflict[node] & pool) for node in pool}
            min_deg = min(degrees.values())
            candidates_for_pick = [node for node, deg in degrees.items() if deg == min_deg]
            node = random.choice(candidates_for_pick)
            
            selected.append(node)
            pool.discard(node)
            pool.difference_update(conflict[node])
            
        return selected

    def greedy_randomized_weighted(pool, seed=None):
        """
        Randomized greedy with weighted selection: prefer low-degree nodes but with some randomness.
        """
        if seed is not None:
            random.seed(seed)
        pool = set(pool)
        selected = []
        while pool:
            degrees = {node: len(conflict[node] & pool) for node in pool}
            min_deg = min(degrees.values())
            # Weight: lower degree = higher weight
            weights = {}
            for node, deg in degrees.items():
                # Exponential decay based on degree difference from min
                weights[node] = math.exp(-0.5 * (deg - min_deg))
            total_w = sum(weights.values())
            r = random.random() * total_w
            cum = 0
            node = None
            for nd in pool:
                cum += weights[nd]
                if cum >= r:
                    node = nd
                    break
            if node is None:
                node = list(pool)[-1]
            
            selected.append(node)
            pool.discard(node)
            pool.difference_update(conflict[node])
            
        return selected

    def local_search_swap(solution, pool, max_iters=200):
        """
        Try to improve solution by swapping elements: remove one, try to add others.
        """
        sol_set = set(solution)
        remaining = [x for x in pool if x not in sol_set]
        improved = list(solution)
        sol_set = set(improved)
        
        for _ in range(max_iters):
            found = False
            for i, elem in enumerate(improved):
                # Try removing elem
                sol_set.remove(elem)
                # Try adding any remaining element that doesn't conflict
                for new_elem in remaining:
                    if new_elem not in sol_set and not (conflict[new_elem] & sol_set):
                        sol_set.add(new_elem)
                        improved[i] = new_elem
                        found = True
                        break
                if not found:
                    sol_set.add(elem)  # restore
                    break
                else:
                    # Remove new_elem from remaining
                    if new_elem in remaining:
                        remaining.remove(new_elem)
                    break
            if not found:
                break
        
        return improved

    best_solution = []
    
    # 1. Deterministic Min-Degree Greedy
    sol = greedy_min_degree(candidates)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # 2. Randomized Restart Greedy (min-degree)
    max_restarts = 500
    for i in range(max_restarts):
        if time.time() - start_time > deadline - 10.0:
            break
        sol = greedy_randomized(candidates, seed=i)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # 3. Randomized Restart Greedy (weighted)
    max_restarts_w = 300
    for i in range(max_restarts_w):
        if time.time() - start_time > deadline - 10.0:
            break
        sol = greedy_randomized_weighted(candidates, seed=10000 + i)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # 4. Local Improvement: Try to add any remaining candidate
    best_set = set(best_solution)
    improved = list(best_solution)
    for a in candidates:
        if a not in best_set:
            # Check if a conflicts with any in best_set
            if not (conflict[a] & best_set):
                improved.append(a)
                best_set.add(a)
                
    if len(improved) > len(best_solution):
        best_solution = improved
        
    # 5. Swap-based local search
    remaining_pool = [x for x in candidates if x not in set(best_solution)]
    if time.time() - start_time < deadline - 5.0:
        swapped = local_search_swap(best_solution, remaining_pool, max_iters=500)
        if len(swapped) > len(best_solution):
            best_solution = swapped
            # Try to add more after swap
            best_set = set(best_solution)
            for a in candidates:
                if a not in best_set:
                    if not (conflict[a] & best_set):
                        best_solution.append(a)
                        best_set.add(a)
                        
    # 6. Multiple rounds of local search from different starting points
    if time.time() - start_time < deadline - 5.0:
        for trial in range(50):
            if time.time() - start_time > deadline - 5.0:
                break
            # Start from a randomized greedy solution
            seed = 50000 + trial
            base_sol = greedy_randomized(candidates, seed=seed)
            # Add any non-conflicting elements
            base_set = set(base_sol)
            for a in candidates:
                if a not in base_set:
                    if not (conflict[a] & base_set):
                        base_sol.append(a)
                        base_set.add(a)
            # Swap local search
            rem = [x for x in candidates if x not in base_set]
            trial_sol = local_search_swap(base_sol, rem, max_iters=200)
            if len(trial_sol) > len(best_solution):
                best_solution = trial_sol
                best_set = set(best_solution)
                for a in candidates:
                    if a not in best_set:
                        if not (conflict[a] & best_set):
                            best_solution.append(a)
                            best_set.add(a)
                            
    return best_solution
