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

    best_solution = []
    
    # 1. Deterministic Min-Degree Greedy
    sol = greedy_min_degree(candidates)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # 2. Randomized Restart Greedy
    # Estimate time per restart. N=245, steps~100-200, inner loop ~100-200. ~20k ops. Very fast.
    # We can run many restarts.
    max_restarts = 200
    for i in range(max_restarts):
        if time.time() - start_time > deadline - 5.0:
            break
        sol = greedy_randomized(candidates, seed=i)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # 3. Local Improvement: Try to add any remaining candidate
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
        
    # Final verification (optional, for safety)
    # For a in best_solution:
    #     if 2*a in squares: return [] # Should not happen
    # for i in range(len(best_solution)):
    #     for j in range(i, len(best_solution)):
    #         if best_solution[i] + best_solution[j] in squares: return []
            
    return best_solution
