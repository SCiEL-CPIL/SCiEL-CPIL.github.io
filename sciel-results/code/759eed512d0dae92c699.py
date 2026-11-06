"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s kill
    
    N = 256
    
    # Precompute perfect squares
    squares = set()
    for i in range(1, 20):
        squares.add(i * i)
    
    # Build conflict graph
    # Edge between a and b if a+b is a perfect square
    # Include diagonal: 2a is a perfect square
    conflicts = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        # Diagonal: 2a is a perfect square
        if 2 * a in squares:
            conflicts[a].add(a)
        for b in range(a + 1, N + 1):
            if a + b in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Initial solution: greedy min-degree
    def greedy_min_degree():
        remaining = set(range(1, N + 1))
        solution = set()
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            solution.add(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v from remaining
            for u in list(remaining):
                if u in conflicts[best_v]:
                    remaining.discard(u)
        return solution
    
    best_solution = greedy_min_degree()
    best_size = len(best_solution)
    
    # Local search to improve initial solution
    def local_search(sol, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sol = set(sol)
            # Try adding vertices
            for v in range(1, N + 1):
                if v not in sol and not (conflicts[v] & sol):
                    sol.add(v)
                    improved = True
            # Try swaps: remove one, add one or two
            for v in list(sol):
                if time.time() >= deadline:
                    break
                sol.remove(v)
                # Try adding one
                for u in range(1, N + 1):
                    if u not in sol and u != v and not (conflicts[u] & sol):
                        sol.add(u)
                        improved = True
                        break
                else:
                    # Try adding two
                    added = False
                    for u in range(1, N + 1):
                        if u not in sol and u != v and not (conflicts[u] & sol):
                            sol.add(u)
                            for w in range(u + 1, N + 1):
                                if w not in sol and w != v and not (conflicts[w] & sol):
                                    sol.add(w)
                                    added = True
                                    break
                            if added:
                                improved = True
                                break
                    if not added:
                        sol.add(v)  # restore
        return sol
    
    best_solution = local_search(best_solution, deadline)
    best_size = len(best_solution)
    
    # Tomita-style branch and bound for maximum independent set
    # Use coloring upper bound
    
    def color_bound(vertices, conflicts):
        """Greedy coloring to get upper bound on independent set size."""
        if not vertices:
            return 0
        colors = {}
        color_classes = []
        uncolored = set(vertices)
        while uncolored:
            color_class = set()
            for v in list(uncolored):
                # v can be in this color class if no conflict with any in color_class
                if not (conflicts[v] & color_class):
                    color_class.add(v)
                    uncolored.discard(v)
            color_classes.append(color_class)
        return len(color_classes)
    
    def branch_and_bound(candidates, current_set, current_size, best_size_ref, deadline):
        """Recursive branch and bound."""
        if time.time() >= deadline:
            return current_size, set(current_set)
        
        if not candidates:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                best_size_ref[1] = set(current_set)
            return current_size, set(current_set)
        
        # Upper bound: current_size + coloring bound on candidates
        ub = current_size + color_bound(candidates, conflicts)
        if ub <= best_size_ref[0]:
            return current_size, set(current_set)
        
        # Pick a vertex to branch on
        # Use degree heuristic: pick vertex with max degree in candidates
        v = max(candidates, key=lambda x: len(conflicts[x] & candidates))
        
        # Branch 1: include v
        new_candidates = candidates - {v} - (conflicts[v] & candidates)
        current_set.add(v)
        size1, best1 = branch_and_bound(new_candidates, current_set, current_size + 1, best_size_ref, deadline)
        current_set.remove(v)
        
        # Branch 2: exclude v
        new_candidates = candidates - {v}
        size2, best2 = branch_and_bound(new_candidates, current_set, current_size, best_size_ref, deadline)
        
        return max(size1, size2), best1 if size1 >= size2 else best2
    
    # Run branch and bound with time limit
    best_size_ref = [best_size, set(best_solution)]
    
    # Use iterative deepening or just run with deadline
    candidates = set(range(1, N + 1))
    
    # To make it more effective, we can use the current solution to prune
    # But first, let's just run B&B on the full graph
    # This might be too slow, so we'll use a simpler approach:
    # Start from the greedy solution and try to improve via B&B on the complement
    
    # Alternative: Use B&B but with a tighter time limit and restart if needed
    # For now, let's just run local search more aggressively
    
    # More aggressive local search with random restarts
    for _ in range(100):
        if time.time() >= deadline:
            break
        
        # Random perturbation
        perturbed = set(best_solution)
        for _ in range(random.randint(1, 5)):
            if perturbed:
                v = random.choice(list(perturbed))
                perturbed.discard(v)
            # Try to add a random valid vertex
            candidates_to_add = [u for u in range(1, N + 1) if u not in perturbed and not (conflicts[u] & perturbed)]
            if candidates_to_add:
                perturbed.add(random.choice(candidates_to_add))
        
        # Local search on perturbed
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Add
            for v in range(1, N + 1):
                if v not in perturbed and not (conflicts[v] & perturbed):
                    perturbed.add(v)
                    improved = True
            # Swap
            for v in list(perturbed):
                if time.time() >= deadline:
                    break
                perturbed.remove(v)
                added = False
                for u in range(1, N + 1):
                    if u not in perturbed and u != v and not (conflicts[u] & perturbed):
                        perturbed.add(u)
                        added = True
                        break
                if not added:
                    perturbed.add(v)
        
        if len(perturbed) > best_size:
            best_size = len(perturbed)
            best_solution = perturbed
    
    return sorted(best_solution)
# EVOLVE-BLOCK-END
