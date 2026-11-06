import math
import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = 85.0

    n = 256
    max_sum = 2 * n  # 512

    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    # Remove 1 (can't be sum of two positive integers >= 1)
    squares.discard(1)

    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)

    # Build conflict graph
    conflict = {a: set() for a in candidates}
    
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Helper: check if a set is valid
    def is_valid(s):
        s = set(s)
        for a in s:
            if 2*a in squares:
                return False
        for i, a in enumerate(s):
            for b in list(s)[i+1:]:
                if a + b in squares:
                    return False
        return True

    # Greedy min-degree
    def greedy_min_degree(pool):
        pool = set(pool)
        selected = []
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in sorted(pool):  # sorted for determinism
                deg = len(conflict[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
            if min_deg_node is None:
                break
            selected.append(min_deg_node)
            pool.discard(min_deg_node)
            pool.difference_update(conflict[min_deg_node])
        return selected

    # Greedy with priority queue for efficiency
    def greedy_priority(pool):
        pool = set(pool)
        selected = []
        while pool:
            best_node = None
            best_deg = float('inf')
            for node in pool:
                deg = len(conflict[node] & pool)
                if deg < best_deg or (deg == best_deg and (best_node is None or node < best_node)):
                    best_deg = deg
                    best_node = node
            if best_node is None:
                break
            selected.append(best_node)
            pool.discard(best_node)
            pool.difference_update(conflict[best_node])
        return selected

    best_solution = greedy_min_degree(candidates)

    # Local search: try to improve by adding elements
    def try_add(solution):
        sol_set = set(solution)
        improved = list(solution)
        changed = True
        while changed:
            changed = False
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    improved.append(a)
                    sol_set.add(a)
                    changed = True
        return improved

    best_solution = try_add(best_solution)

    # Randomized restarts with different tie-breaking
    for seed in range(500):
        if time.time() - start_time > deadline - 5.0:
            break
        random.seed(seed)
        pool = set(candidates)
        selected = []
        while pool:
            # Compute degrees
            degrees = {}
            for node in pool:
                degrees[node] = len(conflict[node] & pool)
            min_deg = min(degrees.values())
            # Pick random from min-degree nodes
            min_nodes = [node for node, deg in degrees.items() if deg == min_deg]
            node = random.choice(min_nodes)
            selected.append(node)
            pool.discard(node)
            pool.difference_update(conflict[node])
        
        selected = try_add(selected)
        if len(selected) > len(best_solution):
            best_solution = selected

    # Try swap-based local search
    def local_search_swap(solution, max_iterations=1000):
        sol_set = set(solution)
        best = list(solution)
        for _ in range(max_iterations):
            if time.time() - start_time > deadline - 3.0:
                break
            improved = False
            # Try to add one element
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    new_sol = list(sol_set) + [a]
                    if len(new_sol) > len(best):
                        best = new_sol
                        sol_set = set(best)
                        improved = True
                        break
            if improved:
                continue
            # Try swap: remove one, add one (net gain 0, but might enable future additions)
            # This is complex; skip for now
            break
        return best

    best_solution = local_search_swap(best_solution)

    # Final verification
    best_set = set(best_solution)
    for a in best_solution:
        if 2*a in squares:
            return []
    for i, a in enumerate(best_solution):
        for b in best_solution[i+1:]:
            if a + b in squares:
                return []
    
    return best_solution
