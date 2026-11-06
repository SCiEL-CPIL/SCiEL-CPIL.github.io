def construct():
    import math
    import random
    
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    def is_diagonal_safe(a):
        return 2 * a not in squares
    
    # Candidates that are diagonal-safe
    all_candidates = [a for a in range(1, n + 1) if is_diagonal_safe(a)]
    
    # Build conflict graph
    conflicts = {a: set() for a in all_candidates}
    for idx, a in enumerate(all_candidates):
        for b in all_candidates[idx+1:]:
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def greedy_min_degree(candidates):
        remaining = set(candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            independent_set.add(chosen)
            remaining.remove(chosen)
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set
    
    # Try multiple random perturbations
    best_set = greedy_min_degree(all_candidates)
    best_size = len(best_set)
    
    # Local search: try to add vertices back
    def try_add(set_s, v):
        for u in set_s:
            if u != v and (u + v) in squares:
                return False
        return True
    
    # Iterative improvement
    improved = True
    current_set = set(best_set)
    while improved:
        improved = False
        for v in all_candidates:
            if v not in current_set and try_add(current_set, v):
                current_set.add(v)
                improved = True
    
    best_set = current_set
    best_size = len(best_set)
    
    # Try random restarts with different orderings
    random.seed(42)
    for trial in range(50):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        # Greedy by shuffled order: add if no conflict with already added
        current_set = set()
        for v in shuffled:
            if try_add(current_set, v):
                current_set.add(v)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Local search improvement
        improved = True
        while improved:
            improved = False
            for v in all_candidates:
                if v not in best_set and try_add(best_set, v):
                    best_set.add(v)
                    improved = True
    
    result = sorted(best_set)
    
    # Final verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
    
    return result
