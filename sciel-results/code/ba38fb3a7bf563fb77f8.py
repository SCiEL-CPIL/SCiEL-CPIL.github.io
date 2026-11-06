"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
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
    
    def try_add_fast(s, v):
        return len(conflicts[v] & s) == 0
        
    def local_search_improve(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True
            
            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                s.remove(u)
                
                addable = []
                for v in all_candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflicts[u] & s)
                    deg_v = len(conflicts[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)
                    
        return s

    # Try random restarts with different orderings
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > TIME_LIMIT:
            break
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        current_set = set()
        for v in shuffled:
            if try_add_fast(current_set, v):
                current_set.add(v)
        
        current_set = local_search_improve(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    result = sorted(best_set)
    
    # Final verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
# EVOLVE-BLOCK-END
