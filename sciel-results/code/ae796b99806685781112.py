"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of prime
    factors of all pairwise sums (distinct indices) is minimized.
    """
    max_val = 10000
    n_needed = 12
    
    # Candidate prime sets to try, in order of size
    prime_sets = [
        (2, 3),
        (2, 3, 5),
        (2, 3, 5, 7),
        (2, 3, 5, 7, 11),
    ]
    
    for primes_tuple in prime_sets:
        primes = list(primes_tuple)
        
        # 1. Generate S-smooth numbers <= max_val
        smooth_numbers = set()
        queue = deque([1])
        while queue:
            val = queue.popleft()
            if val > max_val:
                continue
            smooth_numbers.add(val)
            for p in primes:
                new_val = val * p
                if new_val <= max_val:
                    queue.append(new_val)
        
        smooth_list = sorted(smooth_numbers)
        smooth_set = set(smooth_list)
        
        if len(smooth_list) < n_needed:
            continue
            
        # If the set is too large, construction might be slow.
        # 5000 nodes -> 12.5M edges checks. ~1-2s in Python.
        # 10000 nodes -> 50M edges checks. ~5-10s.
        # We'll proceed but be aware.
        
        n = len(smooth_list)
        neighbors = [set() for _ in range(n)]
        
        # Build graph
        for i in range(n):
            vi = smooth_list[i]
            for j in range(i + 1, n):
                vj = smooth_list[j]
                if (vi + vj) in smooth_set:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Find clique
        def find_clique(candidates, current_clique):
            if len(current_clique) == n_needed:
                return list(current_clique)
            
            if not candidates:
                return None
            
            if len(candidates) < n_needed - len(current_clique):
                return None
            
            # Simple iteration
            for v in list(candidates):
                new_candidates = candidates.intersection(neighbors[v])
                if len(new_candidates) < n_needed - len(current_clique) - 1:
                    continue
                
                current_clique.append(v)
                result = find_clique(new_candidates, current_clique)
                if result is not None:
                    return result
                current_clique.pop()
            
            return None
        
        result_indices = find_clique(set(range(n)), [])
        
        if result_indices is not None:
            result_values = [smooth_list[i] for i in result_indices]
            return result_values
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# EVOLVE-BLOCK-END
