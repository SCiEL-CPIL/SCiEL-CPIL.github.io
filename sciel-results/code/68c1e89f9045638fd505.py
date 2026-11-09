"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import random
import sys

def _generate_smooth_numbers(limit, primes):
    """Generate all numbers <= limit that are composed only of the given primes."""
    smooth_nums = {1}
    queue = [1]
    while queue:
        current = queue.pop(0)
        for p in primes:
            next_val = current * p
            if next_val <= limit:
                if next_val not in smooth_nums:
                    smooth_nums.add(next_val)
                    queue.append(next_val)
            else:
                break
    return smooth_nums

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of
    prime factors of all pairwise sums (distinct indices) is minimized.
    
    Strategy:
    1. Target a solution where all pairwise sums are 7-smooth (primes {2,3,5,7}).
       This limits the distinct prime factors to at most 4.
    2. Generate all 7-smooth numbers <= 10000 as candidates.
    3. Generate all 7-smooth numbers <= 20000 to check sums.
    4. Build a compatibility graph where an edge exists if the sum of two candidates is 7-smooth.
    5. Find a clique of size 12 in this graph using backtracking with forward checking.
    6. If successful, return the clique.
    7. If not, expand the prime set to {2,3,5,7,11} and repeat, or fall back to a heuristic.
    """
    max_val = 10000
    n_needed = 12
    
    # Try with primes {2, 3, 5, 7} first
    target_primes_4 = [2, 3, 5, 7]
    
    # Generate 7-smooth numbers up to 10000 (candidates)
    # and up to 20000 (for sum checking)
    smooth_candidates_4 = _generate_smooth_numbers(max_val, target_primes_4)
    smooth_sums_set_4 = _generate_smooth_numbers(2 * max_val, target_primes_4)
    
    # Filter candidates to positive integers
    candidates_4 = [x for x in smooth_candidates_4 if x > 0]
    candidates_4.sort()
    
    if len(candidates_4) >= n_needed:
        n_cand = len(candidates_4)
        adj = [set() for _ in range(n_cand)]
        
        # Build adjacency
        for i in range(n_cand):
            ci = candidates_4[i]
            for j in range(i + 1, n_cand):
                cj = candidates_4[j]
                s = ci + cj
                if s in smooth_sums_set_4:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Pre-sort by degree
        order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
        
        search_count = 0
        max_search_nodes = 200000

        def find_clique_4(clique, available):
            nonlocal search_count
            search_count += 1
            if search_count > max_search_nodes:
                return None
            if len(clique) == n_needed:
                return clique
            if not available:
                return None
                
            for v in order:
                if v not in available:
                    continue
                if all(v in adj[u] for u in clique):
                    new_clique = clique + [v]
                    new_available = {u for u in available if u in adj[v]}
                    result = find_clique_4(new_clique, new_available)
                    if result is not None:
                        return result
            return None

        result_indices = find_clique_4([], set(range(n_cand)))
        if result_indices is not None:
            return [candidates_4[i] for i in result_indices]

    # If 7-smooth clique not found, try 11-smooth (primes 2,3,5,7,11)
    target_primes_5 = [2, 3, 5, 7, 11]
    smooth_candidates_5 = _generate_smooth_numbers(max_val, target_primes_5)
    smooth_sums_set_5 = _generate_smooth_numbers(2 * max_val, target_primes_5)
    
    candidates_5 = [x for x in smooth_candidates_5 if x > 0]
    candidates_5.sort()
    
    if len(candidates_5) >= n_needed:
        n_cand = len(candidates_5)
        adj = [set() for _ in range(n_cand)]
        
        for i in range(n_cand):
            ci = candidates_5[i]
            for j in range(i + 1, n_cand):
                cj = candidates_5[j]
                s = ci + cj
                if s in smooth_sums_set_5:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
        
        search_count = 0
        max_search_nodes = 200000

        def find_clique_5(clique, available):
            nonlocal search_count
            search_count += 1
            if search_count > max_search_nodes:
                return None
            if len(clique) == n_needed:
                return clique
            if not available:
                return None
                
            for v in order:
                if v not in available:
                    continue
                if all(v in adj[u] for u in clique):
                    new_clique = clique + [v]
                    new_available = {u for u in available if u in adj[v]}
                    result = find_clique_5(new_clique, new_available)
                    if result is not None:
                        return result
            return None

        result_indices = find_clique_5([], set(range(n_cand)))
        if result_indices is not None:
            return [candidates_5[i] for i in result_indices]

    # Fallback: Just return the first 12 7-smooth numbers.
    if len(candidates_4) >= n_needed:
        return candidates_4[:n_needed]
        
    return list(range(1, n_needed + 1))

# EVOLVE-BLOCK-END
