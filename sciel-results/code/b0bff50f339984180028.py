"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # Identify all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    
    if not bad_triples:
        return numbers

    # Branch and Bound for Minimum Hitting Set
    # Minimize removed vertices such that every bad triple is hit.
    
    # Precompute vertex frequencies in ALL bad triples (static)
    # This helps in initial ordering or heuristics
    
    # We will use a recursive B&B.
    # State: removed_count, removed_set, remaining_triples (list of indices), available_set
    
    best_removed = n
    best_solution = set()
    
    # To speed up filtering, we can precompute for each triple its vertices.
    # We already have bad_triples.
    
    # Heuristic Lower Bound: Greedy Packing
    # We compute this only if it's likely to prune.
    
    def get_packing_bound(remaining_triples):
        # Greedy maximal matching in 3-uniform hypergraph
        # Sort by index? No, just greedy.
        used = set()
        count = 0
        for idx in remaining_triples:
            a, b, c = bad_triples[idx]
            if a not in used and b not in used and c not in used:
                used.add(a)
                used.add(b)
                used.add(c)
                count += 1
        return count

    def solve_bnb(removed_count, removed_set, remaining_triples, available_set):
        nonlocal best_removed, best_solution, deadline
        
        if time.time() > deadline:
            return
        
        if not remaining_triples:
            if removed_count < best_removed:
                best_removed = removed_count
                best_solution = set(removed_set)
            return
        
        # Pruning
        if removed_count >= best_removed:
            return
            
        # Lower Bound
        # Simple LB: 1
        # If removed_count + 1 >= best_removed, prune? 
        # No, we need at least 1 more, so if removed_count + 1 >= best_removed, we can't improve.
        if removed_count + 1 >= best_removed:
            return
            
        # Stronger LB: Packing
        # Only compute if we are close to best or list is small?
        # Let's compute it if remaining_triples is not too large.
        lb = 0
        if len(remaining_triples) < 50: # Threshold for expensive LB
            lb = get_packing_bound(remaining_triples)
        else:
            lb = 1 # Weak LB for large sets to save time
        
        if removed_count + lb >= best_removed:
            return
            
        # Select a triple to branch on
        # Heuristic: Pick a triple with a vertex that has high degree in remaining_triples
        # Calculate degree of vertices in the first few triples?
        # Or just pick the first triple.
        # Let's pick the triple with the "rarest" vertex to constrain the search?
        # Actually, picking a triple with high-degree vertices is better for Hitting Set branching.
        
        # Let's just pick the first triple in the list.
        # To improve, we could sort remaining_triples by some metric, but that's expensive.
        
        idx = remaining_triples[0]
        a, b, c = bad_triples[idx]
        
        # Order branches by degree in remaining_triples (higher degree first)
        # Degree of v in remaining_triples = number of triples in remaining_triples containing v
        # This is O(K). We do it for 3 vertices.
        
        def count_degree(v):
            cnt = 0
            for t_idx in remaining_triples:
                ta, tb, tc = bad_triples[t_idx]
                if v in (ta, tb, tc):
                    cnt += 1
            return cnt

        # To save time, we can approximate degree or just use static degree?
        # Static degree is fixed. Dynamic degree is better.
        # Let's use dynamic degree for the 3 candidates.
        
        deg_a = count_degree(a)
        deg_b = count_degree(b)
        deg_c = count_degree(c)
        
        candidates = [(a, deg_a), (b, deg_b), (c, deg_c)]
        # Sort by degree descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for v, _ in candidates:
            if v not in available_set:
                continue
                
            new_removed = removed_set | {v}
            
            # Filter remaining_triples
            # A triple is removed if it contains v
            new_remaining = []
            for t_idx in remaining_triples:
                ta, tb, tc = bad_triples[t_idx]
                if v not in (ta, tb, tc):
                    new_remaining.append(t_idx)
            
            solve_bnb(removed_count + 1, new_removed, new_remaining, available_set - {v})

    # Initial Call
    initial_remaining = list(range(len(bad_triples)))
    initial_available = set(numbers)
    
    solve_bnb(0, set(), initial_remaining, initial_available)
    
    # Fallback
    if not best_solution:
        # Greedy Hitting Set
        remaining = set(initial_remaining)
        removed = set()
        while remaining:
            if time.time() > deadline - 1:
                break
            counts = {v: 0 for v in numbers if v not in removed}
            for idx in remaining:
                a, b, c = bad_triples[idx]
                if a not in removed: counts[a] += 1
                if b not in removed: counts[b] += 1
                if c not in removed: counts[c] += 1
            
            best_v = max(counts, key=counts.get)
            removed.add(best_v)
            
            new_remaining = set()
            for idx in remaining:
                a, b, c = bad_triples[idx]
                if best_v not in (a, b, c):
                    new_remaining.add(idx)
            remaining = new_remaining
        best_solution = removed
        
    result = [x for x in numbers if x not in best_solution]
    return result
# EVOLVE-BLOCK-END
