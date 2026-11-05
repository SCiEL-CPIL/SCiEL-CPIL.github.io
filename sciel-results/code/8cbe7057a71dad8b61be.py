"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _has_triangle_circulant(n, S):
    """Check if circulant graph C_n(S) has a triangle."""
    # Triangle exists if there are distinct a, b in S such that a+b in S
    # and a, b, a+b are all distinct modulo n
    S_set = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            s = (a + b) % n
            if s in S_set and s != a and s != b and s != 0:
                return True
    return False

def _alpha_circulant(n, S):
    """Estimate or calculate independence number of C_n(S)."""
    # adj_mask[i] = bitmask of neighbors
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    # Max Independent Set using Branch and Bound
    # candidates: bitmask of vertices available
    # current_size: size of current independent set
    best = 0
    
    def dfs(candidates, current_size):
        nonlocal best
        # Upper bound: current_size + remaining candidates
        if current_size + bin(candidates).count('1') <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        
        # Pick vertex with max degree in the candidate subgraph
        # This helps prune the search tree more effectively
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        max_deg = -1
        best_v_bit = v_bit
        temp = candidates
        while temp:
            t_bit = temp & -temp
            t = t_bit.bit_length() - 1
            deg = bin(adj_mask[t] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v_bit = t_bit
            temp ^= t_bit
        
        v_bit = best_v_bit
        v = v_bit.bit_length() - 1
        
        # Option 1: Include v
        # New candidates are those not adjacent to v
        new_cand = candidates & ~adj_mask[v] & ~v_bit
        dfs(new_cand, current_size + 1)
        
        # Option 2: Exclude v
        dfs(candidates & ~v_bit, current_size)

    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    return best

def _verify_circulant(n, S):
    if _has_triangle_circulant(n, S):
        return False
    alpha = _alpha_circulant(n, S)
    if alpha >= 10:
        return False
    return True

def _circulant_to_edges(n, S):
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid circulant graph for n from 40 down to 20
    # We prioritize larger n
    
    for n in range(40, 20, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random sum-free sets S
        # S should be symmetric? No, if S is not symmetric, we add -S to make it undirected?
        # Standard circulant definition: edges (i, i+s) for s in S.
        # For undirected, if s in S, -s should be in S? Or we just iterate all s.
        # If we define edges for all s in S, and S contains both s and -s, it's fine.
        # To ensure triangle-free, we just need no a,b in S such that a+b in S.
        
        found = False
        
        # First, try systematic search with structured sets
        # Then fall back to random search
        
        # Strategy 1: Try sets based on quadratic residues or other algebraic structures
        # Strategy 2: Random search with incremental growth
        
        # For n=39, try known constructions first
        if n == 39:
            # Try the Paley graph construction or similar
            # For n=39, try S = {1, 2, 4, 8, 16} (powers of 2)
            test_sets = [
                [1, 2, 4, 8, 16],
                [1, 2, 4, 8, 17],
                [1, 2, 4, 8, 16, 32],
                [1, 3, 9, 27],
                [1, 2, 3, 5, 8],
                [1, 2, 4, 7, 14, 28],
            ]
            for S_test in test_sets:
                S_test = [s % n for s in S_test if s % n != 0]
                S_test = list(set(S_test))
                if _has_triangle_circulant(n, S_test):
                    continue
                alpha = _alpha_circulant(n, S_test)
                if alpha < 10:
                    best_n = n
                    best_edges = _circulant_to_edges(n, S_test)
                    found = True
                    break
            if found:
                break
        
        # Random search
        max_attempts = 2000 if n < 35 else 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate a random subset of {1, ..., n-1}
            # Strategy: Randomly select a set S
            # Ensure symmetry: if s in S, add (n-s) % n
            # Check sum-free property during generation
            
            S = set()
            # Try to build S incrementally
            # Randomly shuffle 1..n-1
            candidates = list(range(1, n))
            random.shuffle(candidates)
            
            for c in candidates:
                # Check if adding c (and -c) breaks sum-free property
                new_S = set(S)
                new_S.add(c)
                new_S.add((n - c) % n)
                
                is_sum_free = True
                # Check all pairs a, b in new_S
                # If a+b in new_S and a+b not in {0, a, b}, it's a triangle
                for a in new_S:
                    for b in new_S:
                        if a == b:
                            continue
                        s = (a + b) % n
                        if s in new_S and s != a and s != b and s != 0:
                            is_sum_free = False
                            break
                    if not is_sum_free:
                        break
                
                if is_sum_free:
                    S = new_S
                    
            # If S is empty, skip
            if not S:
                continue
                
            # Verify triangle-free (redundant if we built it sum-free, but safe)
            if _has_triangle_circulant(n, list(S)):
                continue
                
            # Check Independence Number
            alpha = _alpha_circulant(n, list(S))
            if alpha < 10:
                # Found a valid graph!
                best_n = n
                best_edges = _circulant_to_edges(n, list(S))
                found = True
                break
        
        if found:
            # We found a valid graph for this n.
            # Can we try to improve alpha or just accept?
            # Since we go from high n to low n, the first found is the best n.
            # We can try to find a better S for the same n?
            # Not necessary if n is maxed out.
            # But maybe we can find n=40?
            # If we found n=39, we can stop.
            # The loop breaks on n descent.
            break
    
    # Fallback: If no circulant graph found, use the known Andrásfai graph (n=26)
    if best_n == 0:
        n0 = 26
        edges0 = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        best_edges = edges0
    else:
        # Final verification of the found circulant graph
        # Just in case
        pass # Already verified in loop

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
