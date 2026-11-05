"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _precompute_iso():
    """
    Precompute for each point p, the list of pairs (a, b) such that p, a, b form an isosceles triangle.
    Also precompute for each pair (a, b), the list of points p such that p, a, b form an isosceles triangle.
    """
    points = list(itertools.product(range(N), repeat=2))
    point_to_idx = {p: i for i, p in enumerate(points)}
    n_pts = len(points)
    
    # iso_pairs[p] = list of (a, b) indices
    iso_pairs = [[] for _ in range(n_pts)]
    # iso_with_pair[(a, b)] = list of p indices
    # Using a dict of lists or a 2D array of sets? 
    # Number of pairs is ~32k. 
    # Let's use a list of sets for iso_with_pair, indexed by pair index.
    # Or simpler: just store the triples.
    
    # For efficiency in checking maximality, we want:
    # Given S (set of indices), is it maximal?
    # For each p not in S, check if exists (a,b) in S such that (a,b,p) is iso.
    
    # Let's precompute: for each p, list of pairs (a,b) that cover p.
    # pair_to_idx = {(a,b): idx}
    
    pairs = []
    pair_to_idx = {}
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            pair_to_idx[(i,j)] = len(pairs)
            pairs.append((i,j))
            
    # For each p, which pairs cover it?
    # covers[p] = list of pair indices
    covers = [[] for _ in range(n_pts)]
    
    # Iterate over all triples? 256^3 is 16M. Might be too slow in Python if done naively.
    # 16M operations * ~10ns = 0.16s? Maybe.
    # Let's optimize.
    # For each p, and each a, find b such that dist(p,a) == dist(p,b) or dist(p,a) == dist(a,b) or dist(a,b) == dist(p,b).
    # This is still complex.
    
    # Alternative: Iterate pairs (a,b). Find all p such that (a,b,p) is iso.
    # For a fixed pair (a,b), the locus of p such that dist(a,p)==dist(b,p) is the perpendicular bisector.
    # The locus of p such that dist(a,p)==dist(a,b) is a circle around a.
    # The locus of p such that dist(b,p)==dist(a,b) is a circle around b.
    
    # We can just iterate p for each pair? 32k pairs * 256 points = 8M checks.
    # 8M checks is fast in Python if optimized.
    
    for idx, (a, b) in enumerate(pairs):
        ax, ay = points[a]
        bx, by = points[b]
        d_ab_sq = (ax-bx)**2 + (ay-by)**2
        
        for p_idx, (px, py) in enumerate(points):
            if p_idx == a or p_idx == b:
                continue
            d_pa_sq = (px-ax)**2 + (py-ay)**2
            d_pb_sq = (px-bx)**2 + (py-by)**2
            
            if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                covers[p_idx].append(idx)
                
    return points, covers, pair_to_idx

def _is_independent(S, points):
    """Check if S is an independent set (no isosceles triangles)."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            # Check if any c in S forms iso with a,b
            # We can use the precomputed covers?
            # If (a,b) is a pair, we need to know if any c in S is in covers[c] containing pair(a,b)?
            # No, covers[c] contains pairs that cover c.
            # We need: for pair (a,b), is there a c in S such that c is covered by (a,b)?
            # This requires a reverse map: pair -> points it covers.
            
            # Let's just calculate directly for speed since S is small.
            ax, ay = points[a]
            bx, by = points[b]
            d_ab_sq = (ax-bx)**2 + (ay-by)**2
            
            for k in range(j+1, len(S_list)):
                c = S_list[k]
                cx, cy = points[c]
                d_ac_sq = (cx-ax)**2 + (cy-ay)**2
                d_bc_sq = (cx-bx)**2 + (cy-by)**2
                if d_ac_sq == d_bc_sq or d_ac_sq == d_ab_sq or d_bc_sq == d_ab_sq:
                    return False
    return True

def _is_maximal(S, points, covers):
    """Check if S is maximal. S is set of indices."""
    S_set = set(S)
    n_pts = len(points)
    # For each p not in S, check if it is covered by some pair in S
    for p in range(n_pts):
        if p in S_set:
            continue
        # Check if any pair in S covers p
        # covers[p] is a list of pair indices
        # We need to check if any of these pairs are fully contained in S
        # Pair index -> (a, b)
        # We don't have pair_to_points easily accessible in this scope unless passed.
        # Let's pass pair_to_points or just use the global pairs list.
        pass # Placeholder, will implement in construct where data is available

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    points, covers, pair_to_idx = _precompute_iso()
    n_pts = len(points)
    
    # Build reverse map: pair_idx -> list of points it covers
    pair_covers = [[] for _ in range(len(pair_to_idx))]
    for p in range(n_pts):
        for pair_idx in covers[p]:
            pair_covers[pair_idx].append(p)
            
    # Helper: check if S is independent
    def is_independent(S):
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True
        for i in range(n):
            a = S_list[i]
            ax, ay = points[a]
            for j in range(i+1, n):
                b = S_list[j]
                bx, by = points[b]
                d_ab_sq = (ax-bx)**2 + (ay-by)**2
                for k in range(j+1, n):
                    c = S_list[k]
                    cx, cy = points[c]
                    d_ac_sq = (cx-ax)**2 + (cy-ay)**2
                    d_bc_sq = (cx-bx)**2 + (cy-by)**2
                    if d_ac_sq == d_bc_sq or d_ac_sq == d_ab_sq or d_bc_sq == d_ab_sq:
                        return False
        return True

    # Helper: check if S is maximal
    def is_maximal(S):
        S_set = set(S)
        # For each p not in S, check if covered by some pair in S
        for p in range(n_pts):
            if p in S_set:
                continue
            # Check if any pair in S covers p
            # covers[p] contains pair indices.
            # We need to check if (a,b) from pair is in S.
            # To speed up, we can precompute for each p, the list of pairs.
            # Then for each pair, check if a in S and b in S.
            covered = False
            for pair_idx in covers[p]:
                a, b = pair_to_idx_inv[pair_idx] # Need inverse map
                if a in S_set and b in S_set:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Create inverse map: pair_idx -> (a, b)
    pair_to_idx_inv = [None] * len(pair_to_idx)
    for (a,b), idx in pair_to_idx.items():
        pair_to_idx_inv[idx] = (a,b)

    # Refine is_maximal using the inverse map
    def is_maximal_fast(S):
        S_set = set(S)
        for p in range(n_pts):
            if p in S_set:
                continue
            for pair_idx in covers[p]:
                a, b = pair_to_idx_inv[pair_idx]
                if a in S_set and b in S_set:
                    break
            else:
                return False
        return True

    # Greedy construction of a random maximal independent set
    def greedy_random_maximal():
        S = set()
        remaining = list(range(n_pts))
        random.shuffle(remaining)
        for p in remaining:
            # Check if p can be added (independence)
            if not is_independent(S | {p}):
                continue
            # If we add p, does it stay maximal? 
            # We don't know yet. We just build a maximal independent set first.
            S.add(p)
            
        # Now S is maximal independent? 
        # Greedy addition ensures independence. 
        # Does it ensure maximality? 
        # Yes, if we iterate through all points and add if possible, the result is maximal.
        # Because if we didn't add a point, it was because it conflicted.
        # So any point not in S conflicts with some point in S? 
        # Not necessarily. It might conflict with a point that wasn't added.
        # But standard greedy for maximal independent set:
        # Iterate vertices. Add if independent of current set.
        # This produces a maximal independent set.
        
        # However, we need to ensure the "no isosceles" condition is 3-uniform.
        # The check `is_independent(S | {p})` is O(|S|^2).
        # This is correct.
        
        return S

    best_set = None
    best_size = n_pts + 1
    
    # Initial construction
    random.seed(42)
    
    for attempt in range(1000): # Limit attempts based on time
        if time.time() > deadline - 10:
            break
            
        S = greedy_random_maximal()
        if not is_maximal_fast(S):
             # Should not happen if greedy is correct, but just in case
             # Try to repair? Or skip.
             continue
             
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Local Search: Minimize
        improved = True
        while improved:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for v in S_list:
                if time.time() > deadline - 5:
                    break
                S_new = S - {v}
                if is_maximal_fast(S_new):
                    S = S_new
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = set(S)
            if time.time() > deadline - 5:
                break
                
        # Perturbation: Swap
        # Try to swap a point v in S with a point u not in S
        # to escape local minimum
        if time.time() > deadline - 10:
            break
            
        # Try a few swaps
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            if len(S) < 2:
                break
            v = random.choice(list(S))
            # Find a candidate u
            # u must be independent of S - {v}
            # and S - {v} + {u} must be maximal
            candidates = [u for u in range(n_pts) if u not in S]
            random.shuffle(candidates)
            
            found_swap = False
            for u in candidates[:10]: # Check only a few
                S_trial = (S - {v}) | {u}
                if is_independent(S_trial) and is_maximal_fast(S_trial):
                    S = S_trial
                    found_swap = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = set(S)
                    break
                    
            if found_swap:
                # Continue local search after swap
                improved = True
                while improved:
                    improved = False
                    S_list = list(S)
                    random.shuffle(S_list)
                    for v in S_list:
                        S_new = S - {v}
                        if is_maximal_fast(S_new):
                            S = S_new
                            improved = True
                            if len(S) < best_size:
                                best_size = len(S)
                                best_set = set(S)

    if best_set is None:
        # Fallback
        best_set = greedy_random_maximal()
        
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
