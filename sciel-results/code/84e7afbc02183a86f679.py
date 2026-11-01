"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    """
    Use a structural approach: generate norm-4 integer vectors, build a compatibility
    graph, and use spectral ordering + multi-restart greedy + local search to find
    a large independent set (valid kissing configuration).
    
    For vectors with norm^2 = 4, the condition is <u,v> <= 2 for all pairs.
    We also try to add norm^2 = 2 vectors compatible with the selected set.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Generate all norm^2 = 4 integer vectors
    # Type A: one coordinate ±2 (22 vectors)
    # Type B: four coordinates ±1 (C(11,4)*16 = 5280 vectors)
    candidates = []
    
    # Type A: one ±2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type B: four ±1's
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Precompute dot products matrix for norm-4 vectors
    # <u,v> for all pairs
    dots = candidates @ candidates.T  # shape (n, n)
    
    # Compatibility: <u,v> <= 2 for u != v
    # Conflict: <u,v> > 2 (i.e., <u,v> >= 3, but since integer, <u,v> >= 3)
    # Actually <u,v> is integer, so conflict if <u,v> >= 3
    # But wait: for distinct vectors, <u,v> can be at most 4 (if they share all 4 positions with same signs)
    # <u,v> = 4 means same vector (for type B) or... let me check
    # Two type B vectors sharing all 4 positions with same signs: <u,v> = 4, but they're the same vector
    # Two type B vectors sharing all 4 positions: <u,v> = sum of 4 products, each ±1, so <u,v> in {-4,-2,0,2,4}
    # <u,v> = 4 only if same signs on all 4 shared positions, meaning same vector
    # <u,v> = 2 if 3 match and 1 differ, or 2 match and 2 don't... wait
    # If they share k positions, <u,v> = sum of k products. For k=4: -4,-2,0,2,4
    # For k=3: -3,-1,1,3. <u,v>=3 is a conflict!
    # For k=2: -2,0,2. OK.
    # For k=1: -1,1. OK.
    # For k=0: 0. OK.
    # Type A vs Type A: different positions give <u,v>=0. Same position: <u,v>=±4.
    # Type A vs Type B: Type A has ±2 at position i, Type B has ±1 at 4 positions.
    #   If i is in Type B's positions: <u,v> = ±2. OK (<=2).
    #   If i not in Type B's positions: <u,v> = 0. OK.
    # So Type A vs Type B is always compatible!
    # Type A vs Type A: same position same sign: <u,v>=4 (same vector, not distinct)
    #   Same position diff sign: <u,v>=-4. OK.
    #   Different positions: <u,v>=0. OK.
    
    # So conflicts only occur among Type B vectors that share 3 or more positions with high dot product
    # Specifically: <u,v> >= 3 is a conflict. Since <u,v> is integer, <u,v> >= 3.
    # For Type B vectors, <u,v> can be 3 when they share 3 positions with 3 matching signs and 1 differing... 
    # Wait, if they share exactly 3 positions, <u,v> = sum of 3 products = -3,-1,1,3.
    # <u,v>=3 means all 3 match. That's a conflict.
    # If they share 4 positions, <u,v> can be 4 (same vector) or 2 (one sign differs) or 0 or -2 or -4.
    # <u,v>=4 is same vector, <u,v>=2 is OK.
    
    # Build conflict matrix: conflict[i,j] = True if <c_i, c_j> > 2 and i != j
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # Also handle: same vectors should be treated as conflict (but they shouldn't appear twice)
    # Our generation doesn't create duplicates, so this is fine.
    
    # Spectral ordering: use Fiedler vector of the conflict graph Laplacian
    # L = D - A where A is adjacency (conflict) matrix
    A = conflict.astype(np.float64)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    
    # Compute Fiedler vector (eigenvector of second smallest eigenvalue)
    # Use power iteration on shifted matrix for speed
    # Or just use numpy eigvalsh for the small matrix (5302 x 5302 might be slow)
    # Let's use a randomized SVD approach or just random ordering for speed
    
    # Actually, let's try multiple strategies:
    # 1. Random ordering greedy
    # 2. Degree-based ordering (highest degree first - most constrained)
    # 3. Spectral if feasible
    
    best_selected = []
    best_count = 0
    
    # Strategy 1: Multiple random restarts with greedy + local search
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random permutation
        order = rng.permutation(n)
        
        # Greedy selection
        selected = []
        selected_set = set()
        for idx in order:
            # Check if idx is compatible with all selected
            # conflict[idx, selected] should all be False
            if selected:
                sel_arr = np.array(selected)
                if np.any(conflict[idx, sel_arr]):
                    continue
            selected.append(idx)
            selected_set.add(idx)
        
        # Local search: try to swap out a selected vector for an unselected one
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            unselected = [i for i in range(n) if i not in selected_set]
            
            # Try removing each selected and adding a better unselected
            for sel_idx in list(selected):
                # Try to find an unselected vector that is compatible with all selected except sel_idx
                # and has higher "degree of freedom"
                remaining_selected = [s for s in selected if s != sel_idx]
                
                for un_idx in unselected[:100]:  # Limit for speed
                    if un_idx in selected_set:
                        continue
                    # Check compatibility with remaining_selected
                    if remaining_selected:
                        rem_arr = np.array(remaining_selected)
                        if np.any(conflict[un_idx, rem_arr]):
                            continue
                    # Also check un_idx is not sel_idx (it isn't, since unselected)
                    
                    # Perform swap
                    new_selected = [s for s in selected if s != sel_idx] + [un_idx]
                    # This swap is valid by construction
                    
                    # Try to add more vectors after the swap
                    new_selected_set = set(new_selected)
                    for add_idx in unselected:
                        if add_idx in new_selected_set:
                            continue
                        if add_idx == sel_idx:
                            continue
                        if new_selected:
                            ns_arr = np.array(new_selected)
                            if np.any(conflict[add_idx, ns_arr]):
                                continue
                        new_selected.append(add_idx)
                        new_selected_set.add(add_idx)
                    
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        selected_set = set(selected)
                        unselected = [i for i in range(n) if i not in selected_set]
                        improved = True
                        break
                if improved:
                    break
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    # Now try to add norm^2 = 2 vectors
    # Generate all norm^2 = 2 vectors (two ±1's)
    norm2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                norm2_candidates.append(v)
    
    norm2_candidates = np.array(norm2_candidates, dtype=np.int64)
    n2 = len(norm2_candidates)
    
    # For each norm-2 vector, check compatibility with all selected norm-4 vectors
    # Condition: ||u-v||^2 >= 4 where u has norm^2=4, v has norm^2=2
    # ||u-v||^2 = 4 + 2 - 2<u,v> = 6 - 2<u,v> >= 4 => <u,v> <= 1
    # Also need norm-2 vectors to be compatible with each other:
    # ||v1-v2||^2 = 2+2-2<v1,v2> = 4-2<v1,v2> >= 4 => <v1,v2> <= 0
    # But if max norm is 2 (from norm-4 vectors), we need dist >= 2, so dist^2 >= 4.
    # For two norm-2 vectors: dist^2 = 4-2<v1,v2> >= 4 => <v1,v2> <= 0.
    # <v1,v2> for two norm-2 vectors: if they share 0 positions: 0. Share 1: ±1. Share 2: ±2.
    # <v1,v2> <= 0 means: share 0 (OK), share 1 with negative product (OK), share 1 with positive (NOT OK),
    # share 2 with any (NOT OK since ±2 or 0... wait <v1,v2> can be -2,0,2 when sharing 2 positions)
    # <v1,v2> = 0 when sharing 2 positions with opposite signs on one. That's OK.
    
    # Let's be more careful. We need to add norm-2 vectors that are:
    # 1. Compatible with all selected norm-4 vectors: <u, v> <= 1 for each selected u
    # 2. Compatible with each other: <v1, v2> <= 0
    
    # First, filter norm-2 candidates against selected norm-4 vectors
    if best_selected:
        sel_vectors = candidates[best_selected]  # shape (best_count, D)
        # Compute <u, v> for all pairs
        cross_dots = norm2_candidates @ sel_vectors.T  # shape (n2, best_count)
        # A norm-2 candidate is compatible if all cross dots <= 1
        compatible_mask = np.all(cross_dots <= 1, axis=1)
        compatible_norm2 = norm2_candidates[compatible_mask]
    else:
        compatible_norm2 = norm2_candidates
    
    # Now do greedy selection on compatible norm-2 vectors with mutual compatibility
    if len(compatible_norm2) > 0 and time.time() < deadline - 2:
        cn2 = compatible_norm2
        cn2_n = len(cn2)
        cn2_dots = cn2 @ cn2.T
        cn2_conflict = (cn2_dots > 0)  # <v1,v2> > 0 is a conflict
        np.fill_diagonal(cn2_conflict, False)
        
        # Greedy with random restarts
        best_norm2 = []
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            order = rng.permutation(cn2_n)
            sel2 = []
            sel2_set = set()
            for idx in order:
                if sel2:
                    if np.any(cn2_conflict[idx, np.array(sel2)]):
                        continue
                sel2.append(idx)
                sel2_set.add(idx)
            if len(sel2) > len(best_norm2):
                best_norm2 = sel2[:]
        
        if best_norm2:
            # Add these to our result
            norm2_vectors = cn2[best_norm2]
            norm4_vectors = candidates[best_selected]
            result = np.vstack([norm4_vectors, norm2_vectors])
        else:
            result = candidates[best_selected]
    else:
        result = candidates[best_selected]
    
    return result
# EVOLVE-BLOCK-END
