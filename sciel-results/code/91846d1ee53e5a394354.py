"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

D = 11
TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # We want to maximize k such that for all pairs (i,j),
    # ||v_i - v_j|| >= max_norm, where max_norm = max_i ||v_i||
    # Equivalently: ||v_i - v_j||^2 >= max_norm^2
    # i.e., ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j >= max_norm^2
    
    # Strategy: Use a pool of candidate vectors, then greedily select
    # an independent set in the conflict graph where an edge exists
    # if the pair violates the distance constraint.
    
    # To make the constraint simpler, we can try to use vectors of equal norm.
    # If all vectors have norm^2 = N, then the constraint becomes:
    # N + N - 2*v_i·v_j >= N  =>  N >= 2*v_i·v_j  =>  v_i·v_j <= N/2
    
    # Let's build candidate pools with different norms and handle the general case.
    
    # Approach: Generate many candidate vectors, then use a greedy algorithm
    # to pick as many as possible satisfying the constraint.
    
    # Key insight from the task: the seed uses weight-2 vectors (two non-zero entries of ±1).
    # These have norm^2 = 2. The constraint for two such vectors v, w:
    # ||v-w||^2 >= 2  =>  2 + 2 - 2*v·w >= 2  =>  v·w <= 1
    # For weight-2 vectors, v·w can be: 2 (same pair, same signs), 1 (same pair, one sign diff... wait no)
    # Actually v·w for weight-2 vectors: if they share both positions with same signs, dot=2; 
    # if share one position, dot=±1; if share none, dot=0.
    # So constraint v·w <= 1 means: cannot have two vectors that are identical or negatives in both positions.
    # Wait: if v=(1,1,0,...) and w=(1,1,0,...), dot=2 > 1, conflict.
    # If v=(1,1,0,...) and w=(1,-1,0,...), dot=0, OK.
    # If v=(1,1,0,...) and w=(1,0,1,...), dot=1, OK (1<=1).
    # So the seed picks all C(11,2)*4 = 220 weight-2 vectors and they all satisfy the constraint?
    # Let me verify: max norm^2 = 2. For any two distinct weight-2 vectors, dot product <= 1.
    # Two vectors are "same" only if they have the same positions AND same signs.
    # Since we include all sign combinations, (i,j,+1,+1) and (i,j,+1,+1) are the same vector - but we only add each once.
    # (i,j,+1,+1) and (i,j,-1,-1): dot = -1-1 = -2 <= 1, OK.
    # (i,j,+1,+1) and (i,j,+1,-1): dot = 1-1 = 0 <= 1, OK.
    # (i,j,+1,+1) and (j,k,+1,+1): dot = 1 <= 1, OK.
    # So indeed all 220 weight-2 vectors work. That's the seed.
    
    # To do better, we need to add more vectors or use a different set.
    
    # Idea: Use weight-3 vectors (three non-zero entries of ±1), norm^2=3.
    # Constraint: for v with norm^2=3 and w with norm^2=2:
    # ||v-w||^2 >= max(3,2) = 3  =>  3+2-2*v·w >= 3  =>  v·w <= 1
    # For v (weight 3) and w (weight 2): max |dot| = 2 (if w's two positions are in v's three, same signs).
    # So we need v·w <= 1, meaning we can't have w's two positions both in v's support with same signs.
    
    # For two weight-3 vectors v, w: 3+3-2*v·w >= 3 => v·w <= 3
    # Max dot for weight-3 is 3 (identical). So we can't have two identical weight-3 vectors.
    # v·w=3 only if identical. v·w=2 if they share 2 positions with same signs. That's OK.
    
    # Mixed approach: combine weight-2 and weight-3 vectors.
    # Let's try a greedy approach with a large candidate pool.
    
    # Generate candidates:
    # - All weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    # - All weight-3 vectors (norm^2=3): C(11,3)*8 = 165*8 = 1320
    # - All weight-1 vectors (norm^2=1): 11*2 = 22
    # - All weight-4 vectors (norm^2=4): C(11,4)*16 = 330*16 = 5280
    
    # But mixing different norms makes the constraint more complex.
    # Let's focus on same-norm sets first, then try mixing.
    
    # For weight-3 vectors (norm^2=3): constraint is v·w <= 3, which is always true for distinct vectors.
    # Wait, that means ALL weight-3 vectors are mutually compatible!
    # Let me double check: ||v-w||^2 = ||v||^2 + ||w||^2 - 2v·w = 3+3-2v·w = 6-2v·w
    # We need this >= 3 (max norm^2), so 6-2v·w >= 3 => v·w <= 1.5, i.e., v·w <= 1 (integer).
    # Hmm, I made an error. Let me redo.
    # max_norm = max_i ||v_i||. If all have norm^2=3, max_norm^2=3.
    # Constraint: ||v_i - v_j||^2 >= 3 for all i != j.
    # 6 - 2*v_i·v_j >= 3 => v_i·v_j <= 1.5 => v_i·v_j <= 1 (since integer).
    # So for weight-3 vectors, we need dot product <= 1.
    # Two weight-3 vectors can have dot product up to 3 (if identical).
    # Dot=2 if they share exactly 2 positions with same signs.
    # So we CAN'T just take all weight-3 vectors.
    
    # Let me reconsider. The seed takes 220 vectors. Can we do better?
    
    # Let's try a greedy approach: build a candidate pool, then greedily add vectors
    # that don't conflict with already-selected ones.
    
    def generate_candidates():
        candidates = []
        # Weight 2: norm^2 = 2
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(np.array(v, dtype=np.int64))
        
        # Weight 3: norm^2 = 3
        for i, j, k in itertools.combinations(range(D), 3):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0]*D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        candidates.append(np.array(v, dtype=np.int64))
        
        # Weight 1: norm^2 = 1
        for i in range(D):
            for si in (1, -1):
                v = [0]*D
                v[i] = si
                candidates.append(np.array(v, dtype=np.int64))
        
        return np.array(candidates, dtype=np.int64)
    
    def check_constraint(selected, new_vec):
        """Check if new_vec can be added to selected set."""
        if len(selected) == 0:
            return True
        sel_arr = np.array(selected, dtype=np.int64)
        # Compute squared distances
        # ||sel - new||^2 for each sel
        diffs = sel_arr - new_vec
        sq_dists = np.sum(diffs**2, axis=1)
        max_norm_sq = np.max(np.sum(sel_arr**2, axis=1))
        max_norm_sq_new = np.sum(new_vec**2)
        max_norm_sq = max(max_norm_sq, max_norm_sq_new)
        return np.all(sq_dists >= max_norm_sq)
    
    # Greedy selection with multiple orderings
    candidates = generate_candidates()
    n_candidates = len(candidates)
    
    # Precompute norms
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Try greedy with different orderings
    best_set = []
    
    # Ordering 1: by norm (smallest first)
    order1 = np.argsort(norms_sq)
    
    # Ordering 2: random (several tries)
    np.random.seed(42)
    
    def greedy_select(order, candidates, max_time=None):
        selected = []
        selected_arr = []
        for idx in order:
            if max_time is not None and time.time() - start_time > max_time:
                break
            v = candidates[idx]
            if len(selected_arr) == 0:
                selected.append(idx)
                selected_arr.append(v)
                continue
            sel_arr = np.array(selected_arr, dtype=np.int64)
            diffs = sel_arr - v
            sq_dists = np.sum(diffs**2, axis=1)
            max_norm_sq = max(np.max(np.sum(sel_arr**2, axis=1)), np.sum(v**2))
            if np.all(sq_dists >= max_norm_sq):
                selected.append(idx)
                selected_arr.append(v)
        return selected
    
    # Try ordering by norm
    t0 = time.time()
    sel1 = greedy_select(order1, candidates, max_time=TIME_LIMIT - 5)
    if len(sel1) > len(best_set):
        best_set = sel1
    
    # Try several random orderings
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        order = np.random.permutation(n_candidates)
        sel = greedy_select(order, candidates, max_time=TIME_LIMIT - 5)
        if len(sel) > len(best_set):
            best_set = sel
    
    # Try a smarter greedy: prioritize vectors that conflict with fewest others
    if time.time() - start_time < TIME_LIMIT - 15:
        # Build conflict matrix for a subset (too large for all)
        # Let's work with weight-2 and weight-3 only
        w2_mask = norms_sq == 2
        w3_mask = norms_sq == 3
        w1_mask = norms_sq == 1
        
        # Start with all weight-2 (known to work)
        w2_indices = np.where(w2_mask)[0]
        selected = list(w2_indices)
        selected_arr = candidates[selected]
        
        # Now try to add weight-3 vectors
        w3_indices = np.where(w3_mask)[0]
        for idx in w3_indices:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            v = candidates[idx]
            sel_arr = np.array(selected_arr, dtype=np.int64)
            diffs = sel_arr - v
            sq_dists = np.sum(diffs**2, axis=1)
            max_norm_sq = max(3, np.max(np.sum(sel_arr**2, axis=1)))
            if np.all(sq_dists >= max_norm_sq):
                selected.append(idx)
                selected_arr = np.vstack([selected_arr, v.reshape(1, -1)])
        
        if len(selected) > len(best_set):
            best_set = selected
    
    # Try adding weight-1 vectors to the best set
    if time.time() - start_time < TIME_LIMIT - 5 and len(best_set) > 0:
        best_arr = candidates[best_set]
        w1_indices = np.where(w1_mask)[0]
        new_best = list(best_set)
        for idx in w1_indices:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            v = candidates[idx]
            sel_arr = np.array(candidates[new_best], dtype=np.int64)
            diffs = sel_arr - v
            sq_dists = np.sum(diffs**2, axis=1)
            max_norm_sq = max(np.sum(v**2), np.max(np.sum(sel_arr**2, axis=1)))
            if np.all(sq_dists >= max_norm_sq):
                new_best.append(idx)
        if len(new_best) > len(best_set):
            best_set = new_best
    
    if len(best_set) == 0:
        # Fallback: return seed result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set]

# EVOLVE-BLOCK-END
