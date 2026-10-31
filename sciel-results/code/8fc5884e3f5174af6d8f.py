import numpy as np
import itertools
import time

D = 11
DEADLINE = 145.0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Generate candidate vectors of various weights
    candidates = []
    
    # Weight 1: 22 vectors, norm^2 = 1
    for i in range(D):
        for si in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            candidates.append(v)
    
    # Weight 2: 220 vectors, norm^2 = 4
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Weight 3: 1320 vectors, norm^2 = 9... wait no, norm^2 = 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates.astype(float)**2, axis=1)
    
    # Greedy: sort by norm^2 (prefer higher norm to maximize distance threshold? No, lower norm is easier to satisfy)
    # Actually, we want to maximize count. Let's use a greedy approach:
    # Sort candidates, add each if compatible with all already selected
    
    # The constraint: for selected set S, for new candidate a:
    # min_{s in S} ||a-s||^2 >= max_{t in S ∪ {a}} ||t||^2
    # This is complex because max_norm changes as we add vectors.
    
    # Simpler: fix a target norm^2. Use all vectors with norm^2 = 4.
    # Then constraint is: min dist^2 >= 4.
    
    # For norm^2 = 4 vectors only: all 220 work.
    # Can we add norm^2 < 4 vectors? If max_norm^2 = 4, need dist^2 >= 4.
    # norm^2=1 vector a, norm^2=4 vector s: dist^2 = 1+4-2a.s = 5-2a.s >= 4 => a.s <= 0.5 => a.s <= 0
    # norm^2=3 vector a, norm^2=4 vector s: dist^2 = 3+4-2a.s = 7-2a.s >= 4 => a.s <= 1.5 => a.s <= 1
    
    # Strategy: use all weight-2 (220), then greedily add weight-3 and weight-1 that are compatible.
    # But we showed no weight-3 is compatible with ALL weight-2.
    
    # Alternative: drop some weight-2 to add more weight-3?
    # Or: use a different base.
    
    # Let's try: greedy selection from all candidates, sorted by norm^2 descending.
    # Higher norm vectors are "more demanding" but also increase the threshold.
    
    # Actually, let's just do a simple greedy: sort by norm^2, try to add each.
    # For efficiency, vectorize the compatibility check.
    
    # Sort by norm^2 descending (add high-norm first)
    order = np.argsort(-norms_sq)
    sorted_cands = candidates[order]
    sorted_norms = norms_sq[order]
    
    selected = []
    selected_norms = []
    
    for i in range(n_cand):
        if elapsed() > DEADLINE:
            break
        
        if len(selected) == 0:
            selected.append(sorted_cands[i])
            selected_norms.append(sorted_norms[i])
            continue
        
        a = sorted_cands[i]
        na = sorted_norms[i]
        sel = np.array(selected, dtype=np.int64)
        sel_norms = np.array(selected_norms, dtype=float)
        
        # max_norm_sq after adding a
        new_max_norm_sq = max(na, np.max(sel_norms))
        
        # Check: for all s in selected, ||a-s||^2 >= new_max_norm_sq
        # ||a-s||^2 = na + ||s||^2 - 2*a.s
        dots = sel @ a  # shape (len(selected),)
        dists_sq = na + sel_norms - 2*dots
        if np.min(dists_sq) >= new_max_norm_sq - 1e-9:
            selected.append(a)
            selected_norms.append(na)
    
    result = np.array(selected, dtype=np.int64)
    
    # Verify
    if len(result) > 1:
        sel_f = result.astype(float)
        norms_sq_final = np.sum(sel_f**2, axis=1)
        max_norm_sq_final = np.max(norms_sq_final)
        
        n = len(result)
        if n <= 2000:
            d2 = np.sum((sel_f[:, np.newaxis, :] - sel_f[np.newaxis, :, :])**2, axis=2)
            np.fill_diagonal(d2, np.inf)
            min_d2 = np.min(d2)
            if min_d2 < max_norm_sq_final - 1e-9:
                # Fallback
                w2 = []
                for i, j in itertools.combinations(range(D), 2):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            w2.append(v)
                return np.array(w2, dtype=np.int64)
    
    return result
