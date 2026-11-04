"""Kissing number in dimension 11: construct integer vectors with min pairwise distance >= max norm."""
import numpy as np
import time
import random

D = 11
DEADLINE = 135  # safety margin before 150s kill


def _norm2(v):
    return int(np.dot(v, v))


def _dist2(u, v):
    d = u - v
    return int(np.dot(d, d))


def _min_pairwise_dist2(vectors):
    """Return minimum squared pairwise distance."""
    n = len(vectors)
    if n < 2:
        return float('inf')
    # Use numpy for speed
    mat = np.array(vectors, dtype=np.int64)
    # squared distances
    sq = np.sum(mat ** 2, axis=1)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    gram = mat @ mat.T
    dists2 = sq[:, None] + sq[None, :] - 2 * gram
    np.fill_diagonal(dists2, float('inf'))
    return int(dists2.min())


def _max_norm2(vectors):
    mat = np.array(vectors, dtype=np.int64)
    return int(np.max(np.sum(mat ** 2, axis=1)))


def _is_valid(vectors):
    if len(vectors) == 0:
        return True
    mnd = _min_pairwise_dist2(vectors)
    mnn = _max_norm2(vectors)
    return mnd >= mnn


def _greedy_add(candidates, current_vectors, max_norm2):
    """Try to add candidates one by one, keeping only those that satisfy constraint."""
    result = list(current_vectors)
    for c in candidates:
        # Check against all existing vectors
        ok = True
        for v in result:
            if _dist2(np.array(c, dtype=np.int64), np.array(v, dtype=np.int64)) < max_norm2:
                ok = False
                break
        if ok:
            result.append(c)
            # Update max_norm2 if needed
            cn = _norm2(np.array(c, dtype=np.int64))
            if cn > max_norm2:
                # New max norm means we need to re-check all pairs
                # For simplicity, recompute
                max_norm2 = max(max_norm2, cn)
                # Re-validate: actually this is expensive. Let's just track.
    return result, max_norm2


def _generate_type_d_vectors():
    """Generate vectors of type: e_i + e_j (with signs) and e_i - e_j."""
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vecs.append(tuple(v))
    return vecs


def _generate_type_e_vectors():
    """Generate vectors with three nonzero entries: patterns like (1,1,1), (1,1,-1), etc. on 3 coords."""
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            vecs.append(tuple(v))
    return vecs


def _generate_type_f_vectors():
    """Vectors with 2 nonzero entries of value 2 or -2, or one entry of value 2."""
    vecs = []
    # Single coordinate with value 2 or -2
    for i in range(D):
        for val in (2, -2):
            v = [0] * D
            v[i] = val
            vecs.append(tuple(v))
    # Two coordinates with values from {-2, 2}
    for i in range(D):
        for j in range(i + 1, D):
            for vi in (2, -2):
                for vj in (2, -2):
                    v = [0] * D
                    v[i] = vi
                    v[j] = vj
                    vecs.append(tuple(v))
    return vecs


def _generate_type_g_vectors():
    """Vectors with pattern like (2, 1, 0, ...) or (2, -1, 0, ...) on 2-3 coords."""
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for vi in (2, -2):
                for vj in (1, -1):
                    v = [0] * D
                    v[i] = vi
                    v[j] = vj
                    vecs.append(tuple(v))
                    # Also add the symmetric
                    v2 = [0] * D
                    v2[i] = vj
                    v2[j] = vi
                    vecs.append(tuple(v2))
    return vecs


def _generate_type_h_vectors():
    """Vectors with 4 nonzero entries of value 1/-1."""
    vecs = []
    # This could be large: C(11,4) * 2^4 = 330 * 16 = 5280
    from itertools import combinations
    for coords in combinations(range(D), 4):
        for signs in [(s1, s2, s3, s4) for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1) for s4 in (1, -1)]:
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vecs.append(tuple(v))
    return vecs


def _simulated_annealing_select(candidates, n_iter=50000, t0=1.0, t_end=0.01):
    """
    Use simulated annealing to select a subset of candidate vectors
    maximizing count subject to min pairwise dist >= max norm.
    """
    n = len(candidates)
    if n == 0:
        return []
    
    cand_arr = np.array(candidates, dtype=np.int64)
    cand_norms2 = np.sum(cand_arr ** 2, axis=1)
    
    # Precompute pairwise squared distances
    gram = cand_arr @ cand_arr.T
    dists2 = cand_norms2[:, None] + cand_norms2[None, :] - 2 * gram
    np.fill_diagonal(dists2, 0)
    
    # State: binary selection vector
    # Start with random subset
    selected = np.zeros(n, dtype=bool)
    selected[np.random.choice(n, size=min(n, 20), replace=False)] = True
    
    best_selected = selected.copy()
    best_score = -1
    
    def _score(sel):
        if not sel.any():
            return 0
        idx = np.where(sel)[0]
        sub_norms = cand_norms2[idx]
        sub_dists = dists2[np.ix_(idx, idx)]
        max_n = sub_norms.max()
        # min dist among selected (off-diagonal)
        if len(idx) < 2:
            return len(idx)
        d = sub_dists.copy()
        np.fill_diagonal(d, float('inf'))
        min_d = d.min()
        if min_d < max_n:
            return 0  # invalid
        return len(idx)
    
    current_score = _score(selected)
    if current_score > best_score:
        best_score = current_score
        best_selected = selected.copy()
    
    for it in range(n_iter):
        t = t0 * (t_end / t0) ** (it / max(n_iter - 1, 1))
        
        # Random move: add or remove one vector
        i = np.random.randint(n)
        new_sel = selected.copy()
        if selected[i]:
            new_sel[i] = False
        else:
            new_sel[i] = True
        
        new_score = _score(new_sel)
        delta = new_score - current_score
        
        if delta >= 0 or np.random.random() < np.exp(delta / max(t, 1e-10)):
            selected = new_sel
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_selected = selected.copy()
    
    # Greedy refinement
    sel = best_selected.copy()
    changed = True
    while changed:
        changed = False
        # Try adding unselected
        for i in range(n):
            if not sel[i]:
                trial = sel.copy()
                trial[i] = True
                if _score(trial) > _score(sel):
                    sel = trial
                    changed = True
                    break
        # Try removing selected
        for i in range(n):
            if sel[i]:
                trial = sel.copy()
                trial[i] = False
                if _score(trial) > _score(sel):
                    sel = trial
                    changed = True
                    break
    
    if _score(sel) > _score(best_selected):
        best_selected = sel
    
    result = []
    for i in range(n):
        if best_selected[i]:
            result.append(candidates[i])
    return result


def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Generate candidate vectors of various types
    all_candidates = []
    all_candidates.extend(_generate_type_d_vectors())
    all_candidates.extend(_generate_type_e_vectors())
    all_candidates.extend(_generate_type_f_vectors())
    all_candidates.extend(_generate_type_g_vectors())
    all_candidates.extend(_generate_type_h_vectors())
    
    # Deduplicate
    all_candidates = list(set(all_candidates))
    
    # Filter: all must have norm >= 1 (nonzero)
    all_candidates = [c for c in all_candidates if sum(x * x for x in c) >= 1]
    
    # First pass: greedy selection to get a good initial solution
    random.shuffle(all_candidates)
    current = []
    current_norms2 = []
    max_norm2 = 0
    
    for c in all_candidates:
        cn2 = sum(x * x for x in c)
        # Check if adding c violates constraint
        ok = True
        for i, v in enumerate(current):
            d2 = sum((c[k] - v[k]) ** 2 for k in range(D))
            new_max = max(max_norm2, cn2, current_norms2[i])
            if d2 < new_max:
                ok = False
                break
        if ok:
            current.append(c)
            current_norms2.append(cn2)
            max_norm2 = max(max_norm2, cn2)
    
    best_count = len(current)
    best_vectors = list(current)
    
    # Check time
    if time.time() > deadline - 30:
        return np.array(best_vectors, dtype=np.int64)
    
    # Simulated annealing on the full candidate set for improvement
    # To keep it tractable, work with a subset of candidates
    # Sort candidates by norm and take top ones
    cand_arr = np.array(all_candidates, dtype=np.int64)
    norms2 = np.sum(cand_arr ** 2, axis=1)
    
    # Take a manageable subset: up to 2000 candidates
    max_cands = 2000
    if len(all_candidates) > max_cands:
        # Prefer lower norm vectors (they are easier to pack)
        order = np.argsort(norms2)
        idx = order[:max_cands]
        subset_cands = [all_candidates[i] for i in idx]
    else:
        subset_cands = all_candidates
    
    # Run SA with multiple restarts
    n_restarts = 5
    n_iter_per_restart = 20000
    best_sa_count = best_count
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        sa_result = _simulated_annealing_select(subset_cands, n_iter=n_iter_per_restart)
        if len(sa_result) > best_sa_count:
            best_sa_count = len(sa_result)
            best_vectors = list(sa_result)
    
    # Final greedy refinement on best_vectors
    # Try to add more from all_candidates
    best_arr = np.array(best_vectors, dtype=np.int64) if best_vectors else np.array([], dtype=np.int64).reshape(0, D)
    if len(best_vectors) > 0:
        bnorms2 = np.sum(best_arr ** 2, axis=1)
        bmax_norm2 = int(bnorms2.max())
        # Try adding remaining candidates
        existing_set = set(best_vectors)
        for c in all_candidates:
            if time.time() > deadline - 5:
                break
            if c in existing_set:
                continue
            cn2 = sum(x * x for x in c)
            new_max = max(bmax_norm2, cn2)
            ok = True
            ca = np.array(c, dtype=np.int64)
            d2s = np.sum((best_arr - ca) ** 2, axis=1)
            if d2s.min() < new_max:
                ok = False
            if ok:
                best_vectors.append(c)
                existing_set.add(c)
                bmax_norm2 = max(bmax_norm2, cn2)
                best_arr = np.vstack([best_arr, ca])
                bnorms2 = np.sum(best_arr ** 2, axis=1)
    
    return np.array(best_vectors, dtype=np.int64)
