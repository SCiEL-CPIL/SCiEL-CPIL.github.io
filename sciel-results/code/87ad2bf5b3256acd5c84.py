"""Improved kissing number in dimension 11 via greedy construction."""
import numpy as np
import time

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Generate candidate vectors with small integer entries
    # We'll use vectors with entries in {-2, -1, 0, 1, 2} but focus on
    # sparse vectors first for efficiency
    
    def _generate_candidates():
        """Generate a diverse set of candidate integer vectors."""
        candidates = []
        
        # Weight-1 vectors: single non-zero entry
        for i in range(D):
            for val in [1, -1, 2, -2]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = val
                candidates.append(v)
        
        # Weight-2 vectors: two non-zero entries
        for i in range(D):
            for j in range(i + 1, D):
                for vi in [1, -1, 2, -2]:
                    for vj in [1, -1, 2, -2]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = vi
                        v[j] = vj
                        candidates.append(v)
        
        # Weight-3 vectors: three non-zero entries (subset to keep count manageable)
        for i in range(D):
            for j in range(i + 1, D):
                for k in range(j + 1, D):
                    # Use only ±1 to limit combinations
                    for vi in [1, -1]:
                        for vj in [1, -1]:
                            for vk in [1, -1]:
                                v = np.zeros(D, dtype=np.int64)
                                v[i] = vi
                                v[j] = vj
                                v[k] = vk
                                candidates.append(v)
        
        return candidates
    
    # For the seed approach, we used weight-2 vectors with entries ±1
    # Let's start with a broader candidate pool and use greedy selection
    
    # Actually, let's think about this differently.
    # The constraint is: min pairwise distance >= max norm
    # If all vectors have the same norm (say norm = 2, so squared norm = 4),
    # then we need pairwise squared distance >= 4
    # Squared distance between u and v: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 8 - 2<u,v>
    # We need 8 - 2<u,v> >= 4, i.e., <u,v> <= 2
    # Also ||u||^2 <= 4 for all u
    
    # So we need vectors with squared norm <= 4 and pairwise dot product <= 2
    
    # Let's collect all integer vectors with squared norm exactly 4 or less
    # and non-zero, then greedily select a maximum set with pairwise dot <= 2
    
    def _get_all_vectors_max_norm_sq(max_sq_norm):
        """Get all non-zero integer vectors with squared norm <= max_sq_norm."""
        vecs = []
        
        def _rec(idx, current, sq_norm):
            if sq_norm > max_sq_norm:
                return
            if idx == D:
                if sq_norm > 0:
                    vecs.append(current.copy())
                return
            
            # Try values -2 to 2 (since max_sq_norm=4, individual entries are -2..2)
            for val in [-2, -1, 0, 1, 2]:
                current[idx] = val
                _rec(idx + 1, current, sq_norm + val * val)
            current[idx] = 0
        
        _rec(0, np.zeros(D, dtype=np.int64), 0)
        return vecs
    
    # Get candidates with squared norm up to 4
    # But that's a lot. Let's be smarter: only consider vectors with sq_norm in {1, 2, 3, 4}
    # and use the constraint that pairwise dot <= 2 (when both have norm squared <= 4, 
    # actually the constraint depends on the max norm in the final set)
    
    # Better approach: since we want to maximize the number of vectors, and the constraint
    # is min_dist >= max_norm, let's try to find a large set where all vectors have 
    # the same norm and satisfy the distance constraint.
    
    # For squared norm = 4: need <u,v> <= 2 for all pairs
    # For squared norm = 2: need <u,v> <= 1 for all pairs (since 4 - 2<u,v> >= 2 => <u,v> <= 1)
    
    # Let's try squared norm = 2 first (simpler, more vectors potentially)
    # Vectors with sq_norm = 2: these are weight-1 with entry ±2, or weight-2 with entries ±1
    
    def _greedy_select(candidates, max_sq_norm, max_dot):
        """Greedy selection of vectors satisfying pairwise dot product constraint."""
        selected = []
        remaining = list(range(len(candidates)))
        
        while remaining and time.time() < deadline:
            # Pick the vector that has the most compatible remaining vectors
            # (or just pick the first one for simplicity)
            # For speed, just pick the first available
            idx = remaining[0]
            remaining = remaining[1:]
            
            v = candidates[idx]
            sq_norm = np.dot(v, v)
            
            # Check compatibility with all selected
            compatible = True
            for sv in selected:
                dot = np.dot(v, sv)
                # The constraint: ||v - sv||^2 >= max_norm_sq
                # ||v - sv||^2 = sq_norm + sv_sq_norm - 2*dot
                # We need this >= max_sq_norm (the max norm squared in our set)
                sv_sq = np.dot(sv, sv)
                pair_sq_dist = sq_norm + sv_sq - 2 * dot
                if pair_sq_dist < max_sq_norm:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
        
        return selected
    
    # Strategy: try different max squared norms and find the largest valid set
    best_set = []
    
    # Generate candidate vectors with small entries
    # Weight 1: entries ±1, ±2
    # Weight 2: entries from {-2,-1,0,1,2} on two positions
    # Weight 3: entries ±1 on three positions
    
    candidates = []
    
    # Weight 1
    for i in range(D):
        for val in [-2, -1, 1, 2]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
    
    # Weight 2
    for i in range(D):
        for j in range(i + 1, D):
            for vi in [-2, -1, 1, 2]:
                for vj in [-2, -1, 1, 2]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = vi
                    v[j] = vj
                    candidates.append(v)
    
    # Weight 3 (only ±1 to limit count)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for vi in [-1, 1]:
                    for vj in [-1, 1]:
                        for vk in [-1, 1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = vi
                            v[j] = vj
                            v[k] = vk
                            candidates.append(v)
    
    # Weight 4 (only ±1)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for vi in [-1, 1]:
                        for vj in [-1, 1]:
                            for vk in [-1, 1]:
                                for vl in [-1, 1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = vi
                                    v[j] = vj
                                    v[k] = vk
                                    v[l] = vl
                                    candidates.append(v)
    
    # Now do greedy selection with the constraint
    # For a set S, let M = max{||v||^2 : v in S}
    # For all u,v in S: ||u-v||^2 >= M
    
    # Greedy: add vectors one by one, maintaining the constraint
    # Start with empty set, M = 0
    # When adding v, new M = max(M, ||v||^2)
    # Check all pairs in new set satisfy ||u-w||^2 >= new M
    
    selected = []
    current_max_sq = 0
    
    # Sort candidates by squared norm to process in some order
    # Actually, let's try multiple strategies
    
    # Strategy 1: Greedy by adding vectors, checking constraint
    for v in candidates:
        if time.time() >= deadline:
            break
        v_sq = int(np.dot(v, v))
        if v_sq == 0:
            continue
        new_max = max(current_max_sq, v_sq)
        
        # Check if v can be added
        ok = True
        for sv in selected:
            sv_sq = int(np.dot(sv, sv))
            pair_dist_sq = v_sq + sv_sq - 2 * int(np.dot(v, sv))
            if pair_dist_sq < new_max:
                ok = False
                break
        
        if ok:
            selected.append(v)
            current_max_sq = new_max
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Try to find a large set with uniform norm
    # For squared norm = 4: weight-2 with ±1, or weight-1 with ±2
    # For squared norm = 2: weight-1 with ±2 (no, that's norm^2=4)... 
    # weight-1 with ±1 has norm^2=1, weight-2 with ±1 has norm^2=2
    
    # Let's try all vectors with exactly norm^2 = 4
    cands_norm4 = []
    for i in range(D):
        for val in [-2, 2]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            cands_norm4.append(v)
    for i in range(D):
        for j in range(i + 1, D):
            for vi in [-1, 1]:
                for vj in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = vi
                    v[j] = vj
                    cands_norm4.append(v)
    
    # For norm^2 = 4, need ||u-v||^2 >= 4, i.e., 8 - 2<u,v> >= 4, i.e., <u,v> <= 2
    sel4 = []
    for v in cands_norm4:
        if time.time() >= deadline:
            break
        ok = True
        for sv in sel4:
            if int(np.dot(v, sv)) > 2:
                ok = False
                break
        if ok:
            sel4.append(v)
    
    if len(sel4) > len(best_set):
        best_set = sel4[:]
    
    # Also try norm^2 = 2: weight-2 with ±1
    cands_norm2 = []
    for i in range(D):
        for j in range(i + 1, D):
            for vi in [-1, 1]:
                for vj in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = vi
                    v[j] = vj
                    cands_norm2.append(v)
    
    # For norm^2 = 2, need ||u-v||^2 >= 2, i.e., 4 - 2<u,v> >= 2, i.e., <u,v> <= 1
    sel2 = []
    for v in cands_norm2:
        if time.time() >= deadline:
            break
        ok = True
        for sv in sel2:
            if int(np.dot(v, sv)) > 1:
                ok = False
                break
        if ok:
            sel2.append(v)
    
    if len(sel2) > len(best_set):
        best_set = sel2[:]
    
    # Try norm^2 = 1: weight-1 with ±1
    cands_norm1 = []
    for i in range(D):
        for val in [-1, 1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            cands_norm1.append(v)
    
    # For norm^2 = 1, need ||u-v||^2 >= 1, i.e., 2 - 2<u,v> >= 1, i.e., <u,v> <= 0.5, so <u,v> <= 0
    # Integer dot products, so <u,v> <= 0
    sel1 = []
    for v in cands_norm1:
        if time.time() >= deadline:
            break
        ok = True
        for sv in sel1:
            if int(np.dot(v, sv)) > 0:
                ok = False
                break
        if ok:
            sel1.append(v)
    
    if len(sel1) > len(best_set):
        best_set = sel1[:]
    
    if not best_set:
        # Fallback: return the seed result
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_set, dtype=np.int64)
