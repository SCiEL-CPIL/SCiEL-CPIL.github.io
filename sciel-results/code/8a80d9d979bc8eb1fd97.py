"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Generate all roots of D11 lattice (vectors with two non-zero entries of +/- 1)
    # Norm squared = 2. All 220 roots are mutually compatible (dot <= 1).
    
    roots = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
                    
    roots_arr = np.array(roots, dtype=np.float64)
    n_roots = len(roots_arr)
    
    # Generate a broader candidate pool with various norms
    # We'll use vectors with squared norm 2, 4, 6, 8, etc.
    # Key insight: we need min_dist >= max_norm, so if max_norm_sq = M,
    # then for any pair u,v: ||u-v||^2 >= M => u.u + v.v - 2u.v >= M
    # => u.v <= (u.u + v.v - M) / 2
    
    # Strategy: Start with roots (norm^2=2), then greedily add vectors
    # with larger norms, checking compatibility with all existing vectors.
    
    # Generate candidates with various norms
    candidates = []
    
    # Norm^2 = 4: one entry +/-2, or four entries +/-1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
    
    # Norm^2 = 6: three entries +/-1 and one entry +/-sqrt(3)? No, must be integer.
    # Three entries: one +/-2 and two +/-1? 4+1+1=6. Yes!
    for i in range(D):
        for s_i in (1, -1):
            for j in range(D):
                if j == i: continue
                for s_j in (1, -1):
                    for k in range(D):
                        if k == i or k == j: continue
                        for s_k in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = 2 * s_i
                            v[j] = s_j
                            v[k] = s_k
                            candidates.append(v)
    
    # Norm^2 = 8: four entries +/-2? 4*4=16 no. Two entries +/-2: 4+4=8.
    for i in range(D):
        for j in range(i+1, D):
            for s_i in (1, -1):
                for s_j in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * s_i
                    v[j] = 2 * s_j
                    candidates.append(v)
    
    # Eight entries +/-1: 8*1=8
    for support in itertools.combinations(range(D), 8):
        for signs in itertools.product([-1, 1], repeat=8):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
    
    cands = np.array(candidates, dtype=np.float64)
    n_cands = len(cands)
    
    # Compute squared norms of candidates
    cand_norms_sq = np.sum(cands * cands, axis=1)
    
    # Filter candidates that are compatible with ALL roots
    # For a candidate v with norm^2 = c, and root r with norm^2 = 2:
    # ||v-r||^2 >= max(2, c) => c + 2 - 2*v.r >= max(2,c)
    # If c >= 2: c + 2 - 2*v.r >= c => 2 >= 2*v.r => v.r <= 1
    # If c < 2: c + 2 - 2*v.r >= 2 => c >= 2*v.r => v.r <= c/2
    # Since all our candidates have c >= 4, condition is v.r <= 1 for all roots.
    
    if time.time() < deadline - 10 and n_cands > 0:
        dots_roots_cands = roots_arr @ cands.T
        max_dots = np.max(dots_roots_cands, axis=0)
        valid_cand_mask = max_dots <= 1
        valid_cands = cands[valid_cand_mask]
        valid_cand_norms_sq = cand_norms_sq[valid_cand_mask]
    else:
        valid_cands = np.array([], dtype=np.float64)
        valid_cand_norms_sq = np.array([], dtype=np.float64)
    
    # Greedily select from valid_cands using vectorized conflict checking
    # Maintain selected vectors and their norms
    selected_indices = []
    selected_vectors = []
    selected_norms_sq = []
    
    if len(valid_cands) > 0:
        # Sort by norm descending to try larger norms first (they're more restrictive)
        # Actually, let's sort by norm ascending - smaller norms are easier to fit
        order = np.argsort(valid_cand_norms_sq)
        
        # Track which candidates are still available
        available = np.ones(len(valid_cands), dtype=bool)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if not available[idx]:
                continue
            
            v = valid_cands[idx]
            v_norm_sq = valid_cand_norms_sq[idx]
            
            # Check compatibility with all selected vectors
            compatible = True
            for i, sv in enumerate(selected_vectors):
                sv_norm_sq = selected_norms_sq[i]
                max_norm_sq = max(v_norm_sq, sv_norm_sq)
                # Need: ||v - sv||^2 >= max_norm_sq
                # v.v + sv.sv - 2*v.sv >= max_norm_sq
                # v_norm_sq + sv_norm_sq - 2*dot >= max_norm_sq
                dot = np.dot(v, sv)
                dist_sq = v_norm_sq + sv_norm_sq - 2 * dot
                if dist_sq < max_norm_sq - 1e-9:
                    compatible = False
                    break
            
            if compatible:
                selected_indices.append(idx)
                selected_vectors.append(v)
                selected_norms_sq.append(v_norm_sq)
    
    # Combine roots and selected vectors
    if len(selected_vectors) > 0:
        norm4_vectors = np.array(selected_vectors)
        result = np.vstack([roots_arr, norm4_vectors]).astype(np.int64)
    else:
        result = roots_arr.astype(np.int64)
    
    # Verification
    if len(result) > 1:
        norms_sq = np.sum(result * result, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        n = len(result)
        valid = True
        for i in range(n):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, n):
                diff = result[i] - result[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            result = roots_arr.astype(np.int64)
            
    return result
# EVOLVE-BLOCK-END
