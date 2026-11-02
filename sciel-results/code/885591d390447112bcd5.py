"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Generate all ±1 vectors of weight 1 through 4 in R^11
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Weight 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight 3
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Weight 4
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1).astype(np.int64)
    n_cand = len(candidates)
    
    best_selected = None
    best_len = 0
    rng = np.random.RandomState(42)
    
    distinct_norms = np.unique(norms_sq)
    
    for M in distinct_norms:
        if time.time() > deadline:
            break
        
        M_int = int(M)
        
        # Get candidates with norm^2 <= M
        mask = norms_sq <= M
        cand_indices = np.where(mask)[0]
        n_sub = len(cand_indices)
        if n_sub == 0:
            continue
        
        # Build conflict graph: edge(i,j) if ||v_i - v_j||^2 < M
        sub_cands = candidates[cand_indices]
        gram = sub_cands @ sub_cands.T
        sub_norms = norms_sq[cand_indices]
        dist_sq = sub_norms[:, None].astype(np.int64) + sub_norms[None, :].astype(np.int64) - 2 * gram.astype(np.int64)
        np.fill_diagonal(dist_sq, 0)
        
        # Conflict: dist_sq < M means NOT compatible
        conflict = (dist_sq < M_int).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        
        # Create mapping from local index to global index
        local_to_global = cand_indices  # cand_indices[local_idx] = global_idx
        
        # Try multiple greedy orderings to find large independent sets
        deg = conflict.sum(axis=1).astype(np.int32)
        
        # Spectral ordering via power iteration on conflict adjacency
        if n_sub > 1:
            v_rand = rng.randn(n_sub).astype(np.float64)
            v_rand -= v_rand.mean()
            vn = np.linalg.norm(v_rand)
            if vn > 1e-10:
                v_rand /= vn
            else:
                v_rand = np.zeros(n_sub)
                v_rand[0] = 1.0
            
            for _ in range(20):
                v_new = conflict.astype(np.float64) @ v_rand
                v_new -= v_new.mean()
                vn = np.linalg.norm(v_new)
                if vn < 1e-10:
                    break
                v_new /= vn
                if np.linalg.norm(v_new - v_rand) < 1e-6:
                    break
                v_rand = v_new
            
            spectral_order = np.argsort(v_rand)
        else:
            spectral_order = np.array([0])
        
        degree_order = np.argsort(deg)
        
        # Also try: order by (degree, then random tiebreak)
        # And several random orderings
        orders = [spectral_order, degree_order]
        for _ in range(8):
            orders.append(rng.permutation(n_sub))
        
        for order in orders:
            if time.time() > deadline:
                break
            
            selected_local = []
            selected_bool = np.zeros(n_sub, dtype=bool)
            
            for idx in order:
                if time.time() > deadline:
                    break
                
                if selected_bool[idx]:
                    continue
                
                # Check: is idx compatible with all already selected?
                # i.e., no conflict edge from idx to any selected node
                if len(selected_local) > 0:
                    if conflict[idx, selected_local].any():
                        continue
                
                selected_local.append(idx)
                selected_bool[idx] = True
            
            # Convert to global indices
            global_sel = cand_indices[selected_local]
            
            if len(global_sel) > best_len:
                best_len = len(global_sel)
                best_selected = global_sel.copy()
        
        # Local search: try to add vertices to the best set for this M
        if best_selected is not None and best_len > 0:
            # Work in local coordinates
            # Find which local indices correspond to best_selected
            global_set = set(best_selected.tolist())
            local_selected = []
            for li in range(n_sub):
                if cand_indices[li] in global_set:
                    local_selected.append(li)
            
            local_selected_arr = np.array(local_selected)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for li in range(n_sub):
                    if time.time() > deadline:
                        break
                    if li in local_selected:
                        continue
                    # Check compatibility
                    if len(local_selected) > 0:
                        if conflict[li, local_selected].any():
                            continue
                    local_selected.append(li)
                    improved = True
            
            if len(local_selected) > best_len:
                best_len = len(local_selected)
                best_selected = cand_indices[local_selected].copy()
    
    if best_selected is None or best_len == 0:
        # Fallback: weight-2 vectors only
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
