import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    D = 11
    
    # Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Build adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    adj = [[] for _ in range(n_supports)]
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                adj[i].append(j)
                adj[j].append(i)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []  # which support each vector belongs to
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Precompute pairwise dot products (5280 x 5280)
    # dot[i][j] = dot product of vector i and vector j
    # Two vectors are compatible if dot <= 2 (since norm^2 = 4, min distance >= 4 means (v-u)^2 >= 4, i.e., 8 - 2*dot >= 4, i.e., dot <= 2)
    # Actually: |v-u|^2 = |v|^2 + |u|^2 - 2*v.u = 4 + 4 - 2*v.u = 8 - 2*v.u
    # We need |v-u| >= max_norm = 2, so |v-u|^2 >= 4
    # 8 - 2*v.u >= 4 => 2*v.u <= 4 => v.u <= 2
    
    # Precompute compatibility matrix
    # compatible[i,j] = True if dot(i,j) <= 2
    dots = all_vectors_np @ all_vectors_np.T  # 5280 x 5280
    compatible = dots <= 2  # Boolean matrix
    
    # For each vector, precompute list of compatible vectors
    # Actually, let's just use the boolean matrix directly
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Random ordering of supports
        order = rng.permutation(n_supports)
        
        selected = np.zeros(n_total, dtype=bool)
        selected_count = 0
        selected_by_support = [[] for _ in range(n_supports)]
        
        for s_idx in order:
            # Get all vectors for this support
            # Vectors for support s_idx are at indices: s_idx*16 to s_idx*16+15
            base = s_idx * 16
            candidates = list(range(base, base + 16))
            
            # Filter candidates: must be compatible with all already selected vectors
            # Only need to check against selected vectors from adjacent supports
            # (non-adjacent supports have intersection <= 2, so dot <= 2 always)
            
            # Get adjacent supports that have selected vectors
            adj_with_selected = [n for n in adj[s_idx] if selected_by_support[n]]
            
            local_selected = []
            for cand in candidates:
                ok = True
                for n_idx in adj_with_selected:
                    for sel_idx in selected_by_support[n_idx]:
                        if not compatible[cand, sel_idx]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    local_selected.append(cand)
            
            for idx in local_selected:
                selected[idx] = True
                selected_count += 1
                selected_by_support[s_idx].append(idx)
        
        if selected_count > best_k:
            best_k = selected_count
            best_selected = selected.copy()
        
        # If we've found a good solution, try to improve with local search
        if best_k > 450 and elapsed < deadline - 20:
            # Try to add unselected vectors that are compatible with all selected
            improved = True
            while improved:
                improved = False
                for i in range(n_total):
                    if selected[i]:
                        continue
                    # Check if vector i is compatible with all selected
                    # Vector i belongs to support vec_support[i]
                    s_i = vec_support[i]
                    # Only need to check selected vectors from adjacent supports
                    ok = True
                    for n_idx in adj[s_i]:
                        for sel_idx in selected_by_support[n_idx]:
                            if not compatible[i, sel_idx]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        selected[i] = True
                        selected_count += 1
                        selected_by_support[s_i].append(i)
                        improved = True
                        if selected_count > best_k:
                            best_k = selected_count
                            best_selected = selected.copy()
                
                if elapsed > deadline - 10:
                    break
        
        if best_k > 500:
            break
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
