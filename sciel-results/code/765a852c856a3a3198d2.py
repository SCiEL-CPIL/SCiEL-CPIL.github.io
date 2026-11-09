import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate Weight 3 vectors
    vecs_w3 = []
    for combo in itertools.combinations(range(D), 3):
        i, j, k_idx = combo
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k_idx] = s1, s2, s3
                    vecs_w3.append(v)
    
    V3 = np.array(vecs_w3, dtype=np.int8)
    N3 = len(V3)
    
    # 2. Generate Weight 2 vectors
    vecs_w2 = []
    for combo in itertools.combinations(range(D), 2):
        i, j = combo
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i], v[j] = s1, s2
                vecs_w2.append(v)
                
    V2 = np.array(vecs_w2, dtype=np.int8)
    N2 = len(V2)
    
    # 3. Compute Conflict Matrices
    # Conflict W3: dot >= 2
    dots3 = V3 @ V3.T
    np.fill_diagonal(dots3, -1)
    conflicts3 = dots3 >= 2
    
    # Precompute neighbors for W3
    neighbors3 = [np.where(conflicts3[i])[0] for i in range(N3)]
    
    # Conflict W2-W2: dot >= 1 (since 4-2dot >= 3 => dot <= 0.5 => dot <= 0 is OK, dot >= 1 is Conflict)
    # Wait, for W2-W2 in mixed set (R=sqrt(3)):
    # ||u-v||^2 >= 3 => 4 - 2 dot >= 3 => 2 dot <= 1 => dot <= 0.
    # So conflict if dot >= 1.
    dots2 = V2 @ V2.T
    np.fill_diagonal(dots2, -1)
    conflicts2_2 = dots2 >= 1
    
    # Conflict W2-W3: dot >= 2 (since 2+3-2dot >= 3 => 2dot <= 2 => dot <= 1. Conflict if dot >= 2)
    # Note: Max dot between W2 and W3 is 2 (share 2 indices, same signs).
    dots23 = V2 @ V3.T
    conflicts23 = dots23 >= 2
    
    # 4. Heuristic for MIS on W3
    best_selected3 = None
    best_k3 = 0
    
    # Randomized Greedy
    num_restarts = 2000
    for _ in range(num_restarts):
        if time.time() > deadline - 20.0: # Leave time for W2
            break
            
        available = np.ones(N3, dtype=bool)
        selected = []
        
        # To speed up, we can pick random available node
        # But np.where(available) creates a list.
        # We can use a set or just random index and check.
        # Given N=1320, checking is fast.
        
        while np.any(available):
            # Pick random available index
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            u_idx = avail_indices[np.random.randint(len(avail_indices))]
            
            selected.append(u_idx)
            available[u_idx] = False
            # Remove neighbors
            if len(neighbors3[u_idx]) > 0:
                available[neighbors3[u_idx]] = False
        
        k_curr = len(selected)
        if k_curr > best_k3:
            best_k3 = k_curr
            best_selected3 = np.array(selected, dtype=int)
            
    if best_selected3 is None:
        best_selected3 = np.array([], dtype=int)
        
    S3 = V3[best_selected3]
    
    # 5. Find compatible W2 vectors
    # For each W2 vector, check if it conflicts with any in S3
    # conflicts23 is (N2, N3)
    # We need col j to be False for all j in best_selected3
    
    if len(best_selected3) > 0:
        # Mask of W2 vectors that are compatible with S3
        # conflicts23[:, best_selected3] is (N2, K3)
        # We want no True in this sub-matrix
        compatible_w2_mask = ~np.any(conflicts23[:, best_selected3], axis=1)
    else:
        compatible_w2_mask = np.ones(N2, dtype=bool)
        
    # Get indices of compatible W2
    compat_w2_indices = np.where(compatible_w2_mask)[0]
    
    # Now find MIS on this subset of W2 with conflicts2_2
    # Subset size might be large.
    # Greedy on W2 subset
    best_selected2 = []
    if len(compat_w2_indices) > 0:
        # Build local conflict matrix for these W2 vectors
        # Or just use global indices
        # Greedy
        available_w2 = np.zeros(N2, dtype=bool)
        available_w2[compat_w2_indices] = True
        
        # Precompute neighbors for W2 (global)
        neighbors2 = [np.where(conflicts2_2[i])[0] for i in range(N2)]
        
        num_restarts_w2 = 500
        best_k2 = 0
        
        for _ in range(num_restarts_w2):
            if time.time() > deadline - 5.0:
                break
                
            avail = available_w2.copy()
            sel = []
            
            while np.any(avail):
                avail_indices = np.where(avail)[0]
                if len(avail_indices) == 0:
                    break
                u_idx = avail_indices[np.random.randint(len(avail_indices))]
                sel.append(u_idx)
                avail[u_idx] = False
                if len(neighbors2[u_idx]) > 0:
                    avail[neighbors2[u_idx]] = False
                    
            if len(sel) > best_k2:
                best_k2 = len(sel)
                best_selected2 = sel
                
    S2 = V2[best_selected2] if best_selected2 else np.array([], dtype=np.int8).reshape(0, D)
    
    # Combine
    if len(S3) > 0 and len(S2) > 0:
        result = np.vstack((S3, S2))
    elif len(S3) > 0:
        result = S3
    elif len(S2) > 0:
        result = S2
    else:
        # Fallback to original weight 2
        result = V2
        
    return result.astype(np.int64)
