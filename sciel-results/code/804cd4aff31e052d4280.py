import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 145  # 145s out of 150s
    
    # Generate all vectors with exactly two non-zero entries (+/- 1)
    # These correspond to the minimal norm vectors in the D_11 lattice.
    vectors = []
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    if not vectors:
        # Fallback to basis vectors if something goes wrong
        vectors = [[0]*D for _ in range(2*D)]
        for i in range(D):
            vectors[i][i] = 1
            vectors[i+D][i] = -1
            
    arr = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify constraint to be safe
    # If verification fails, return a smaller valid set
    if len(arr) > 0:
        norms = np.linalg.norm(arr, axis=1)
        max_norm = np.max(norms)
        
        # Calculate min pairwise distance
        # For 220 vectors, this is fast
        min_dist = float('inf')
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                dist = np.linalg.norm(arr[i] - arr[j])
                if dist < min_dist:
                    min_dist = dist
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if min_dist < max_norm:
            # If invalid, return basis vectors (norm 1, dist 2 between orthogonal, 2 between opposite? No, dist 2 between e_i and -e_i is 2. Norm 1. 2>=1. OK.)
            # Basis vectors: e_i and -e_i.
            # Dist between e_i and e_j (i!=j) is sqrt(2). Norm 1. sqrt(2) >= 1. OK.
            # Dist between e_i and -e_i is 2. Norm 1. 2 >= 1. OK.
            # So 22 basis vectors are always valid.
            fallback = []
            for i in range(D):
                v_pos = [0] * D
                v_neg = [0] * D
                v_pos[i] = 1
                v_neg[i] = -1
                fallback.append(v_pos)
                fallback.append(v_neg)
            return np.array(fallback, dtype=np.int64)

    return arr
