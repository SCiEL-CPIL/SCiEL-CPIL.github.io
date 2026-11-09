import numpy as np

def construct():
    n, m, p = 2, 3, 8
    R = 44
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Define sub-decomposition for 2x3x4
    # u: 6 entries
    # v: 12 entries
    # w: 8 entries
    
    sub_vectors = []
    
    # Term 1: A1 B11 (Rank 7)
    # M1
    u = [1, 0, 0, 0, 1, 0]
    v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    w = [1, 0, 0, 1, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M2
    u = [0, 0, 0, 1, 1, 0]
    v = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 1, 0, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M3
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 1, 1, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M4
    u = [0, 0, 0, 0, 1, 0]
    v = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [1, 1, 0, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M5
    u = [1, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    w = [-1, 0, 1, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M6
    u = [-1, 0, 0, 1, 0, 0]
    v = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 1, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # M7
    u = [0, 1, 0, 0, -1, 0]
    v = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    w = [1, 0, 0, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # Term 2: A2 B21 (Rank 4)
    # A02 B20 -> C00
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    w = [1, 0, 0, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # A02 B21 -> C01
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    w = [0, 0, 1, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # A12 B20 -> C10
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    w = [0, 1, 0, 0, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # A12 B21 -> C11
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    w = [0, 0, 0, 1, 0, 0, 0, 0]
    sub_vectors.append((u, v, w))
    
    # Term 3: A1 B12 (Rank 7)
    # Shift v by +2, w by +4
    for u, v, w in sub_vectors[:7]:
        v_new = [0] * 12
        w_new = [0] * 8
        for idx, val in enumerate(v):
            v_new[idx + 2] = val
        for idx, val in enumerate(w):
            w_new[idx + 4] = val
        sub_vectors.append((u, v_new, w_new))
        
    # Term 4: A2 B22 (Rank 4)
    # Shift v by +2, w by +4
    for u, v, w in sub_vectors[7:11]:
        v_new = [0] * 12
        w_new = [0] * 8
        for idx, val in enumerate(v):
            v_new[idx + 2] = val
        for idx, val in enumerate(w):
            w_new[idx + 4] = val
        sub_vectors.append((u, v_new, w_new))
        
    # Total 22 vectors
    
    # Fill global matrices
    # First half (k=0..3)
    for r in range(22):
        u, v, w = sub_vectors[r]
        for i in range(6):
            U[i][r] = u[i]
        for i in range(12):
            V[i][r] = v[i]
        for i in range(8):
            W[i][r] = w[i]
            
    # Second half (k=4..7)
    for r in range(22):
        u, v, w = sub_vectors[r]
        col = 22 + r
        for i in range(6):
            U[i][col] = u[i]
        # Shift V indices
        # Local v index j*4 + k' -> Global j*8 + (4+k')
        for j in range(3):
            for k_prime in range(4):
                local_idx = j * 4 + k_prime
                global_idx = j * 8 + (4 + k_prime)
                V[global_idx][col] = v[local_idx]
        # Shift W indices
        # Local w index k'*2 + i -> Global (4+k')*2 + i
        for k_prime in range(4):
            for i in range(2):
                local_idx = k_prime * 2 + i
                global_idx = (4 + k_prime) * 2 + i
                W[global_idx][col] = w[local_idx]
                
    return {"U": U, "V": V, "W": W}
