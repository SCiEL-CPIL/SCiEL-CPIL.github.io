import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    # u, v, w are vectors of length 4 (indices 0:00, 1:01, 2:10, 3:11)
    triads_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Build rank 49 decomposition for 4x4x4
    # U4: 16 x 49, V4: 16 x 49, W4: 16 x 49
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    col = 0
    for r in range(7):
        u_r, v_r, w_r = triads_2x2[r]
        # Apply to 2x2 blocks of 4x4 matrices
        # A is 2x2 blocks of 2x2.
        # u_r has 4 components, each is a scalar multiplier for a 2x2 block of A.
        # v_r has 4 components, each is a scalar multiplier for a 2x2 block of B.
        # w_r has 4 components, each is a scalar multiplier for a 2x2 block of C.
        
        # For each sub-triad in 2x2
        for s in range(7):
            u_s, v_s, w_s = triads_2x2[s]
            
            # u_r[i] * u_s[k] contributes to A block i, element k
            # v_r[j] * v_s[l] contributes to B block j, element l
            # w_r[m] * w_s[n] contributes to C block m, element n
            
            # A indices: block i (0..3), element k (0..3) -> idx = i*4 + k
            # B indices: block j (0..3), element l (0..3) -> idx = j*4 + l
            # C indices: block m (0..3), element n (0..3) -> idx = m*4 + n
            
            for i in range(4):
                for k in range(4):
                    val_u = u_r[i] * u_s[k]
                    if val_u != 0:
                        U4[i*4 + k][col] = val_u
                        
            for j in range(4):
                for l in range(4):
                    val_v = v_r[j] * v_s[l]
                    if val_v != 0:
                        V4[j*4 + l][col] = val_v
                        
            for m_idx in range(4):
                for n_idx in range(4):
                    val_w = w_r[m_idx] * w_s[n_idx]
                    if val_w != 0:
                        W4[m_idx*4 + n_idx][col] = val_w
                        
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Now construct for 4x4x8
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 8x4. Split into C1 (4x4) and C2 (4x4).
    # Rank 98.
    
    U = [[0] * 98 for _ in range(16)]
    V = [[0] * 98 for _ in range(32)]
    W = [[0] * 98 for _ in range(32)]
    
    # Copy U4 to U (same for both halves)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][c + 49] = U4[i][c]
            
    # Copy V4 to V
    # V4 is 16x49. V is 32x98.
    # B1 is rows 0-3, cols 0-3 of B. Indices 0-15 in V4?
    # V4 index j*4+l corresponds to B[j,l].
    # In global B (4x8), B1 is j in 0..3, k in 0..3. Index j*8 + k.
    # B2 is j in 0..3, k in 4..7. Index j*8 + (k-4).
    
    for j in range(4):
        for l in range(4):
            # V4 index
            idx_v4 = j*4 + l
            # V index for B1
            idx_v1 = j*8 + l
            # V index for B2
            idx_v2 = j*8 + (l + 4)
            
            for c in range(49):
                val = V4[idx_v4][c]
                V[idx_v1][c] = val
                V[idx_v2][c + 49] = val
                
    # Copy W4 to W
    # W4 is 16x49. W is 32x98.
    # C1 is k in 0..3, i in 0..3. Index k*4 + i.
    # C2 is k in 4..7, i in 0..3. Index k*4 + i.
    # W4 index m*4+n corresponds to C[m,n].
    
    for m_idx in range(4):
        for n_idx in range(4):
            idx_w4 = m_idx*4 + n_idx
            idx_w1 = m_idx*4 + n_idx
            idx_w2 = (m_idx + 4)*4 + n_idx
            
            for c in range(49):
                val = W4[idx_w4][c]
                W[idx_w1][c] = val
                W[idx_w2][c + 49] = val
                
    return {"U": U, "V": V, "W": W}

