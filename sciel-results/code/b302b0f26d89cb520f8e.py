import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We construct the decomposition using a recursive Strassen-like approach.
    # The tensor T is defined by T[i*4+j, j*8+k, k*2+i] = 1.
    # We decompose it into 8 blocks of 2x2x2.
    
    # Strassen 2x2x2 decomposition factors
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    
    S_U = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1]
    ]
    # Wait, the standard Strassen matrices are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let's define the rank-1 factors for the 7 multiplications.
    # Each multiplication is a tensor of rank 1? No, each M is a product of two linear forms.
    # So each M corresponds to 1 rank in the bilinear decomposition.
    
    # U factors (for A):
    # M1: A11+A22 -> [1,0,0,1]
    # M2: A21+A22 -> [0,1,1,0]
    # M3: A11 -> [1,0,0,0]
    # M4: A22 -> [0,0,0,1]
    # M5: A11+A12 -> [1,1,0,0]
    # M6: A21-A11 -> [-1,1,0,0]
    # M7: A12-A22 -> [0,1,-1,0]
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, -1],
        [1, 0, 0, 1, 0, 0, 0]
    ])
    
    # V factors (for B):
    # M1: B11+B22 -> [1,0,0,1]
    # M2: B11 -> [1,0,0,0]
    # M3: B21-B22 -> [0,1,-1,0]
    # M4: B12-B11 -> [-1,0,1,0]
    # M5: B22 -> [0,0,0,1]
    # M6: B11+B12 -> [1,1,0,0]
    # M7: B21+B22 -> [0,1,1,0]
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ])
    
    # W factors (for C):
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # W is 4x7, W[i*2+j, r]
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ])
    
    # Verify Strassen correctness for 2x2x2
    # C = W^T (V^T B U)? No.
    # T = sum_r U[:,r] V[:,r]^T W[:,r]^T?
    # The decomposition is T = sum_r u_r v_r^T w_r^T?
    # In our convention: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We have 8 blocks.
    # Block (i_off, j_off, k_off)
    # i_off in {0} (since n=2, only one 2-row block)
    # j_off in {0, 2} (m=4, two 2-col blocks)
    # k_off in {0, 2, 4, 6} (p=8, four 2-col blocks)
    
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            i_off = 0
            base_rank = block_idx * 7
            for r in range(7):
                cr = base_rank + r
                # Add U_strassen column r to U at positions corresponding to (i_off, j_off)
                for li in range(2):
                    for lj in range(2):
                        gi = li + i_off
                        gj = lj + j_off
                        U[gi * 4 + gj, cr] += U_strassen[li * 2 + lj, r]
                        
                for lj in range(2):
                    for lk in range(2):
                        gj = lj + j_off
                        gk = lk + k_off
                        V[gj * 8 + gk, cr] += V_strassen[lj * 2 + lk, r]
                        
                for lk in range(2):
                    for li in range(2):
                        gk = lk + k_off
                        gi = li + i_off
                        W[gk * 2 + gi, cr] += W_strassen[lk * 2 + li, r]
            
            block_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
