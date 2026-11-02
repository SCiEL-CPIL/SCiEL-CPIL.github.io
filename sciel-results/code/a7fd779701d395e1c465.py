import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 54
    # Strategy:
    # Split m=5 into 4+1
    # Split p=6 into 4+2
    # A = [A1, A2] (2x4, 2x1)
    # B = [[B11, B12], [B21, B22]] (4x4, 4x2, 1x4, 1x2)
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # Ranks:
    # A1 B11: 2x4x4 -> 4 * Strassen(2x2x2) -> 4 * 7 = 28
    # A1 B12: 2x4x2 -> 2 * Strassen(2x2x2) -> 2 * 7 = 14
    # A2 B21: 2x1x4 -> Outer -> 2 * 4 = 8
    # A2 B22: 2x1x2 -> Outer -> 2 * 2 = 4
    # Total: 28 + 14 + 8 + 4 = 54
    
    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # Local indices: i in 0..1, j in 0..1, k in 0..1
    # Global offsets: i_off, j_off, k_off
    # U index: (i_off+i)*m + (j_off+j)
    # V index: (j_off+j)*p + (k_off+k)
    # W index: (k_off+k)*n + (i_off+i)
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen 2x2x2 decomposition
        # 7 terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add contributions to U, V, W
        # U[i,j] corresponds to A_ij
        # V[j,k] corresponds to B_jk
        # W[k,i] corresponds to C_ki (trace convention W[k*n+i])
        
        # Term 1: (A11+A22)(B11+B22)
        # U: A11 (+1), A22 (+1)
        # V: B11 (+1), B22 (+1)
        # W: C11 (+1), C22 (+1)
        
        # This is a sum of products.
        # Each Strassen term is a product of two sums.
        # (A_u + A_v)(B_x + B_y) = A_u B_x + A_u B_y + A_v B_x + A_v B_y
        # This expands to 4 rank-1 terms?
        # No, Strassen reduces 8 multiplications to 7.
        # The 7 "multiplications" are the rank-1 terms in the tensor decomposition?
        # Yes.
        
        # Let's define the 7 terms explicitly.
        # Each term r has:
        # U contributions: list of (local_i, local_j, coeff)
        # V contributions: list of (local_j, local_k, coeff)
        # W contributions: list of (local_k, local_i, coeff)
        
        # Actually, it's easier to just list the 7 rank-1 tensors.
        # A rank-1 tensor is u v^T w^T?
        # T = sum u_r v_r w_r
        # u_r is vector of size 4 (2x2)
        # v_r is vector of size 4 (2x2)
        # w_r is vector of size 4 (2x2)
        
        # We can hardcode the 7 vectors for Strassen.
        # u, v, w for each of the 7 terms.
        
        # Term 1: M1
        # u1 = [1, 0, 0, 1] (A11, A12, A21, A22) -> A11 + A22
        # v1 = [1, 0, 0, 1] (B11, B12, B21, B22) -> B11 + B22
        # w1 = [1, 0, 1, 0] (C11, C12, C21, C22) -> C11 + C22
        # Wait, W index is k*n+i.
        # C11: k=0, i=0 -> 0*2+0=0
        # C12: k=1, i=0 -> 1*2+0=2
        # C21: k=0, i=1 -> 0*2+1=1
        # C22: k=1, i=1 -> 1*2+1=3
        # So w1 = [1, 0, 0, 1] (C11, C21, C12, C22)?
        # Order in W matrix: index 0,1,2,3.
        # w1[0]=1, w1[1]=0, w1[2]=0, w1[3]=1.
        
        # Let's verify indices.
        # U index: i*m+j. m=2.
        # A11: 0*2+0=0
        # A12: 0*2+1=1
        # A21: 1*2+0=2
        # A22: 1*2+1=3
        
        # V index: j*p+k. p=2.
        # B11: 0*2+0=0
        # B12: 0*2+1=1
        # B21: 1*2+0=2
        # B22: 1*2+1=3
        
        # W index: k*n+i. n=2.
        # C11: 0*2+0=0
        # C21: 0*2+1=1
        # C12: 1*2+0=2
        # C22: 1*2+1=3
        
        # Term 1:
        # u = [1, 0, 0, 1]
        # v = [1, 0, 0, 1]
        # w = [1, 0, 0, 1]
        
        # Term 2: M2 = (A21+A22)B11
        # u = [0, 0, 1, 1]
        # v = [1, 0, 0, 0]
        # w = [0, 1, 0, 0] (C21)
        
        # Term 3: M3 = A11(B12-B22)
        # u = [1, 0, 0, 0]
        # v = [0, 1, 0, -1]
        # w = [0, 0, 1, 1] (C12 + C22)
        
        # Term 4: M4 = A22(B21-B11)
        # u = [0, 0, 0, 1]
        # v = [0, 0, 1, -1]
        # w = [1, 1, 0, 0] (C11 + C21)
        
        # Term 5: M5 = (A11+A12)B22
        # u = [1, 1, 0, 0]
        # v = [0, 0, 0, 1]
        # w = [-1, 0, 0, 1] (-C11 + C22)
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # u = [-1, 0, 1, 0]
        # v = [1, 1, 0, 0]
        # w = [0, 0, 0, 1] (C22)
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # u = [0, 1, 0, -1]
        # v = [0, 0, 1, 1]
        # w = [1, 0, 0, 0] (C11)
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        for u, v, w in terms:
            # Add to global U, V, W
            for li in range(2):
                for lj in range(2):
                    if u[lj + 2*li] != 0: # u index lj + 2*li? No, u index is lj + 2*li?
                       # u is flattened 2x2. Index = li*2 + lj.
                       pass
            
            # Correct indexing
            for idx in range(4):
                val = u[idx]
                if val != 0:
                    li = idx // 2
                    lj = idx % 2
                    gi = i_off + li
                    gj = j_off + lj
                    U[gi * m + gj][col] += val
                
                val = v[idx]
                if val != 0:
                    lj = idx // 2
                    lk = idx % 2
                    gj = j_off + lj
                    gk = k_off + lk
                    V[gj * p + gk][col] += val
                
                val = w[idx]
                if val != 0:
                    lk = idx // 2
                    li = idx % 2
                    gk = k_off + lk
                    gi = i_off + li
                    W[gk * n + gi][col] += val
            
            col += 1

    # Block 1: A1 B11 (2x4x4)
    # i_off=0
    # j_off=0, k_off=0 (B11)
    add_strassen(0, 0, 0)
    # j_off=0, k_off=2 (B12 is 4x2? No B11 is 4x4. B is split 4+2.
    # B11 is 4x4. j_off=0..3, k_off=0..3.
    # A1 is 2x4. j_off=0..3.
    # So B11 is at j=0..3, k=0..3.
    # We need 4 Strassen blocks for 4x4.
    # (0,0), (0,2), (2,0), (2,2)
    
    # I already added (0,0).
    add_strassen(0, 2, 0) # j=2, k=0
    add_strassen(0, 0, 2) # j=0, k=2
    add_strassen(0, 2, 2) # j=2, k=2
    
    # Block 2: A1 B12 (2x4x2)
    # j_off=0..3, k_off=4..5
    # 2 Strassen blocks?
    # B12 is 4x2.
    # Split j into 0..1, 2..3.
    add_strassen(0, 0, 4) # j=0, k=4
    add_strassen(0, 2, 4) # j=2, k=4
    
    # Block 3: A2 B21 (2x1x4)
    # j_off=4, k_off=0..3
    # Outer product.
    # A2 is 2x1. j=4.
    # B21 is 1x4. k=0..3.
    # Rank 8.
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 4: A2 B22 (2x1x2)
    # j_off=4, k_off=4..5
    # Outer product.
    # Rank 4.
    for i in range(2):
        for k in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + (4 + k)][col] = 1
            W[(4 + k) * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
