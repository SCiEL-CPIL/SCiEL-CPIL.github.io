import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    # W index: W[k*n + i, r]
    
    # Structural approach: Kronecker product of Strassen (rank 7) for outer 2x2
    # with identity (rank 4) for inner ring multiplication
    # 
    # Model: 2x4 * 4x8 = 2x8
    # Partition A (2x4) into 2x2 blocks of size 1x2:
    #   A = [[A00, A01], [A10, A11]] where Aij is 1x2
    # Partition B (4x8) into 2x2 blocks of size 2x4:
    #   B = [[B00, B01], [B10, B11]] where Bij is 2x4
    # C (2x8) = [[C00, C01], [C10, C11]] where Cij is 1x4
    #
    # This is 2x2 matrix mult over ring R where R elements multiply as:
    #   (1x2) * (2x4) = (1x4)  -- rank 4 per ring multiplication
    #
    # Strassen for outer 2x2: rank 7
    # Inner ring mult: rank 4
    # Total rank: 7 * 4 = 28
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (a0+a3)(b0+b3) -> c0 += M1, c3 += M1
    # M2 = (a2+a3)(b0)    -> c2 += M2, c3 -= M2
    # M3 = (a0)(b1-b3)    -> c1 += M3, c3 += M3
    # M4 = (a3)(b2-b0)    -> c0 += M4, c2 += M4
    # M5 = (a0+a1)(b3)    -> c0 -= M5, c1 += M5
    # M6 = (a2-a0)(b0+b1) -> c3 += M6
    # M7 = (a1-a3)(b2+b3) -> c0 += M7
    
    strassen_U = np.array([
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ], dtype=int)
    
    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ], dtype=int)
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify Strassen
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_test = A_test @ B_test
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed = np.zeros(4)
    for r in range(7):
        prod = np.dot(A_flat, strassen_U[:,r]) * np.dot(B_flat, strassen_V[:,r])
        C_computed += strassen_W[:,r] * prod
    
    if not np.allclose(C_computed, C_test.flatten()):
        # Fallback to naive
        total_rank = n*m*p
        U = np.zeros((n*m, total_rank), dtype=int)
        V = np.zeros((m*p, total_rank), dtype=int)
        W = np.zeros((p*n, total_rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full decomposition using Kronecker product
    # A is 2x4, indexed as A[i, j] -> flat index i*4 + j
    # B is 4x8, indexed as B[j, k] -> flat index j*8 + k
    # C is 2x8, indexed as C[i, k] -> flat index k*2 + i (W convention: k*n+i)
    #
    # Block structure:
    # A[i, j] where i in {0,1} (outer row), j in {0,1,2,3}
    #   A00 = A[0, 0:2] (1x2), A01 = A[0, 2:4] (1x2)
    #   A10 = A[1, 0:2] (1x2), A11 = A[1, 2:4] (1x2)
    #
    # B[j, k] where j in {0,1,2,3} (outer col), k in {0,...,7}
    #   B00 = B[0:2, 0:4] (2x4), B01 = B[0:2, 4:8] (2x4)
    #   B10 = B[2:4, 0:4] (2x4), B11 = B[2:4, 4:8] (2x4)
    #
    # C[i, k] where i in {0,1} (outer row), k in {0,...,7}
    #   C00 = C[0, 0:4] (1x4), C01 = C[0, 4:8] (1x4)
    #   C10 = C[1, 0:4] (1x4), C11 = C[1, 4:8] (1x4)
    #
    # Ring multiplication: (1x2) * (2x4) = (1x4) has rank 4
    # The rank-4 decomposition for 1x2 * 2x4 -> 1x4:
    #   (a0, a1) * [[b00,b01,b02,b03],[b10,b11,b12,b13]] = (a0*b00+a1*b10, a0*b01+a1*b11, a0*b02+a1*b12, a0*b03+a1*b13)
    #   This is just 4 independent scalar multiplications:
    #   c0 = a0*b00 + a1*b10
    #   c1 = a0*b01 + a1*b11
    #   c2 = a0*b02 + a1*b12
    #   c3 = a0*b03 + a1*b13
    #   Rank 4: each output component uses one multiplication
    #   U_inner = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (4x4 identity for A side)
    #   V_inner = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (4x4 identity for B side)
    #   W_inner = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (4x4 identity for C side)
    
    total_rank = 28
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each Strassen term l (0..6) and each inner component t (0..3):
    # r = l * 4 + t
    
    for l in range(7):
        for t in range(4):
            r = l * 4 + t
            
            # A contribution: for each outer index ia in {0,1,2,3} (Strassen A indices)
            # Strassen A index ia corresponds to A block:
            #   ia=0: A00 = A[0, 0:2]
            #   ia=1: A01 = A[0, 2:4]
            #   ia=2: A10 = A[1, 0:2]
            #   ia=3: A11 = A[1, 2:4]
            # Inner component t selects column t within the 1x2 block
            # A flat index: i*4 + j where i is row, j is col
            #   ia=0: i=0, j=t       -> flat: 0*4 + t = t
            #   ia=1: i=0, j=2+t     -> flat: 0*4 + (2+t) = 2+t
            #   ia=2: i=1, j=t       -> flat: 1*4 + t = 4+t
            #   ia=3: i=1, j=2+t     -> flat: 1*4 + (2+t) = 6+t
            
            a_flat_indices = [t, 2+t, 4+t, 6+t]  # for ia=0,1,2,3
            
            for ia in range(4):
                val = strassen_U[ia, l]
                if val != 0:
                    U[a_flat_indices[ia], r] += val
            
            # B contribution: for each outer index ib in {0,1,2,3} (Strassen B indices)
            # Strassen B index ib corresponds to B block:
            #   ib=0: B00 = B[0:2, 0:4]
            #   ib=1: B01 = B[0:2, 4:8]
            #   ib=2: B10 = B[2:4, 0:4]
            #   ib=3: B11 = B[2:4, 4:8]
            # Inner component t selects column t within the 2x4 block
            # B flat index: j*8 + k where j is row, k is col
            #   ib=0: j in {0,1}, k=t        -> flat: j*8 + t for j=0,1
            #   ib=1: j in {0,1}, k=4+t      -> flat: j*8 + (4+t) for j=0,1
            #   ib=2: j in {2,3}, k=t        -> flat: j*8 + t for j=2,3
            #   ib=3: j in {2,3}, k=4+t      -> flat: j*8 + (4+t) for j=2,3
            
            # For inner component t, we need to multiply by B[j, col]
            # The ring mult (1x2)*(2x4) -> (1x4):
            #   output[t] = sum_{jr=0}^{1} A_block[jr] * B_block[jr, t]
            # So V should pick B[jr, t] for the appropriate block
            # V flat index for B[j, k] = j*8 + k
            
            # For ib=0 (B00 = B[0:2, 0:4]): rows 0,1, col t
            #   V[0*8+t, r] += val, V[1*8+t, r] += val
            # For ib=1 (B01 = B[0:2, 4:8]): rows 0,1, col 4+t
            #   V[0*8+(4+t), r] += val, V[1*8+(4+t), r] += val
            # For ib=2 (B10 = B[2:4, 0:4]): rows 2,3, col t
            #   V[2*8+t, r] += val, V[3*8+t, r] += val
            # For ib=3 (B11 = B[2:4, 4:8]): rows 2,3, col 4+t
            #   V[2*8+(4+t), r] += val, V[3*8+(4+t), r] += val
            
            b_block_rows = [(0,1), (0,1), (2,3), (2,3)]  # row ranges for ib=0,1,2,3
            b_block_col_offset = [0, 4, 0, 4]  # column offset for ib=0,1,2,3
            
            for ib in range(4):
                val = strassen_V[ib, l]
                if val != 0:
                    col = b_block_col_offset[ib] + t
                    for jr in b_block_rows[ib]:
                        V[jr * 8 + col, r] += val
            
            # C contribution: for each outer index ic in {0,1,2,3} (Strassen C indices)
            # Strassen C index ic corresponds to C block:
            #   ic=0: C00 = C[0, 0:4]
            #   ic=1: C01 = C[0, 4:8]
            #   ic=2: C10 = C[1, 0:4]
            #   ic=3: C11 = C[1, 4:8]
            # Inner component t selects column t within the 1x4 block
            # C flat index (W convention): k*n + i = k*2 + i
            #   ic=0: i=0, k=t       -> flat: t*2 + 0 = 2*t
            #   ic=1: i=0, k=4+t     -> flat: (4+t)*2 + 0 = 8+2*t
            #   ic=2: i=1, k=t       -> flat: t*2 + 1 = 2*t+1
            #   ic=3: i=1, k=4+t     -> flat: (4+t)*2 + 1 = 9+2*t
            
            c_flat_indices = [2*t, 8+2*t, 2*t+1, 9+2*t]  # for ic=0,1,2,3
            
            for ic in range(4):
                val = strassen_W[ic, l]
                if val != 0:
                    W[c_flat_indices[ic], r] += val
    
    # Verify the full decomposition
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand
    
    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        
        a_form = np.dot(u_vec, A_rand.flatten())
        b_form = np.dot(v_vec, B_rand.flatten())
        prod = a_form * b_form
        
        C_flat = w_vec * prod
        C_computed += C_flat.reshape((p, n)).T  # W is indexed as [k*n+i], so reshape to (p,n) then transpose
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: naive rank-64
        total_rank = n*m*p
        U = np.zeros((n*m, total_rank), dtype=int)
        V = np.zeros((m*p, total_rank), dtype=int)
        W = np.zeros((p*n, total_rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
