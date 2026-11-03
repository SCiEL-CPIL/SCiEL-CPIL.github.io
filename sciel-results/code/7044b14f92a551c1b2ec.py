import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    # W index: W[k*n + i, r]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Local indices: a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    #                b0=B[0,0], b1=B[0,1], b2=B[1,0], b3=B[1,1]
    #                c0=C[0,0], c1=C[0,1], c2=C[1,0], c3=C[1,1]
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1 = (a0+a3)(b0+b3) -> c0 += M1, c3 += M1
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # M2 = (a2+a3)(b0) -> c2 += M2, c3 -= M2
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # M3 = (a0)(b1-b3) -> c1 += M3, c3 += M3
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    # M4 = (a3)(b2-b0) -> c0 += M4, c2 += M4
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # M5 = (a0+a1)(b3) -> c0 -= M5, c1 += M5
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # M6 = (a2-a0)(b0+b1) -> c3 += M6
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    # M7 = (a1-a3)(b2+b3) -> c0 += M7
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # 2 j-groups × 4 k-blocks = 8 sub-problems, each rank 7
    for group in range(2):      # group 0: j in {0,1}, group 1: j in {2,3}
        for block_col in range(4):  # k-pair: {2*bc, 2*bc+1}
            base_r = (group * 4 + block_col) * 7
            
            # A indices: A[i, 2*group + a_idx] where a_idx maps local 0..3 to (i,j)
            # Local a0=(0, 2*group), a1=(0, 2*group+1), a2=(1, 2*group), a3=(1, 2*group+1)
            a_map = [0]*4
            a_map[0] = 0*4 + (2*group + 0)
            a_map[1] = 0*4 + (2*group + 1)
            a_map[2] = 1*4 + (2*group + 0)
            a_map[3] = 1*4 + (2*group + 1)
            
            # B indices: B[2*group + b_row, 2*block_col + b_col]
            # Local b0=(2*group, 2*bc), b1=(2*group, 2*bc+1), b2=(2*group+1, 2*bc), b3=(2*group+1, 2*bc+1)
            b_map = [0]*4
            b_map[0] = (2*group + 0)*8 + (2*block_col + 0)
            b_map[1] = (2*group + 0)*8 + (2*block_col + 1)
            b_map[2] = (2*group + 1)*8 + (2*block_col + 0)
            b_map[3] = (2*group + 1)*8 + (2*block_col + 1)
            
            # C indices: C[i, 2*block_col + c_col], W index = k*n + i
            # Local c0=(0, 2*bc), c1=(0, 2*bc+1), c2=(1, 2*bc), c3=(1, 2*bc+1)
            c_map = [0]*4
            c_map[0] = (2*block_col + 0)*2 + 0
            c_map[1] = (2*block_col + 1)*2 + 0
            c_map[2] = (2*block_col + 0)*2 + 1
            c_map[3] = (2*block_col + 1)*2 + 1
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    # Verify with multiple random test cases
    rng = np.random.RandomState(42)
    for test_iter in range(3):
        if time.time() > deadline:
            break
        A_test = rng.randint(-5, 6, size=(n, m)).astype(float)
        B_test = rng.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            u_vec = U[:, r].astype(float)
            v_vec = V[:, r].astype(float)
            w_vec = W[:, r].astype(float)
            
            a_form = np.dot(A_test.flatten(), u_vec)
            b_form = np.dot(B_test.flatten(), v_vec)
            prod = a_form * b_form
            
            for idx in range(p * n):
                k, i = divmod(idx, n)
                C_computed[i, k] += w_vec[idx] * prod
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to naive rank-64
            U_naive = np.zeros((n*m, n*m*p), dtype=int)
            V_naive = np.zeros((m*p, n*m*p), dtype=int)
            W_naive = np.zeros((p*n, n*m*p), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_idx = i*m*p + j*p + k
                        U_naive[i*m+j, r_idx] = 1
                        V_naive[j*p+k, r_idx] = 1
                        W_naive[k*n+i, r_idx] = 1
            return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
