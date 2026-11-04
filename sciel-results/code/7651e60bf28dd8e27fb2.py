import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    # W index convention: W[k*n + i, r] where C[i,k] gets contribution
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices for A (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local indices for B (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local indices for C (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
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
    
    # 8 independent 2x2x2 sub-problems:
    # For each pair of middle indices (j, j+1) and pair of output columns (k, k+1)
    # group: which pair of j indices (0: j=0,1; 1: j=2,3)
    # block_col: which pair of k indices (0: k=0,1; 1: k=2,3; 2: k=4,5; 3: k=6,7)
    
    for group in range(2):      # j-pair: {0,1} or {2,3}
        for block_col in range(4):  # k-pair: {0,1}, {2,3}, {4,5}, {6,7}
            base_r = (group * 4 + block_col) * 7
            
            # A indices: A[i, j] where j = 2*group + a_idx_local
            # Local: 0->(i=0,j_local=0), 1->(i=0,j_local=1), 2->(i=1,j_local=0), 3->(i=1,j_local=1)
            a_map = [
                0 * m + (2 * group + 0),   # A[0, 2*group]
                0 * m + (2 * group + 1),   # A[0, 2*group+1]
                1 * m + (2 * group + 0),   # A[1, 2*group]
                1 * m + (2 * group + 1),   # A[1, 2*group+1]
            ]
            
            # B indices: B[j, k] where j = 2*group + b_row_local, k = 2*block_col + b_col_local
            b_map = [
                (2 * group + 0) * p + (2 * block_col + 0),   # B[2g, 2bc]
                (2 * group + 0) * p + (2 * block_col + 1),   # B[2g, 2bc+1]
                (2 * group + 1) * p + (2 * block_col + 0),   # B[2g+1, 2bc]
                (2 * group + 1) * p + (2 * block_col + 1),   # B[2g+1, 2bc+1]
            ]
            
            # C indices: C[i, k] where k = 2*block_col + c_col_local
            # W index: k * n + i
            c_map = [
                (2 * block_col + 0) * n + 0,   # C[0, 2bc]
                (2 * block_col + 1) * n + 0,   # C[0, 2bc+1]
                (2 * block_col + 0) * n + 1,   # C[1, 2bc]
                (2 * block_col + 1) * n + 1,   # C[1, 2bc+1]
            ]
            
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
    
    # Verification with multiple random tests
    np.random.seed(42)
    for trial in range(10):
        A_test = np.random.randint(-5, 6, (n, m))
        B_test = np.random.randint(-5, 6, (m, p))
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(total_rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            a_form = int(np.dot(A_test.flatten(), u_vec))
            b_form = int(np.dot(B_test.flatten(), v_vec))
            prod = a_form * b_form
            
            for idx in range(p * n):
                if w_vec[idx] != 0:
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += int(w_vec[idx]) * prod
        
        if not np.array_equal(C_computed, C_expected):
            # Fallback to naive rank-64
            U_naive = np.zeros((n*m, n*m*p), dtype=int)
            V_naive = np.zeros((m*p, n*m*p), dtype=int)
            W_naive = np.zeros((p*n, n*m*p), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i*m*p + j*p + k
                        U_naive[i*m+j, r] = 1
                        V_naive[j*p+k, r] = 1
                        W_naive[k*n+i, r] = 1
            return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
