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
    
    # Strassen 2x2x2 decomposition
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = [[c0, c1], [c2, c3]]
    # Indexing: A[i*2+j], B[i*2+j], C[i*2+j]
    
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
    
    # Verify Strassen locally
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_test = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed = np.zeros(4)
    for r in range(7):
        prod = np.dot(A_flat, U_local[:,r]) * np.dot(B_flat, V_local[:,r])
        C_computed += W_local[:,r] * prod
    
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
    
    # Build full 2x4 x 4x8 decomposition using 8 Strassen blocks
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # Group j into {0,1} and {2,3}, and k into pairs {0,1}, {2,3}, {4,5}, {6,7}
    # Each group (j-pair, k-pair) is a 2x2 x 2x2 multiplication
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j_group in range(2):  # j_group 0: j in {0,1}, j_group 1: j in {2,3}
        for k_pair in range(4):  # k_pair selects columns {2*k_pair, 2*k_pair+1}
            base_r = (j_group * 4 + k_pair) * 7
            
            # A indices: A[i, 2*j_group + a_idx] for a_idx in {0,1}
            # U index: i*m + j = i*4 + (2*j_group + a_idx)
            a_map = [0]*4
            a_map[0] = 0*4 + (2*j_group + 0)  # A[0, 2*j_group]
            a_map[1] = 0*4 + (2*j_group + 1)  # A[0, 2*j_group+1]
            a_map[2] = 1*4 + (2*j_group + 0)  # A[1, 2*j_group]
            a_map[3] = 1*4 + (2*j_group + 1)  # A[1, 2*j_group+1]
            
            # B indices: B[2*j_group + b_row, 2*k_pair + b_col]
            # V index: j*p + k = (2*j_group + b_row)*8 + (2*k_pair + b_col)
            b_map = [0]*4
            b_map[0] = (2*j_group + 0)*8 + (2*k_pair + 0)
            b_map[1] = (2*j_group + 0)*8 + (2*k_pair + 1)
            b_map[2] = (2*j_group + 1)*8 + (2*k_pair + 0)
            b_map[3] = (2*j_group + 1)*8 + (2*k_pair + 1)
            
            # C indices: C[i, 2*k_pair + c_col]
            # W index: k*n + i = (2*k_pair + c_col)*2 + i
            c_map = [0]*4
            c_map[0] = (2*k_pair + 0)*2 + 0  # C[0, 2*k_pair]
            c_map[1] = (2*k_pair + 1)*2 + 0  # C[0, 2*k_pair+1]
            c_map[2] = (2*k_pair + 0)*2 + 1  # C[1, 2*k_pair]
            c_map[3] = (2*k_pair + 1)*2 + 1  # C[1, 2*k_pair+1]
            
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
