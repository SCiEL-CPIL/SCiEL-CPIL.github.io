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
    
    # Check for linearly dependent rank terms and reduce rank
    # Build the 3-way tensor T[i,j,k] = delta (matrix mult tensor)
    # and check if our 56 terms span a subspace of dimension < 56
    # We can check by forming the matrix of rank-1 tensors and computing its rank
    
    # Flatten each rank-1 term into a vector of size n*m*p = 64
    # Term r has U[:,r] (8), V[:,r] (32), W[:,r] (16)
    # The tensor entry T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For the matrix mult tensor: T[i,j,k] = 1 if i,j,k correspond to valid (i,j,k) in A*B=C
    
    # Build tensor matrix: for each rank term r, create vector of size 64
    # indexed by (i,j,k) where i in [0,2), j in [0,4), k in [0,8)
    # T[i,j,k] = U[i,r] * V[j,r] * W[k*n+i, r] ... wait, W is indexed by (k,i)
    
    # Actually the tensor for matrix mult: M[i,j,k] = 1 where C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k) in [0,2)x[0,4)x[0,8)
    # That's just the all-ones tensor of shape (2,4,8)
    
    # Each rank-1 term contributes U_r ⊗ V_r ⊗ W_r where:
    # U_r[i] = U[i,r], V_r[j] = V[j,r], W_r[k] = W[k*n+i,r]... 
    # Wait, W depends on i too. This is not a simple Kronecker product.
    
    # The tensor decomposition is: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k*n+i, r]
    # This is NOT a standard 3-tensor because W depends on i.
    # So we can't simply check linear dependence of Kronecker products.
    
    # Instead, verify correctness and check if we can find a lower rank by
    # trying to eliminate terms that are linear combinations of others.
    
    # For now, verify the decomposition is correct
    np.random.seed(42)
    valid = True
    for trial in range(10):
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
            
            # W is indexed as [k*n+i], so w_vec[k*n+i] contributes to C[i,k]
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += w_vec[k*n+i] * prod
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            valid = False
            break
    
    if not valid:
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
    
    # Try to reduce rank by checking if any column of the "coefficient matrix" is redundant
    # Build a matrix where each column r represents the rank-1 term's contribution pattern
    # Since the tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k*n+i,r],
    # we can form a matrix M of size (n*m*p) x rank where:
    # M[(i*m+j)*p+k, r] = U[i,r] * V[j,r] * W[k*n+i, r]
    # If this matrix has rank < 56, we can reduce.
    
    M = np.zeros((n*m*p, total_rank), dtype=float)
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[(i*m+j)*p+k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < total_rank:
        # Find a minimal set of columns that spans the column space
        # Use QR with column pivoting
        Q, R = np.linalg.qr(M.T, mode='reduced')
        # R is (rank, total_rank), find pivot columns
        diag_R = np.abs(np.diag(R))
        # Select columns where diag is significant
        threshold = max(diag_R) * 1e-8
        pivot_cols = [idx for idx in range(total_rank) if diag_R[idx] > threshold]
        
        # Verify the reduced set still works
        reduced_rank = len(pivot_cols)
        U_red = U[:, pivot_cols]
        V_red = V[:, pivot_cols]
        W_red = W[:, pivot_cols]
        
        np.random.seed(123)
        valid_red = True
        for trial in range(10):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_computed = np.zeros((n, p))
            for r in range(reduced_rank):
                u_vec = U_red[:, r]
                v_vec = V_red[:, r]
                w_vec = W_red[:, r]
                
                a_form = np.dot(u_vec, A_rand.flatten())
                b_form = np.dot(v_vec, B_rand.flatten())
                prod = a_form * b_form
                
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += w_vec[k*n+i] * prod
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                valid_red = False
                break
        
        if valid_red:
            U = U_red
            V = V_red
            W = W_red
            total_rank = reduced_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
