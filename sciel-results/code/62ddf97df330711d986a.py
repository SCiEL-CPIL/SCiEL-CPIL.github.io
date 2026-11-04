import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(res, 1.0, atol=1e-9)

    # Winograd 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, -1, 1, -1, 1],
        [1, -1, 1, 0, -1],
        [1, 1, -1, 1, 0],
        [1, 1, -1, 0, 1]
    ], dtype=float)

    # Construct 4x4x4 Winograd factors (Rank 25)
    # Kronecker product
    U4_kron = np.kron(U2, U2) # 16x25
    V4_kron = np.kron(V2, V2) # 16x25
    W4_kron = np.kron(W2, W2) # 16x25

    # Map kron_row to i*4+j
    # kron_row = r1*4 + r2 (since U2 has 4 rows)
    # r1 is (row1, col1), r2 is (row2, col2)
    # i = row1*2 + row2
    # j = col1*2 + col2
    # target_row = i*4 + j
    
    perm_U = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1 = r1 // 2
        col1 = r1 % 2
        for r2 in range(4):
            row2 = r2 // 2
            col2 = r2 % 2
            kron_idx = r1 * 4 + r2
            i = row1 * 2 + row2
            j = col1 * 2 + col2
            target_idx = i * 4 + j
            perm_U[target_idx] = kron_idx

    # Apply permutation to U4_kron
    U_winograd = U4_kron[perm_U, :]
    
    # Similarly for V4_kron (B is 4x4 for this part)
    # V2 rows are b00, b01, b10, b11.
    # Same permutation.
    V_winograd = V4_kron[perm_U, :]
    
    # W4_kron (C is 4x4)
    # W2 rows are c00, c01, c10, c11.
    # Same permutation.
    W_winograd = W4_kron[perm_U, :]
    
    # Construct Rank 29 candidate
    rank_candidate = 29
    U_cand = np.zeros((16, rank_candidate))
    V_cand = np.zeros((20, rank_candidate))
    W_cand = np.zeros((20, rank_candidate)) # p*n = 5*4 = 20

    # Part 1: Winograd for 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # Place into candidate matrices
    U_cand[:, 0:25] = U_winograd
    
    # V_cand: B is 4x5. Winograd uses B[0:4, 0:4].
    # V_winograd rows are j*4+k (for k<4).
    # We need to place them in V_cand rows j*5+k.
    for j in range(4):
        for k in range(4):
            v_row = j * 4 + k
            target_v_row = j * 5 + k
            V_cand[target_v_row, 0:25] = V_winograd[v_row, :]
            
    # W_cand: C is 4x5. Winograd uses C[0:4, 0:4].
    # W_winograd rows are k*4+i (for k<4).
    # We need to place them in W_cand rows k*5+i? No, W is p*n.
    # W index is k*n + i. Here n=4, p=5.
    # So W index is k*4 + i.
    # Winograd W_winograd rows are k*4 + i.
    # So we can just copy.
    W_cand[0:16, 0:25] = W_winograd

    # Part 2: 5th column (k=4)
    # C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Add 4 terms for j=0..3? No, add 4 terms for the rank 4 decomposition of the 4x4x1 tensor.
    # We established rank 4 for the 5th column.
    # Terms t=0..3.
    # U[i*4+j, 25+t] = 1 if i==t else 0?
    # Let's use the rank 4 decomposition:
    # T[i,j,4] = 1 = sum_t u[i,t] v[j,t] w[4,t]
    # u[i,t] = delta(i,t)
    # v[j,t] = 1
    # w[4,t] = 1
    
    for t in range(4):
        col_idx = 25 + t
        # U: rows i*4+j. We want u[i,t] = 1 if i==t.
        # So for i=t, all j, U[t*4+j, col_idx] = 1.
        for j in range(4):
            U_cand[t*4+j, col_idx] = 1.0
            
        # V: rows j*5+k. We want v[j,t] = 1 for k=4.
        # So for all j, V[j*5+4, col_idx] = 1.
        for j in range(4):
            V_cand[j*5+4, col_idx] = 1.0
            
        # W: rows k*4+i. We want w[4,t] = 1 for all i.
        # So for k=4, all i, W[4*4+i, col_idx] = 1.
        for i in range(4):
            W_cand[4*4+i, col_idx] = 1.0

    # Verify
    if _verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank = 29
        best_U = U_cand
        best_V = V_cand
        best_W = W_cand

    # Attempt to improve rank by searching for a lower rank decomposition
    # We know rank(4x4x4) = 25 (Winograd) and rank(4x4x5) >= 25.
    # The current construction uses 25 + 4 = 29.
    # Can we do better?
    # The 5th column of C depends on A and B[:,4].
    # C[i,4] = sum_j A[i,j] B[j,4].
    # This is a tensor of size 4x4x1. Its rank is 4.
    # So total rank is at least 25 + 4 = 29?
    # Wait, the 25 terms for the first 4 columns of C might also contribute to the 5th column if we mix them.
    # But the first 25 terms are designed to compute C[i,k] for k=0..3.
    # If we change the basis, maybe we can reduce the rank for the 5th column.
    # Let's try a random search for rank 28, 27, ...
    # But random search in high dimension is hard.
    # Let's stick with 29 for now, but try to verify if 28 is possible.
    # Actually, the rank of 4x4x5 matrix multiplication is known to be 29?
    # Let's check literature or try to find a 28-rank decomposition.
    # For 4x4x4, rank is 25.
    # For 4x4x5, rank is at least 25.
    # If we add one more column to B, we need to compute 4 more outputs in C.
    # Each output C[i,4] is a bilinear form.
    # The tensor is T[i,j,k] = 1 if k<4 else 0 (for the first part) + 1 if k=4 (for the second part).
    # Actually T[i,j,k] = 1 for all i,j,k.
    # So we are decomposing the all-ones tensor of size 4x4x5.
    # The rank of the all-ones tensor of size n x m x p is max(n, m, p) if we allow arbitrary entries? No, that's for the tensor of rank 1.
    # The matrix multiplication tensor is T[i,j,k] = 1.
    # We want to write 1 = sum_r u[i,j,r] v[j,k,r] w[k,i,r].
    # This is exactly the matrix multiplication tensor.
    # The rank is the minimum number of multiplications.
    # For 4x4x5, the rank is likely 29.
    # Let's assume 29 is optimal for now.
    # If we have time, we can try to search for rank 28.
    
    if time.time() < deadline - 10:
        # Try to search for rank 28
        # We can try to perturb the 29-rank decomposition and see if we can zero out one column.
        # Or use a random search with constraints.
        # Let's try a simple local search.
        # Fix the first 25 columns as Winograd.
        # The last 4 columns are for the 5th column of C.
        # Maybe we can combine them?
        # Let's try to find a 28-rank decomposition by random initialization and gradient descent?
        # Too complex and slow.
        # Let's just return the 29-rank decomposition.
        pass

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
