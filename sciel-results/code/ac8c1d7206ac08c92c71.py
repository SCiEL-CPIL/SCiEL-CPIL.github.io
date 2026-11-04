"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def verify(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_block_2x2():
        # Returns U, V, W for 2x2 * 2x2 -> 2x2 multiplication
        # Rank 7
        # Indices: U (4,7), V (4,7), W (4,7)
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen constants
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # Mapping to our tensor indices
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = A B
        
        # U[i,j] corresponds to A[i,j]
        # V[j,k] corresponds to B[j,k]
        # W[k,i] corresponds to C[k,i] -> Note: W index is k*n+i. For 2x2, W[k,i] is C[i,k] transposed?
        # Convention: W[k*n+i, c]. For 2x2, W[k*2+i, c].
        # The term is U[i,j] V[j,k] W[k,i].
        # So W[k,i] contributes to C[i,k].
        
        # Let's define the 7 terms explicitly.
        # Term 1: (a11+a22)(b11+b22)
        # U: a11=1, a22=1 -> U[0,0]=1, U[3,0]=1
        # V: b11=1, b22=1 -> V[0,0]=1, V[3,0]=1
        # W: contributes to C. 
        # C11 += m1, C12 += m1, C21 += m1, C22 += m1? No.
        # C11 = a11b11 + a12b21
        # C12 = a11b12 + a12b22
        # C21 = a21b11 + a22b21
        # C22 = a21b12 + a22b22
        
        # m1 = (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22
        # We need to combine m1..m7 to get C.
        # C11 = m1 - m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 - m3 + m6
        
        # Let's assign columns 0..6 to m1..m7.
        
        # m1:
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # W for m1: 
        # C11 coeff +1, C12 coeff 0? No, m1 appears in C11 and C22.
        # Wait, the formula is:
        # C11 = m1 - m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 - m3 + m6
        
        # So for m1 (col 0):
        # W[k,i] adds to C[i,k].
        # C[0,0] (i=0,k=0) gets +1 from m1. W[0*2+0, 0] = 1.
        # C[1,1] (i=1,k=1) gets +1 from m1. W[1*2+1, 0] = 1.
        W[0, 0] = 1
        W[3, 0] = 1
        
        # m2: (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 (i=1,k=0) gets +1. W[0*2+1, 1] = 1.
        # C22 (i=1,k=1) gets -1. W[1*2+1, 1] = -1.
        W[1, 1] = 1
        W[3, 1] = -1
        
        # m3: a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C12 (i=0,k=1) gets +1. W[1*2+0, 2] = 1.
        # C22 (i=1,k=1) gets -1. W[1*2+1, 2] = -1.
        W[2, 2] = 1
        W[3, 2] = -1
        
        # m4: a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C11 (i=0,k=0) gets -1. W[0*2+0, 3] = -1.
        # C21 (i=1,k=0) gets +1. W[0*2+1, 3] = 1.
        W[0, 3] = -1
        W[1, 3] = 1
        
        # m5: (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 (i=0,k=0) gets -1. W[0*2+0, 4] = -1.
        # C12 (i=0,k=1) gets +1. W[1*2+0, 4] = 1.
        W[0, 4] = -1
        W[2, 4] = 1
        
        # m6: (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 (i=1,k=1) gets +1. W[1*2+1, 5] = 1.
        W[3, 5] = 1
        
        # m7: (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 (i=0,k=0) gets +1. W[0*2+0, 6] = 1.
        W[0, 6] = 1
        
        return U, V, W

    def build_strassen_decomposition():
        # We need 3x4 * 4x8.
        # We can decompose the 4x4 middle part into two 2x2 blocks?
        # A is 3x4. B is 4x8.
        # Split A into A1 (3x2) and A2 (3x2).
        # Split B into B1 (2x8) and B2 (2x8).
        # C = A1 B1 + A2 B2.
        # This doesn't directly use Strassen's 2x2 multiplication because the outer dims are 3 and 8.
        
        # Alternative: Pad A to 4x4 and B to 4x4? No, B is 4x8.
        # Let's use the fact that 4x8 can be seen as two 4x4 multiplications?
        # B = [B_left, B_right] where B_left is 4x4, B_right is 4x4.
        # C = A * [B_left, B_right] = [A B_left, A B_right].
        # So we need to multiply A (3x4) by B_left (4x4) and A (3x4) by B_right (4x4).
        
        # For A (3x4) * B (4x4):
        # Pad A to 4x4 by adding a row of zeros? A_padded = [[A], [0 0 0 0]].
        # Then A_padded (4x4) * B (4x4) -> C_padded (4x4).
        # The top 3 rows of C_padded are the result.
        
        # So we can use Strassen for 4x4 * 4x4?
        # Strassen is for 2x2. We can recursively apply it.
        # Rank of 4x4 * 4x4 using Strassen is 7^2 = 49.
        # We have two such multiplications (for B_left and B_right).
        # Total rank = 49 + 49 = 98. This is worse than schoolbook (96).
        
        # Let's try to share terms?
        # Or just use the schoolbook rank 96 as the baseline, and try to reduce it.
        
        # Let's generate a valid rank 96 solution first.
        U, V, W, rank = schoolbook()
        return U, V, W, rank

    # Strategy: Start with Schoolbook (96), then try to reduce rank using ALS + Greedy Drop.
    
    def als_optimize(U, V, W, n, m, p, rank, max_iters=100):
        """
        Alternating Least Squares to minimize || T - sum U_c V_c W_c ||_F
        """
        # Precompute target tensor T (n, m, p)
        T = np.ones((n, m, p))
        
        # Reshape U, V, W to 3D
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Fix V, W. Optimize U.
            # For each (i, j), U[i, j, :] is a vector of length R.
            # We want sum_c U[i,j,c] V[j,:,c] W[:,i,c] to match T[i,j,:].
            # This is a linear system: A_U[i,j] * u_vec = t_vec
            # Where A_U[i,j] is (p, R) matrix with A_U[i,j][k, c] = V[j, k, c] * W[k, i, c]
            
            U_new = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    # Build A matrix
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j, k, c] * W[k, i, c]
                    # Solve A u = T[i, j, :]
                    try:
                        u_vec, residuals, rank_A, sv = np.linalg.lstsq(A, T[i, j, :], rcond=None)
                    except:
                        u_vec = U[i, j, :]
                    U_new[i, j, :] = u_vec
            U = U_new
            
            # Fix U, W. Optimize V.
            V_new = np.zeros((m, p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i, j, c] * W[k, i, c]
                    try:
                        v_vec, residuals, rank_A, sv = np.linalg.lstsq(A, T[:, j, k], rcond=None)
                    except:
                        v_vec = V[j, k, :]
                    V_new[j, k, :] = v_vec
            V = V_new
            
            # Fix U, V. Optimize W.
            W_new = np.zeros((p, n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(range(rank)):
                            A[j, c] = U[i, j, c] * V[j, k, c]
                    try:
                        w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, T[i, j, k], rcond=None)
                    except:
                        w_vec = W[k, i, :]
                    W_new[k, i, :] = w_vec
            W = W_new
            
        return U.reshape(n*m, rank), V.reshape(m*p, rank), W.reshape(p*n, rank)

    def calculate_residual(U, V, W, n, m, p, rank):
        T = np.ones((n, m, p))
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T_calc = np.zeros((n, m, p))
        for c in range(rank):
            T_calc += np.einsum('ij,jk,ki->ijk', U_3d[:,:,c], V_3d[:,:,c], W_3d[:,:,c])
        return np.linalg.norm(T - T_calc)

    # Main Search Loop
    # Start with Rank 90, 85, 80...
    # For each rank, run ALS for a few iterations, then try to drop terms.
    
    best_solution = None
    best_rank = 96
    U_sb, V_sb, W_sb, _ = schoolbook()
    best_solution = (U_sb, V_sb, W_sb, 96)
    
    # Try ranks from 80 to 95
    for R in range(80, 96):
        if time.time() > deadline - 20:
            break
            
        # Initialize with random values, but scaled
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Run ALS
        U, V, W = als_optimize(U, V, W, n, m, p, R, max_iters=50)
        
        res = calculate_residual(U, V, W, n, m, p, R)
        
        if res < 1e-6:
            # Try to drop terms
            current_rank = R
            U_curr, V_curr, W_curr = U, V, W
            dropped = True
            while dropped and current_rank > 50:
                dropped = False
                for c in range(current_rank):
                    if time.time() > deadline - 10:
                        break
                    # Try to set column c to 0
                    U_test = U_curr.copy()
                    V_test = V_curr.copy()
                    W_test = W_curr.copy()
                    U_test[:, c] = 0
                    V_test[:, c] = 0
                    W_test[:, c] = 0
                    
                    # Re-optimize with this term fixed to 0
                    # We can just run ALS on the remaining terms
                    # But ALS optimizes all terms. We need to constrain c to 0.
                    # Easier: Remove column c, run ALS on rank-1, then pad back.
                    
                    if c == current_rank - 1:
                        U_sub = U_curr[:, :-1]
                        V_sub = V_curr[:, :-1]
                        W_sub = W_curr[:, :-1]
                    else:
                        U_sub = np.delete(U_curr, c, axis=1)
                        V_sub = np.delete(V_curr, c, axis=1)
                        W_sub = np.delete(W_curr, c, axis=1)
                        
                    U_new, V_new, W_new = als_optimize(U_sub, V_sub, W_sub, n, m, p, current_rank-1, max_iters=20)
                    res_new = calculate_residual(U_new, V_new, W_new, n, m, p, current_rank-1)
                    
                    if res_new < 1e-6:
                        # Success
                        U_curr, V_curr, W_curr = U_new, V_new, W_new
                        current_rank -= 1
                        dropped = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_solution = (U_curr, V_curr, W_curr, current_rank)
                            print(f"Dropped term, new rank {current_rank}")
                        break # Restart drop loop with new rank
                        
        if best_solution[3] < R:
            # We already found a better rank, maybe skip lower R?
            # No, we are iterating R upwards. If we found 80, we don't need to try 81?
            # Actually, if we found 80, we should try to go lower.
            # But the loop goes 80->95.
            # If we find 80, we should stop and try to go lower?
            # Let's just keep the best.
            pass

    if best_solution:
        U, V, W, rank = best_solution
        if verify(U, V, W, n, m, p, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    if verify(U, V, W, n, m, p, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
