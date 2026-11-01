"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find a lower-rank decomposition using block structure
    # Split B into p/m blocks of size m x m? No, p=9, m=3. 3 blocks of 3x3.
    # Each block is a 2x3x3 multiplication.
    # Rank of 2x3x3 is 11.
    # Total rank = 3 * 11 = 33.
    
    target_rank_block = 11
    target_rank_total = target_rank_block * (p // m)
    
    if target_rank_total < rank_schoolbook:
        # Try to solve for 2x3x3 decomposition
        n_b, m_b, p_b = 2, 3, 3
        U_b, V_b, W_b = find_low_rank_decomp(n_b, m_b, p_b, target_rank_block, start_time, deadline)
        
        if U_b is not None:
            # Construct full decomposition
            rank = target_rank_total
            U_full = [[0] * rank for _ in range(n * m)]
            V_full = [[0] * rank for _ in range(m * p)]
            W_full = [[0] * rank for _ in range(p * n)]
            
            for block in range(p // m):
                offset = block * target_rank_block
                # U is shared, so copy U_b to each block's columns
                for i in range(n * m):
                    for c in range(target_rank_block):
                        U_full[i][offset + c] = U_b[i][c]
                        
                # V and W are disjoint for each block
                for j in range(m_b):
                    for k_local in range(p_b):
                        k_global = block * p_b + k_local
                        for c in range(target_rank_block):
                            V_full[j * p + k_global][offset + c] = V_b[j * p_b + k_local][c]
                            W_full[k_global * n + i][offset + c] = W_b[k_local * n_b + i][c] # Note: W index is k*n+i
                
            # Verify the decomposition
            if verify_decomposition(U_full, V_full, W_full, n, m, p):
                best = {"U": U_full, "V": V_full, "W": W_full, "rank": rank}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

def find_low_rank_decomp(n, m, p, rank, start_time, deadline):
    # Build tensor T for n x m x p
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Initialize U, V, W
    rng = np.random.RandomState(42)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # ALS iterations
    for _ in range(50):
        if time.time() > deadline - 10:
            return None
            
        # Update U
        # T = sum_c u_c v_c^T w_c^T
        # For fixed V, W, U is linear.
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # This is not linear in U if we sum over all indices.
        # But we can solve for U given V, W.
        # T = U * (V * W)^T ? No.
        # Let M_c = V[:, c] * W[:, c] (outer product)
        # Then T = sum_c U[:, c] * M_c
        # This is a linear system for U.
        
        # Reshape T to (n*m, m*p*p*n)
        T_flat = T.reshape(n*m, -1)
        
        # Build matrix A where A[:, c] is the vectorized M_c
        A = np.zeros((m*p*p*n, rank))
        for c in range(rank):
            M_c = np.outer(V[:, c], W[:, c])
            A[:, c] = M_c.flatten()
            
        # Solve T_flat = U * A^T
        # U = T_flat * (A^T)^+
        try:
            U_new = np.linalg.lstsq(A.T, T_flat.T, rcond=None)[0].T
        except:
            return None
            
        # Round to nearest integer/half-integer?
        # We'll check if it's close to integers later.
        U = U_new
        
        # Similarly for V and W
        # T = sum_c U_c * V_c * W_c
        # For V: T_flat_V = V * (U * W)^T
        T_flat_V = T.reshape(m*p, -1)
        A_V = np.zeros((n*m*p*n, rank))
        for c in range(rank):
            M_c = np.outer(U[:, c], W[:, c])
            A_V[:, c] = M_c.flatten()
        try:
            V_new = np.linalg.lstsq(A_V.T, T_flat_V.T, rcond=None)[0].T
        except:
            return None
        V = V_new
        
        # For W
        T_flat_W = T.reshape(p*n, -1)
        A_W = np.zeros((n*m*m*p, rank))
        for c in range(rank):
            M_c = np.outer(U[:, c], V[:, c])
            A_W[:, c] = M_c.flatten()
        try:
            W_new = np.linalg.lstsq(A_W.T, T_flat_W.T, rcond=None)[0].T
        except:
            return None
        W = W_new
        
    # Check if U, V, W are close to integers/half-integers
    def is_close_to_rational(X, tol=1e-5):
        for val in X.flatten():
            if abs(val - round(val*2)/2) > tol:
                return False
        return True
        
    if is_close_to_rational(U) and is_close_to_rational(V) and is_close_to_rational(W):
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return U.tolist(), V.tolist(), W.tolist()
        
    return None

def verify_decomposition(U, V, W, n, m, p):
    # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k
    # And 0 for others?
    # The tensor T has 1s at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    # We only need to check the 1s, as the 0s are implied by the structure if the 1s are correct?
    # No, we must check that the reconstruction matches T.
    
    # Reconstruct T
    T_rec = np.zeros((n*m, m*p, p*n))
    for c in range(len(U[0])):
        u_c = np.array(U[:, c])
        v_c = np.array(V[:, c])
        w_c = np.array(W[:, c])
        # Outer product
        # T_rec += u_c * v_c * w_c (broadcasting)
        # u_c is (n*m, 1), v_c is (m*p, 1), w_c is (p*n, 1)
        # This is a 3-tensor.
        T_rec += np.einsum('i,j,k->ijk', u_c, v_c, w_c)
        
    # Target T
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1
                
    return np.allclose(T_rec, T_target, atol=1e-5)
# EVOLVE-BLOCK-END
