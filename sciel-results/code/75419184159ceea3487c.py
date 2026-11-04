"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition for the full problem
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    
    # Verification function for the full tensor
    def verify_full(U, V, W, n, m, p, tol=1e-5):
        rank = U.shape[1]
        # Check all entries efficiently using matrix multiplication properties if possible,
        # but for small dims, direct check is safer.
        # T[a, b, c] = sum_c' U[a, c'] V[b, c'] W[c, c']
        # We check a sample of valid and invalid entries.
        # For exactness, we should check all.
        # n=4, m=5, p=5. 
        # a in 20, b in 25, c in 20.
        # Total entries 10000. Rank ~100.
        # 10000 * 100 = 1e6 ops. Fast enough.
        
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                # Compute dot product
                                # val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                                val = 0.0
                                for cc in range(rank):
                                    val += U[a, cc] * V[b, cc] * W[c, cc]
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    # Strategy: Try to find a rank-49 decomposition for 2x5x5.
    # If successful, stack two to get rank-98 for 4x5x5.
    
    def find_rank_49_2x5x5():
        n2, m2, p2 = 2, 5, 5
        target_rank = 49
        sb_rank = n2 * m2 * p2 # 50
        
        # Schoolbook for 2x5x5
        U_sb2 = np.zeros((n2*m2, sb_rank))
        V_sb2 = np.zeros((m2*p2, sb_rank))
        W_sb2 = np.zeros((p2*n2, sb_rank))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U_sb2[i*m2+j, col] = 1
                    V_sb2[j*p2+k, col] = 1
                    W_sb2[k*n2+i, col] = 1
                    col += 1
        
        # Remove one column (index 0)
        U_init = U_sb2[:, 1:].copy()
        V_init = V_sb2[:, 1:].copy()
        W_init = W_sb2[:, 1:].copy()
        
        # Flatten parameters
        def params_to_tensors(params):
            U = params[:n2*m2*target_rank].reshape(n2*m2, target_rank)
            V = params[n2*m2*target_rank:(n2*m2+m2*p2)*target_rank].reshape(m2*p2, target_rank)
            W = params[(n2*m2+m2*p2)*target_rank:].reshape(p2*n2, target_rank)
            return U, V, W

        def residuals(params):
            U, V, W = params_to_tensors(params)
            res = []
            for i in range(n2):
                for j in range(m2):
                    a = i*m2+j
                    for j2 in range(m2):
                        for k in range(p2):
                            b = j2*p2+k
                            for k2 in range(p2):
                                for i2 in range(n2):
                                    c = k2*n2+i2
                                    val = 0.0
                                    for cc in range(target_rank):
                                        val += U[a, cc] * V[b, cc] * W[c, cc]
                                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                    res.append(val - expected)
            return np.array(res)

        # Initial params
        p0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        try:
            if time.time() > deadline - 30:
                return None
            # Use bounded least squares? No bounds, just least squares.
            # This might be slow. Let's limit iterations.
            result = least_squares(residuals, p0, method='lm', max_nfev=500)
            if result.success or result.cost < 1e-6:
                U_opt, V_opt, W_opt = params_to_tensors(result.x)
                # Verify this 2x5x5 decomposition
                # Simple check: error norm
                err = np.linalg.norm(result.fun)
                if err < 1e-4:
                    return U_opt, V_opt, W_opt
        except Exception as e:
            pass
        return None

    U2, V2, W2 = find_rank_49_2x5x5()
    
    if U2 is not None:
        # Construct 4x5x5 from two 2x5x5 blocks
        # A is 4x5, B is 5x5.
        # C = A B.
        # Split A into A1 (2x5) and A2 (2x5).
        # C1 = A1 B, C2 = A2 B.
        # C = [C1; C2].
        
        n, m, p = 4, 5, 5
        rank = U2.shape[1] * 2 # 98
        
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        
        # Block 1: rows 0-1 of A, output rows 0-1 of C
        # U1 maps A1 indices (i in 0..1, j in 0..4) to rank
        # V1 maps B indices (j in 0..4, k in 0..4) to rank
        # W1 maps C1 indices (k in 0..4, i in 0..1) to rank
        
        for c in range(U2.shape[1]):
            # U2 rows: i*m2+j where i in 0..1, j in 0..4
            # U_full rows: i*m+j where i in 0..1, j in 0..4
            for i in range(2):
                for j in range(5):
                    U_full[i*5+j, c] = U2[i*5+j, c]
            
            # V2 rows: j*p2+k where j in 0..4, k in 0..4
            # V_full rows: j*p+k where j in 0..4, k in 0..4
            for j in range(5):
                for k in range(5):
                    V_full[j*5+k, c] = V2[j*5+k, c]
            
            # W2 rows: k*n2+i where k in 0..4, i in 0..1
            # W_full rows: k*n+i where k in 0..4, i in 0..1
            for k in range(5):
                for i in range(2):
                    W_full[k*4+i, c] = W2[k*2+i, c]
        
        # Block 2: rows 2-3 of A, output rows 2-3 of C
        for c in range(U2.shape[1]):
            c2 = c + U2.shape[1]
            for i in range(2):
                for j in range(5):
                    U_full[(i+2)*5+j, c2] = U2[i*5+j, c]
            
            for j in range(5):
                for k in range(5):
                # V is shared? No, V depends on B.
                # Wait, the decomposition is for A*B.
                # The V matrix corresponds to B.
                # In the tensor decomposition T[A, B, C], V is associated with B.
                # Since B is the same for both blocks, can we share V?
                # No, the rank-1 terms are independent.
                # The decomposition is sum_c U_c(A) V_c(B) W_c(C).
                # For block 1: sum_{c1} U1_c1(A1) V1_c1(B) W1_c1(C1) = C1
                # For block 2: sum_{c2} U2_c2(A2) V2_c2(B) W2_c2(C2) = C2
                # We can just use the same V structure for both if we duplicate the columns.
                # The V matrix in the full decomposition will have 98 columns.
                # Columns 0-48 correspond to block 1, 49-97 to block 2.
                # V_full[:, 0:49] = V2
                # V_full[:, 49:98] = V2
                # This is valid because the terms are additive.
                    V_full[j*5+k, c2] = V2[j*5+k, c]
            
            for k in range(5):
                for i in range(2):
                    W_full[k*4+(i+2), c2] = W2[k*2+i, c]
        
        # Verify the full decomposition
        if verify_full(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
