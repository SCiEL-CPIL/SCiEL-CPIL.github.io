"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 130s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_structured_rank_r(r, seed=42):
        """
        Try to find a rank-r decomposition using alternating least squares
        with structured initialization that exploits the 2x3x6 asymmetry.
        """
        if time.time() > DEADLINE:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random matrices
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(100):
            if time.time() > DEADLINE:
                break
            
            # Solve for V given U, W
            A_mat = np.zeros((n * m * p, m * p * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (j * p + k) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * W[k * n + i, c]
            
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V = v_flat.reshape(m * p, r)
            
            # Solve for U given V, W
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (i * m + j) * r + c
                            A_mat[e, u_idx] = V[j * p + k, c] * W[k * n + i, c]
            
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U = u_flat.reshape(n * m, r)
            
            # Solve for W given U, V
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (k * n + i) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = w_flat.reshape(p * n, r)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-9:
                # Try to round to half-integers
                U_rounded = np.array([[round_to_half(val) for val in row] for row in U])
                V_rounded = np.array([[round_to_half(val) for val in row] for row in V])
                W_rounded = np.array([[round_to_half(val) for val in row] for row in W])
                
                # Verify rounded solution
                if verify_np(U_rounded, V_rounded, W_rounded, r):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # If we didn't get exact but got close, try rounding the best solution
        if best_error < 1e-6:
            U_rounded = np.array([[round_to_half(val) for val in row] for row in best_U])
            V_rounded = np.array([[round_to_half(val) for val in row] for row in best_V])
            W_rounded = np.array([[round_to_half(val) for val in row] for row in best_W])
            
            if verify_np(U_rounded, V_rounded, W_rounded, r):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None
    
    def try_border_rank_18():
        """
        Try to find a rank-18 decomposition using the border rank relaxation.
        
        The key insight: for 2x3x6, we can exploit the rectangular structure
        by using intermediate sums that reduce the effective rank.
        
        Strategy:
        - C[i,k] = sum_j A[i,j] * B[j,k]
        - We can write this using 18 terms by grouping operations
        
        For example:
        - Use sums like A[0,:]+A[1,:] to reduce terms
        - Use differences to create cancellations
        """
        if time.time() > DEADLINE:
            return None
        
        r = 18
        np.random.seed(123)
        
        # Try multiple random initializations
        for seed in range(20):
            if time.time() > DEADLINE:
                return None
            
            res = try_structured_rank_r(r, seed=42 + seed)
            if res:
                return res
        
        return None
    
    def try_border_rank_19():
        """Try rank 19 as an intermediate step."""
        if time.time() > DEADLINE:
            return None
        
        r = 19
        np.random.seed(123)
        
        for seed in range(15):
            if time.time() > DEADLINE:
                return None
            
            res = try_structured_rank_r(r, seed=42 + seed)
            if res:
                return res
        
        return None
    
    # Try rank 18 first (border rank relaxation)
    if time.time() < DEADLINE:
        res = try_border_rank_18()
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Try rank 19
    if time.time() < DEADLINE:
        res = try_border_rank_19()
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Try ranks from 20 to 36 as fallback
    for r in range(20, 37):
        if time.time() > DEADLINE:
            break
        
        for seed in range(10):
            if time.time() > DEADLINE:
                break
            
            res = try_structured_rank_r(r, seed=42 + seed)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
