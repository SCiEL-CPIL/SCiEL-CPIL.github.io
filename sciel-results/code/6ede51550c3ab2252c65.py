"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 140s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    def round_to_int(x):
        return round(x)
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def build_A_for_V(U, W, r):
        """Build the linear system matrix for solving V."""
        # A[e, c] = U[i*m+j, c] * W[k*n+i, c] where e = (i*m+j)*p+k
        A = np.zeros((n * m * p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, c] = U[i * m + j, c] * W[k * n + i, c]
        return A
    
    def build_A_for_U(V, W, r):
        """Build the linear system matrix for solving U."""
        A = np.zeros((n * m * p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, c] = V[j * p + k, c] * W[k * n + i, c]
        return A
    
    def build_A_for_W(U, V, r):
        """Build the linear system matrix for solving W."""
        A = np.zeros((n * m * p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    for c in range(r):
                        A[e, c] = U[i * m + j, c] * V[j * p + k, c]
        return A
    
    def compute_error(U, V, W, r):
        """Compute the squared error of the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(n):
                pass
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (val - 1.0) ** 2
        return error
    
    def try_rounding(U, V, W, r, strategies=None):
        """Try multiple rounding strategies and return best valid result."""
        if strategies is None:
            strategies = ['half', 'int', 'quarter', 'eighth']
        
        for strategy in strategies:
            if strategy == 'half':
                U_r = np.array([[round_to_half(val) for val in row] for row in U])
                V_r = np.array([[round_to_half(val) for val in row] for row in V])
                W_r = np.array([[round_to_half(val) for val in row] for row in W])
            elif strategy == 'int':
                U_r = np.array([[round_to_int(val) for val in row] for row in U])
                V_r = np.array([[round_to_int(val) for val in row] for row in V])
                W_r = np.array([[round_to_int(val) for val in row] for row in W])
            elif strategy == 'quarter':
                def round_to_quarter(x):
                    return round(x * 4) / 4.0
                U_r = np.array([[round_to_quarter(val) for val in row] for row in U])
                V_r = np.array([[round_to_quarter(val) for val in row] for row in V])
                W_r = np.array([[round_to_quarter(val) for val in row] for row in W])
            elif strategy == 'eighth':
                def round_to_eighth(x):
                    return round(x * 8) / 8.0
                U_r = np.array([[round_to_eighth(val) for val in row] for row in U])
                V_r = np.array([[round_to_eighth(val) for val in row] for row in V])
                W_r = np.array([[round_to_eighth(val) for val in row] for row in W])
            else:
                continue
            
            if verify_np(U_r, V_r, W_r, r):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None
    
    def polish(U, V, W, r, max_iter=50, tol=1e-12):
        """High-precision polish pass to tighten the solution."""
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Solve for V
            A = build_A_for_V(U, W, r)
            b = np.ones(n * m * p)
            V_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Solve for U
            A = build_A_for_U(V_new, W, r)
            U_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Solve for W
            A = build_A_for_W(U_new, V_new, r)
            W_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Check convergence
            err = compute_error(U_new, V_new, W_new, r)
            if err < tol:
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    def try_structured_rank_r(r, seed=42, max_iter=200):
        """
        Try to find a rank-r decomposition using alternating least squares
        with structured initialization.
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
        
        for iteration in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Solve for V given U, W
            A = build_A_for_V(U, W, r)
            b = np.ones(n * m * p)
            V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat
            
            # Solve for U given V, W
            A = build_A_for_U(V, W, r)
            U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U = U_flat
            
            # Solve for W given U, V
            A = build_A_for_W(U, V, r)
            W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat
            
            # Check error
            error = compute_error(U, V, W, r)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-9:
                # Try to round to half-integers and other strategies
                result = try_rounding(U, V, W, r)
                if result:
                    return result
        
        # If we didn't get exact but got close, try rounding the best solution
        if best_error < 1e-6:
            result = try_rounding(best_U, best_V, best_W, r)
            if result:
                return result
        
        return None
    
    # Try ranks from 18 upward (18 is the theoretical lower bound)
    for r in [18, 19, 20, 21, 22, 23, 24]:
        if time.time() > DEADLINE:
            break
        
        max_seeds = 30 if r <= 19 else 20
        for seed in range(max_seeds):
            if time.time() > DEADLINE:
                break
            
            res = try_structured_rank_r(r, seed=42 + seed)
            if res:
                U, V, W = res
                # Polish the solution
                U_polished, V_polished, W_polished = polish(
                    np.array(U), np.array(V), np.array(W), r, max_iter=30
                )
                result = try_rounding(U_polished, V_polished, W_polished, r)
                if result:
                    U_r, V_r, W_r = result
                    return {"U": U_r, "V": V_r, "W": W_r}
                # If polish didn't help, use the original
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
