import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses Alternating Least Squares (ALS) to minimize the error, then rounds to rationals.
    """
    if time.time() > _DEADLINE - 10:
        return None

    target_rank = 49
    
    try:
        best_error = float('inf')
        best_params = None
        
        max_restarts = 50 # ALS is fast, so we can do many restarts
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 2:
                break
            
            np.random.seed(1000 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS Iterations
            # We want to minimize sum (sum_c U_ij,c V_jk,c W_ki,c - 1)^2
            # For fixed V, W, update U_ij:
            # For each i, j:
            #   System: sum_c U_ij,c * (V_jk,c * W_ki,c) = 1 for all k
            #   Let A_ij be matrix of size (p, r) where A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
            #   Solve A_ij @ U_ij = ones(p)
            
            for it in range(100): # Max ALS iterations
                if time.time() > _DEADLINE - 2:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A: (p, r)
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A @ x = ones(p)
                        try:
                            # Use least squares
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except np.linalg.LinAlgError:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # System: sum_c V_jk,c * (U_ij,c * W_ki,c) = 1 for all i
                        # Let B_jk be matrix of size (n, r) where B_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Solve B_jk @ V_jk = ones(n)
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            y, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = y
                        except np.linalg.LinAlgError:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # System: sum_c W_ki,c * (U_ij,c * V_jk,c) = 1 for all j
                        # Let C_ki be matrix of size (m, r) where C_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Solve C_ki @ W_ki = ones(m)
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = z
                        except np.linalg.LinAlgError:
                            pass
                
                # Check convergence / error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            error += (s - 1.0)**2
                
                if error < 1e-10:
                    break

            if error < best_error:
                best_error = error
                best_params = (U.copy(), V.copy(), W.copy())
            
            if best_error < 1e-6:
                break

        if best_error < 1e-4 and best_params is not None:
            U, V, W = best_params
            
            # Try rounding to rationals
            # The entries of the tensor are integers (1). 
            # Often decompositions use integers or half-integers.
            
            def try_round(U, V, W, denom):
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                # Convert to list of lists of floats for verification
                # The verify function expects exact arithmetic. 
                # If we use floats, 1.0 might not be exact if denom is not power of 2? 
                # 1/2, 1/4, 1/8 are exact in binary.
                # But the verify function uses `!=` on sums of products.
                # If we pass floats, 0.5 * 0.5 * 0.5 = 0.125. Sum might be 1.0 exactly?
                # It's safer to convert to Fractions or check with high precision.
                # However, the task says "integer/half-integer entries".
                # So we only need to check denom=1 and denom=2.
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return U_r, V_r, W_r
                return None

            res = try_round(U, V, W, 1)
            if res:
                U_r, V_r, W_r = res
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

            res = try_round(U, V, W, 2)
            if res:
                U_r, V_r, W_r = res
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If ALS converged to a very low error, maybe the solution is rational 
            # but not aligned to the grid? Unlikely for small dimensions.
            # Or maybe we need to permute/scale?
            # For now, if we can't round, we fail.

    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
