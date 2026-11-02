"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 5, 5, 5
    deadline = time.time() + 140.0  # 10s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def try_rank(rank, max_restarts=5, max_iters=200):
        """Try to find exact rank decomposition via alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(rank * 100 + restart)
            U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n*m, rank))
            V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m*p, rank))
            W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p*n, rank))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.array([[V[j*p+k, c] * W[k*n+i, c] for k in range(p)] for c in range(rank)])
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, c] * W[k*n+i, c] for i in range(n)] for c in range(rank)])
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.array([[U[i*m+j, c] * V[j*p+k, c] for j in range(m)] for c in range(rank)])
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-8:
                    # Round to half-integers and verify
                    U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                    V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                    W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                    err_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(rank))
                                err = abs(val - 1.0)
                                if err > err_r:
                                    err_r = err
                    if err_r < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        
        return None
    
    # Try ranks from 49 upward (known upper bound is 49)
    for rank in [49, 50, 55, 60, 70, 80, 100, 125]:
        if time.time() > deadline - 20:
            break
        result = try_rank(rank, max_restarts=3, max_iters=100)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
