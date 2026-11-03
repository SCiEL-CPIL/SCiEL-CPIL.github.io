"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 10s safety margin
    
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
    
    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        # For all i, k: sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j2*p+k,c] == delta_{j1,j2}
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    def error(U, V, W):
        """Compute the total error of the decomposition."""
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        err += (s - expected) ** 2
        return err
    
    def try_rank(r, max_iters=5000, lr=0.01):
        """Try to find a valid decomposition with rank r."""
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            # Compute error
            e = error(U, V, W)
            if e < 1e-10:
                return U, V, W, r
            
            # Gradient descent (approximate)
            # This is a non-convex problem, so simple GD may not work well
            # Let's skip gradient descent for now and just return if we find a solution
            break
        
        return None
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower-rank solutions
    # Start from a reasonable rank and try to decrease
    # For now, just return the schoolbook solution
    # A proper search would take too long
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
