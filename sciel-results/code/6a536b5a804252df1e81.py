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
    
    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
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
    
    def try_rank_with_linear_solve(r, num_restarts=50):
        """Try to find a valid decomposition with rank r using linear system for W."""
        np.random.seed(42)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization of U and V
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # For each w, solve for W[w, :]
            # The equation is: for all (u, v), sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]
            # This is a linear system: A_w * W[w,:] = b_w
            # where A_w has rows indexed by (u,v) and A_w[(u,v), c] = U[u,c] * V[v,c]
            # and b_w[(u,v)] = T[u,v,w]
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                # Build the linear system for this w
                # Number of (u,v) pairs: n*m * m*p = 16 * 36 = 576
                # But we only need equations where T[u,v,w] could be non-zero
                # Actually, we need all equations to ensure zeros are also satisfied
                
                # Build A (576 x r) and b (576,)
                A = np.zeros((n * m * m * p, r))
                b = np.zeros(n * m * m * p)
                
                row = 0
                for u in range(n * m):
                    i = u // m
                    j = u % m
                    for v in range(m * p):
                        j_v = v // p
                        k = v % p
                        
                        # Check if T[u,v,w] = 1
                        # T[u,v,w] = 1 iff j == j_v AND i == w % n AND k == w // n
                        if j == j_v and i == w % n and k == w // n:
                            b[row] = 1.0
                        # else b[row] = 0.0 (already initialized)
                        
                        # A[row, c] = U[u,c] * V[v,c]
                        A[row, :] = U[u, :] * V[v, :]
                        row += 1
                
                # Solve A * W[w,:] = b using least squares
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is good
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower-rank solutions
    # Start from a reasonable lower bound and go up
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But we need to satisfy all constraints, so it's likely higher
    # Try ranks from 40 to 143
    
    for r in range(40, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_linear_solve(r, num_restarts=20)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            # Found a solution, try to go lower? No, we're going up.
            # Actually, we want the minimum, so if we find one at rank r, 
            # we should check if lower ranks work too. But that's expensive.
            # Let's just return the first one we find in ascending order.
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
