"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
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
    
    def build_A_for_w(w, U, V):
        """Build the linear system matrix A for a given w index.
        A has shape (n*m*m*p, r) where A[(u,v), c] = U[u,c] * V[v,c]
        """
        r = U.shape[1]
        # Vectorized construction
        # u ranges over n*m, v ranges over m*p
        # A[(u,v), c] = U[u, c] * V[v, c]
        # This is an outer product structure per column c
        A = np.zeros((n * m * m * p, r))
        for c in range(r):
            # For column c, A[., c] = U[:, c] outer V[:, c] reshaped
            col_u = U[:, c]  # shape (n*m,)
            col_v = V[:, c]  # shape (m*p,)
            # A[row, c] where row = u * (m*p) + v
            A[:, c] = np.outer(col_u, col_v).flatten()
        return A
    
    def build_b_for_w(w):
        """Build the RHS vector b for a given w index.
        b[(u,v)] = 1 if T[u,v,w] = 1 else 0
        T[u,v,w] = 1 iff j == j_v AND i == w % n AND k == w // n
        where u = i*m + j, v = j_v*p + k
        """
        b = np.zeros(n * m * m * p)
        i_w = w % n
        k_w = w // n
        for j in range(m):
            u = i_w * m + j
            v = j * p + k_w
            b[u * (m * p) + v] = 1.0
        return b
    
    def try_rank_with_linear_solve(r, num_restarts=30, seed=42):
        """Try to find a valid decomposition with rank r using linear system for W."""
        rng = np.random.RandomState(seed)
        
        # Precompute b vectors for all w
        b_vectors = [build_b_for_w(w) for w in range(p * n)]
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization of U and V
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                A = build_A_for_w(w, U, V)
                b = b_vectors[w]
                
                # Solve A * W[w,:] = b using least squares
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is good
                if len(residuals) > 0 and residuals[0] > 1e-6:
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
    # Try ranks from 36 upward
    
    for r in range(36, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_linear_solve(r, num_restarts=20, seed=42 + r)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
