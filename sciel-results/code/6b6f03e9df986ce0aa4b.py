"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
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
        """Build the linear system matrix A for a given w index."""
        r = U.shape[1]
        A = np.zeros((n * m * m * p, r))
        for c in range(r):
            col_u = U[:, c]
            col_v = V[:, c]
            A[:, c] = np.outer(col_u, col_v).flatten()
        return A
    
    def build_b_for_w(w):
        """Build the RHS vector b for a given w index."""
        b = np.zeros(n * m * m * p)
        i_w = w % n
        k_w = w // n
        for j in range(m):
            u = i_w * m + j
            v = j * p + k_w
            b[u * (m * p) + v] = 1.0
        return b
    
    def try_rank_with_linear_solve(r, num_restarts=50, seed=42):
        """Try to find a valid decomposition with rank r using linear system for W."""
        rng = np.random.RandomState(seed)
        
        b_vectors = [build_b_for_w(w) for w in range(p * n)]
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                A = build_A_for_w(w, U, V)
                b = b_vectors[w]
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None
    
    def try_structured_decomposition():
        """Try structured decompositions using Kronecker products."""
        # Try to exploit structure: C = A @ B where A is 4x4, B is 4x9
        # This can be seen as 4 separate 4x4 multiplications plus 4x5 extra columns
        
        # Structure 1: Treat as block diagonal + extra
        # A @ B = [A @ B1 | A @ B2] where B1 is 4x4, B2 is 4x5
        # We can decompose A @ B1 and A @ B2 separately
        
        # For 4x4 @ 4x4, known optimal rank is 49 (Strassen-like)
        # For 4x4 @ 4x5, we need at least 4*4*5 = 80 in schoolbook
        # But we can share some structure
        
        # Let's try a simple structured approach:
        # Use the fact that matrix multiplication is bilinear
        
        # Try rank 49 + 45 = 94 (rough estimate)
        # Actually, let's just try various ranks with the linear solver but with
        # structured initializations
        
        structured_ranks = [49, 50, 55, 60, 64, 72, 80, 81, 84, 90, 96, 100, 108, 112, 120, 128, 135, 140]
        
        for r in structured_ranks:
            if time.time() > deadline - 15:
                break
            
            # Try with structured initialization
            result = try_rank_with_linear_solve(r, num_restarts=20, seed=100 + r)
            if result is not None:
                return result
        
        return None
    
    def try_block_decomposition():
        """Try exploiting block structure of the matrices."""
        # A is 4x4, B is 4x9
        # We can write B = [B1 | B2] where B1 is 4x4, B2 is 4x5
        # C = [A@B1 | A@B2]
        
        # For A@B1 (4x4 @ 4x4), use a rank-49 decomposition if available
        # For A@B2 (4x4 @ 4x5), use a rank-80 decomposition
        
        # Total rank would be 49 + 80 = 129, which is better than 144
        
        # Let's construct this explicitly
        # First, get a rank-49 decomposition for 4x4 @ 4x4
        # We'll use a simple Strassen-like construction
        
        # Actually, let's just use the linear solver with good initializations
        
        # Try rank 129 first (49 + 80)
        for r in [129, 128, 126, 124, 120, 116, 112, 108, 104, 100]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank_with_linear_solve(r, num_restarts=30, seed=200 + r)
            if result is not None:
                return result
        
        return None
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try structured approaches first
    result = try_structured_decomposition()
    if result is not None:
        U, V, W, rank = result
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try block decomposition
    if time.time() < deadline - 20:
        result = try_block_decomposition()
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find even lower-rank solutions
    for r in range(36, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_linear_solve(r, num_restarts=20, seed=42 + r)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
