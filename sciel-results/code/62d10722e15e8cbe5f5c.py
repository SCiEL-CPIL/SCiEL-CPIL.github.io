"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p  # 144
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
        r = U.shape[1]
        i_w = w % n
        k_w = w // n
        A = np.zeros((n * m * m * p, r))
        for c in range(r):
            col_u = U[:, c]
            col_v = V[:, c]
            A[:, c] = np.outer(col_u, col_v).flatten()
        return A

    def build_b_for_w(w):
        b = np.zeros(n * m * m * p)
        i_w = w % n
        k_w = w // n
        for j in range(m):
            u = i_w * m + j
            v = j * p + k_w
            b[u * (m * p) + v] = 1.0
        return b

    def try_rank_with_linear_solve(r, num_restarts=50, seed=42):
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

    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower-rank solutions
    for r in range(100, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_linear_solve(r, num_restarts=20, seed=42 + r)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
