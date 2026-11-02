"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Return the best exact tensor decomposition found within a time budget."""
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def _verify_exact(U, V, W, seed=42, trials=5):
        """Verify exact decomposition using multiple random trials."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        rng = np.random.RandomState(seed)
        for trial in range(trials):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_scalar = np.dot(Ua[:, r], A.flatten())
                v_scalar = np.dot(Va[:, r], B.flatten())
                C_approx += u_scalar * v_scalar * Wa[:, r].reshape(p, n).T
            if not np.allclose(C_approx, C_true, atol=1e-9):
                return False
        return True
    
    def _try_structured_rank(R, rng, max_iters=200):
        """Try structured sparse decompositions."""
        # For each rank-1 term, the U part should have support on rows that
        # correspond to a subset of (i,j) pairs, similarly for V and W.
        # A structured approach: each term contributes to a "block" of the multiplication.
        
        vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Each rank-1 term: U has k_u nonzeros, V has k_v, W has k_w
            # The structure should be that U's nonzeros are on consecutive (i,j) indices
            # or some structured pattern
            
            for r in range(R):
                # Choose a structured pattern
                pattern_type = rng.randint(0, 3)
                
                if pattern_type == 0:
                    # Single row in U, single column in V, single element in W
                    i_idx = rng.randint(n)
                    j_idx = rng.randint(m)
                    k_idx = rng.randint(p)
                    U[i_idx * m + j_idx, r] = 1.0
                    V[j_idx * p + k_idx, r] = 1.0
                    W[k_idx * n + i_idx, r] = 1.0
                elif pattern_type == 1:
                    # Two rows in U, two columns in V, four elements in W
                    i1, i2 = rng.choice(n, 2, replace=False)
                    j1, j2 = rng.choice(m, 2, replace=False)
                    k1, k2 = rng.choice(p, 2, replace=False)
                    U[i1 * m + j1, r] = 1.0
                    U[i2 * m + j2, r] = 1.0
                    V[j1 * p + k1, r] = 1.0
                    V[j2 * p + k2, r] = 1.0
                    W[k1 * n + i1, r] = 1.0
                    W[k2 * n + i2, r] = 1.0
                else:
                    # Random sparse with small values
                    nz_u = rng.randint(1, 4)
                    nz_v = rng.randint(1, 4)
                    nz_w = rng.randint(1, 4)
                    idx_u = rng.choice(n * m, nz_u, replace=False)
                    idx_v = rng.choice(m * p, nz_v, replace=False)
                    idx_w = rng.choice(p * n, nz_w, replace=False)
                    U[idx_u, r] = rng.choice(vals, nz_u)
                    V[idx_v, r] = rng.choice(vals, nz_v)
                    W[idx_w, r] = rng.choice(vals, nz_w)
            
            # Quick check
            ok = True
            for _ in range(3):
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_true = A @ B
                C_approx = np.zeros((n, p))
                for r in range(R):
                    u_scalar = np.dot(U[:, r], A.flatten())
                    v_scalar = np.dot(V[:, r], B.flatten())
                    C_approx += u_scalar * v_scalar * W[:, r].reshape(p, n).T
                if not np.allclose(C_approx, C_true, atol=1e-6):
                    ok = False
                    break
            
            if ok:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify_exact(U_list, V_list, W_list, seed=rng.randint(10000), trials=5):
                    return U_list, V_list, W_list
        
        return None
    
    # Start with schoolbook as baseline
    best_result = _schoolbook()
    best_rank = rank_schoolbook
    
    # Try structured search for lower ranks
    for R in range(28, min(best_rank, 60)):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(R * 7919 + 42)
        result = _try_structured_rank(R, rng, max_iters=300)
        
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = R
            break
    
    return best_result
# EVOLVE-BLOCK-END
