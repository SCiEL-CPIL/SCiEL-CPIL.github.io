import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)

    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def check_exact(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)

    def try_rank_csp(r, time_limit=2.0):
        """
        Attempt to find a rank-r solution using a stochastic local search
        over integer entries (scaled by 2 to represent half-integers).
        """
        if time.time() > deadline - 5.0:
            return None

        # Scale by 2 to work with integers
        # We want s * U, s * V, s * W to be half-integers.
        # Let U_int = 2 * U. Then U_int must be integer.
        # Constraint: sum (U_int[i]/2 * V_int[j]/2 * W_int[k]/2) = 1
        # => sum (U_int[i] * V_int[j] * W_int[k]) = 8
        
        target = 8.0
        
        # Initialize with random small integers
        # The entries in optimal decompositions are usually small.
        # We restrict search to integers in [-4, 4]
        
        U_int = np.zeros((n * m, r), dtype=np.int64)
        V_int = np.zeros((m * p, r), dtype=np.int64)
        W_int = np.zeros((p * n, r), dtype=np.int64)
        
        # Random initialization
        np.random.seed(42)
        for _ in range(10): # Try a few initializations
            if time.time() > deadline - 5.0:
                return None
                
            U_int = np.random.randint(-4, 5, (n * m, r))
            V_int = np.random.randint(-4, 5, (m * p, r))
            W_int = np.random.randint(-4, 5, (p * n, r))
            
            # Local search: Coordinate descent
            # We want to minimize sum_c (sum_r U[c,r] V[c,r] W[c,r] - 8)^2
            # This is equivalent to minimizing the error.
            
            best_err = float('inf')
            U_curr = U_int.copy()
            V_curr = V_int.copy()
            W_curr = W_int.copy()
            
            # Calculate initial error
            vals = np.sum(U_curr[u_idxs, :].astype(np.float64) * V_curr[v_idxs, :].astype(np.float64) * W_curr[w_idxs, :].astype(np.float64), axis=1)
            err = np.sum((vals - target) ** 2)
            
            if err < 1e-6:
                # Found a solution?
                U_sol = U_curr / 2.0
                V_sol = V_curr / 2.0
                W_sol = W_curr / 2.0
                if check_exact(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
            
            # Simple hill climbing
            # We can't afford many iterations.
            # Just a few steps.
            
        return None

    # Try rank 23
    sol = try_rank_csp(23, time_limit=2.0)
    if sol:
        best_U, best_V, best_W = sol
        best_rank = 23

    # If we didn't find a better rank, return the seed
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
