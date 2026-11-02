"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Trivial rank: n*m*p = 105
    trivial_rank = n * m * p
    
    # Build the trivial decomposition
    U = np.zeros((n * m, trivial_rank), dtype=float)
    V = np.zeros((m * p, trivial_rank), dtype=float)
    W = np.zeros((p * n, trivial_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Precompute valid triples (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if the decomposition correctly represents the multiplication tensor."""
        rank = U_mat.shape[1]
        for (a, b, c) in triples:
            val = np.dot(np.dot(U_mat[a], V_mat[b]), W_mat[c])
            # Actually need: sum_t U[a,t] * V[b,t] * W[c,t]
            val = 0.0
            for t in range(rank):
                val += U_mat[a, t] * V_mat[b, t] * W_mat[c, t]
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def time_left():
        return deadline - time.time()
    
    # Try to find a lower rank decomposition using alternating least squares
    # For a fixed rank r, we want:
    #   For all (a,b,c) in triples: sum_t U[a,t]*V[b,t]*W[c,t] = 1
    #
    # We can fix two matrices and solve for the third.
    # Given V and W, for each row a of U, we need:
    #   For all b,c such that (a,b,c) in triples: sum_t U[a,t]*V[b,t]*W[c,t] = 1
    # This is a linear system in U[a,:].
    
    def try_rank(r, max_iters=200, n_seeds=3):
        """Try to find a rank-r decomposition."""
        if r >= trivial_rank:
            return None
        
        for seed in range(n_seeds):
            if time_left() < 10:
                break
            
            np.random.seed(seed)
            
            # Random initialization
            U_mat = np.random.randn(n * m, r) * 0.5
            V_mat = np.random.randn(m * p, r) * 0.5
            W_mat = np.random.randn(p * n, r) * 0.5
            
            # Scale to make the product of magnitudes reasonable
            scale = (1.0 / (r ** 0.5))
            U_mat *= scale
            V_mat *= scale
            W_mat *= scale
            
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Update U: for each row a, solve linear system
                for a in range(n * m):
                    # Find all (b, c) such that (a, b, c) is a valid triple
                    constraints = []
                    rhs = []
                    for (a2, b, c) in triples:
                        if a2 == a:
                            # constraint: sum_t U[a,t] * V[b,t] * W[c,t] = 1
                            # Let x = U[a,:], then sum_t x[t] * (V[b,t]*W[c,t]) = 1
                            row = V[b] * W[c]
                            constraints.append(row)
                            rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        # Solve least squares: A_sys @ x = b_sys
                        if A_sys.shape[0] >= r:
                            x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            U_mat[a, :] = x
                
                # Update V: for each row b, solve linear system
                for b in range(m * p):
                    constraints = []
                    rhs = []
                    for (a, b2, c) in triples:
                        if b2 == b:
                            row = U_mat[a] * W_mat[c]
                            constraints.append(row)
                            rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        if A_sys.shape[0] >= r:
                            x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            V_mat[b, :] = x
                
                # Update W: for each row c, solve linear system
                for c in range(p * n):
                    constraints = []
                    rhs = []
                    for (a, b, c2) in triples:
                        if c2 == c:
                            row = U_mat[a] * V_mat[b]
                            constraints.append(row)
                            rhs.append(1.0)
                    
                    if len(constraints) > 0:
                        A_sys = np.array(constraints)
                        b_sys = np.array(rhs)
                        if A_sys.shape[0] >= r:
                            x, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            W_mat[c, :] = x
                
                # Check convergence
                if it % 20 == 19:
                    if verify(U_mat, V_mat, W_mat, tol=1e-6):
                        # Found a valid decomposition
                        # Try to clean up to get integer/half-integer entries
                        U_clean = np.round(U_mat * 2) / 2
                        V_clean = np.round(V_mat * 2) / 2
                        W_clean = np.round(W_mat * 2) / 2
                        if verify(U_clean, V_clean, W_clean, tol=1e-6):
                            return (U_clean, V_clean, W_clean)
                        else:
                            # Return as-is if close enough
                            return (U_mat, V_mat, W_mat)
        
        return None
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = trivial_rank
    
    # First, try the trivial decomposition as a fallback
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    # Try to find lower rank decompositions
    for r in range(lower_bound, trivial_rank):
        if time_left() < 15:
            break
        
        result = try_rank(r, max_iters=100, n_seeds=2)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break  # Found the lowest rank we can achieve
    
    # Convert to lists for the required interface
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
