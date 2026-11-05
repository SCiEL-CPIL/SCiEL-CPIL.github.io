"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 115  # leave safety margin before 120s kill

    def verify_vec(U, V, W, n, m, p, rank, tol=1e-9):
        """Fully vectorized verification."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Compute all sums at once: T[i,j,k] = sum_t U3[i,j,t]*V3[j,k,t]*W3[k,i,t]
        # Use einsum for clarity
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.all(np.abs(T - 1.0) <= tol)

    def verify_vec_err(U, V, W, n, m, p, rank):
        """Return max absolute error."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))

    def _als_solve(U, V, W, rank, max_iter=200, tol=1e-12):
        """Vectorized ALS solver."""
        # Precompute index arrays for vectorization
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                return None
            
            # Solve for W: for each (k,i), W[k*n+i,:] = lstsq(A, b)
            # A[j,:] = U[i*m+j,:] * V[j*p+k,:]
            # b = ones(m)
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # A is m x rank
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = 0.1 * np.random.randn(rank)
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = 0.1 * np.random.randn(rank)
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = 0.1 * np.random.randn(rank)
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_vec(U, V, W, n, m, p, rank, tol=1e-9):
                    return U, V, W
        
        if verify_vec(U, V, W, n, m, p, rank, tol=1e-9):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=80, seed=0):
        """Polish to half-integers with multiple scales."""
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try rounding to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_vec(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                return U_r, V_r, W_r
            
            err = verify_vec_err(U_r, V_r, W_r, n, m, p, rank)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            # Perturb around the rounded solution
            scale = max(0.02, 0.5 * (1.0 - it / max_iters))
            U = U_r + rng.randn(n * m, rank) * 0.05 * scale
            V = V_r + rng.randn(m * p, rank) * 0.05 * scale
            W = W_r + rng.randn(p * n, rank) * 0.05 * scale
        
        # Final check on best rounded
        if verify_vec(best_U, best_V, best_W, n, m, p, rank, tol=1e-8):
            return best_U, best_V, best_W
        
        return best_U, best_V, best_W

    def try_construct_direct(rank, seed=42):
        """Try to construct a decomposition using structured initialization + ALS."""
        rng = np.random.RandomState(seed)
        
        # Try multiple initialization strategies
        strategies = ['schoolbook_partial', 'structured', 'random']
        
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            
            if strat == 'schoolbook_partial':
                # Start with truncated schoolbook
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for t in range(min(rank, n*m*p)):
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                U += rng.randn(n * m, rank) * 0.1
                V += rng.randn(m * p, rank) * 0.1
                W += rng.randn(p * n, rank) * 0.1
            elif strat == 'structured':
                # Use block structure: partition the multiplication
                # Think of it as: result[i,k] = sum_j A[i,j]*B[j,k]
                # Try to exploit sparsity patterns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Assign terms in a structured way
                # First n*m*p/rank terms get structured values
                for t in range(rank):
                    # Spread indices
                    i = (t * 7) % n
                    j = (t * 11) % m
                    k = (t * 13) % p
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                
                U += rng.randn(n * m, rank) * 0.15
                V += rng.randn(m * p, rank) * 0.15
                W += rng.randn(p * n, rank) * 0.15
            else:  # random
                U = rng.randn(n * m, rank) * 0.5
                V = rng.randn(m * p, rank) * 0.5
                W = rng.randn(p * n, rank) * 0.5
            
            # Run ALS
            result = _als_solve(U.copy(), V.copy(), W.copy(), rank, max_iter=200)
            if result is not None:
                return result
        
        return None

    best_result = None
    best_rank = n * m * p  # 24

    # Try ranks from lower bound up, with more seeds for promising ranks
    rank_seeds = {18: 25, 19: 25, 20: 20, 21: 20, 17: 15, 16: 10, 15: 8, 14: 5}
    
    for rank in range(14, 25):
        if time.time() > deadline - 10:
            break
        
        nseeds = rank_seeds.get(rank, 10)
        found = False
        for seed in range(nseeds):
            if time.time() > deadline - 5:
                break
            
            result = try_construct_direct(rank, seed=seed)
            if result is not None:
                U, V, W = result
                
                # Try rational polish
                U_r, V_r, W_r = rational_polish(U, V, W, rank, max_iters=60, seed=seed)
                
                if verify_vec(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                    best_result = {
                        "U": [[float(x) for x in row] for row in U_r],
                        "V": [[float(x) for x in row] for row in V_r],
                        "W": [[float(x) for x in row] for row in W_r]
                    }
                    best_rank = rank
                    found = True
                    break
                
                # Also try without polish
                if verify_vec(U, V, W, n, m, p, rank, tol=1e-9):
                    best_result = {
                        "U": [[round(float(x), 12) for x in row] for row in U],
                        "V": [[round(float(x), 12) for x in row] for row in V],
                        "W": [[round(float(x), 12) for x in row] for row in W]
                    }
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
