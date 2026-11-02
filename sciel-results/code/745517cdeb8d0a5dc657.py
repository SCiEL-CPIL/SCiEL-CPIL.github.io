"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if decomposition is valid."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False, val
        return True, 1.0
    
    def solve_for_W(U, V):
        """Given U and V, solve for W."""
        r = U.shape[1]
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                a_vec = np.zeros(r)
                for j in range(m):
                    a_vec += U[i*m+j, :] * V[j*p+k, :]
                norm_sq = np.dot(a_vec, a_vec)
                if norm_sq > 1e-12:
                    W[k*n+i, :] = a_vec / norm_sq
                else:
                    W[k*n+i, :] = 0
        return W
    
    def solve_for_V(U, W):
        """Given U and W, solve for V."""
        r = U.shape[1]
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, r))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    V[j*p+k, :] = 0
        return V
    
    def solve_for_U(V, W):
        """Given V and W, solve for U."""
        r = V.shape[1]
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, r))
                for k in range(p):
                    B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    U[i*m+j, :] = 0
        return U
    
    def alternating_minimization(rank, n_iter=50, seed=42):
        """Run alternating minimization for given rank."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(n_iter):
            if time.time() > deadline:
                return None
            
            W = solve_for_W(U, V)
            V = solve_for_V(U, W)
            U = solve_for_U(V, W)
            
            valid, _ = check_decomposition(U, V, W, tol=1e-5)
            if valid:
                return U, V, W
        
        valid, _ = check_decomposition(U, V, W, tol=1e-5)
        if valid:
            return U, V, W
        return None
    
    # Try ranks from schoolbook-1 down
    for rank in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = alternating_minimization(rank, n_iter=50, seed=seed)
            if result is not None:
                U, V, W = result
                valid, _ = check_decomposition(U, V, W, tol=1e-8)
                if valid:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            continue
        else:
            result = alternating_minimization(rank, n_iter=200, seed=0)
            if result is not None:
                U, V, W = result
                valid, _ = check_decomposition(U, V, W, tol=1e-8)
                if valid:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
