"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _polish(U, V, W, T, max_iters=50, tol=1e-12):
    """Gauss-Newton style polish to refine the decomposition."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T - T_recon
        err = np.max(np.abs(residual))
        if err < tol:
            break
        
        # Compute Jacobian and solve normal equations for each factor
        # For U: dT/dU[a,r] = V[:,r] * W[:,r] (outer product)
        # This is expensive, so use a simplified approach:
        # Update U via least squares with fixed V, W
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)
        T_flat = T.reshape(n * m, -1)
        try:
            U_new = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)[0].T
        except:
            U_new = U
        
        N = np.einsum('ar,cr->acr', U_new, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_v = T.reshape(m * p, -1)
        try:
            V_new = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            V_new = V
        
        P = np.einsum('ar,br->abr', U_new, V_new)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_w = T.reshape(p * n, -1)
        try:
            W_new = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            W_new = W
        
        U, V, W = U_new, V_new, W_new
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _try_rank(n, m, p, rank, max_iters=200, seed=42, polish=True):
    """Try to find a rank-r decomposition using ALS with optional polish."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # SVD-based initialization: matricize T and use top-r singular vectors
    T_flat = T.reshape(n * m, -1)  # (n*m, m*p*p*n)
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T_flat, full_matrices=False)
        # Use top-r left singular vectors for U initialization
        U = U_svd[:, :rank].copy()
    except:
        U = rng.randn(n * m, rank) * 0.1
    
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for it in range(max_iters):
        # Solve for U
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)
        T_flat = T.reshape(n * m, -1)
        try:
            U_new = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)[0].T
            U = U_new
        except:
            pass
        
        # Solve for V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_v = T.reshape(m * p, -1)
        try:
            V_new = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
            V = V_new
        except:
            pass
        
        # Solve for W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_w = T.reshape(p * n, -1)
        try:
            W_new = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
            W = W_new
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-12:
            break
    
    # Polish pass
    if polish:
        U, V, W, success = _polish(U, V, W, T, max_iters=30, tol=1e-13)
        if success:
            return U, V, W, True
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with trivial decomposition
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    search_ranks = list(range(lower_bound, rank_trivial))
    
    for r in search_ranks:
        if r >= best_rank:
            continue
        if time.time() - start_time > DEADLINE - 5:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            U_arr, V_arr, W_arr, success = _try_rank(n, m, p, r, max_iters=150, seed=seed, polish=True)
            if success:
                # Convert to list of lists with integer/half-integer check
                U_list = [[int(round(val * 2)) / 2 for val in row] for row in U_arr]
                V_list = [[int(round(val * 2)) / 2 for val in row] for row in V_arr]
                W_list = [[int(round(val * 2)) / 2 for val in row] for row in W_arr]
                
                # Verify the rounded solution is still exact
                U_chk = np.array(U_list, dtype=np.float64)
                V_chk = np.array(V_list, dtype=np.float64)
                W_chk = np.array(W_list, dtype=np.float64)
                T_recon = np.einsum('ar,br,cr->abc', U_chk, V_chk, W_chk)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a solution at this rank; continue to try lower ranks if possible
            # But since we're going from low to high, we already tried lower ones
            # Actually we go from lower_bound upward, so if found, we can stop
            # unless we want to verify no lower rank exists (but we already tried those)
            break
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
