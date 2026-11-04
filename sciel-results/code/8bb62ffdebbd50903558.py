"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
DEADLINE = 140.0

def _compute_tensor(n, m, p):
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that U @ V^T * W decomposition matches T."""
    # T_decomp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_decomp - T)) < tol

def _als_step(n, m, p, rank, T, V, W, max_iters=100, tol=1e-10):
    """Solve for U given V and W using alternating least squares."""
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For each a: T[a,b,c] for all b,c = sum_r U[a,r] * (V[b,r] * W[c,r])
    # Let M_r[b,c] = V[b,r] * W[c,r]
    # Then T[a,:,:] = sum_r U[a,r] * M_r
    # This is a linear system: [M_0, M_1, ..., M_{rank-1}] * U[a,:]^T = T[a,:,:]
    
    # Build the linear system
    # A_sys: (n*m * m*p, rank) where each row corresponds to (b,c) for a given a
    # Actually, for each a, we have m*p*n*p equations... no.
    # For fixed a, T[a,b,c] for b in [0,m*p), c in [0,n*p)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Let's vectorize: for fixed a, define vector t_a of length m*p*n*p
    # t_a[b*n*p + c] = T[a,b,c]
    # And matrix A_a of shape (m*p*n*p, rank) where A_a[b*n*p+c, r] = V[b,r]*W[c,r]
    # Then A_a @ U[a,:]^T = t_a
    
    U = np.zeros((n * m, rank))
    for a in range(n * m):
        # Build A_a
        A_a = np.outer(V, W)  # (m*p*n*p, rank)? No.
        # V is (m*p, rank), W is (n*p, rank)
        # V[b,r] * W[c,r] for all b,c,r
        A_a = np.einsum('br,cr->bcr', V, W)  # (m*p, n*p, rank)
        A_a_flat = A_a.reshape(-1, rank)  # (m*p*n*p, rank)
        
        t_a = T[a].reshape(-1)  # (m*p*n*p,)
        
        # Solve least squares
        try:
            sol, _, _, _ = np.linalg.lstsq(A_a_flat, t_a, rcond=None)
            U[a] = sol
        except:
            U[a] = np.random.randn(rank) * 0.1
    
    return U

def _als_step_V(n, m, p, rank, T, U, W, max_iters=100, tol=1e-10):
    """Solve for V given U and W."""
    V = np.zeros((m * p, rank))
    for b in range(m * p):
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed b: T[a,b,c] for all a,c = sum_r V[b,r] * (U[a,r] * W[c,r])
        A_b = np.einsum('ar,cr->acr', U, W)  # (n*m, n*p, rank)
        A_b_flat = A_b.reshape(-1, rank)
        t_b = T[:, b, :].reshape(-1)
        try:
            sol, _, _, _ = np.linalg.lstsq(A_b_flat, t_b, rcond=None)
            V[b] = sol
        except:
            V[b] = np.random.randn(rank) * 0.1
    return V

def _als_step_W(n, m, p, rank, T, U, V, max_iters=100, tol=1e-10):
    """Solve for W given U and V."""
    W = np.zeros((n * p, rank))
    for c in range(n * p):
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed c: T[a,b,c] for all a,b = sum_r W[c,r] * (U[a,r] * V[b,r])
        A_c = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        A_c_flat = A_c.reshape(-1, rank)
        t_c = T[:, :, c].reshape(-1)
        try:
            sol, _, _, _ = np.linalg.lstsq(A_c_flat, t_c, rcond=None)
            W[c] = sol
        except:
            W[c] = np.random.randn(rank) * 0.1
    return W

def _als_decompose(n, m, p, rank, T, rng, max_outer=20, max_inner=30):
    """Alternating least squares to find rank-r decomposition."""
    start = time.time()
    
    for outer in range(max_outer):
        if time.time() - start > 60:
            break
        
        # Random init
        U = rng.standard_normal((n * m, rank)) * 0.5
        V = rng.standard_normal((m * p, rank)) * 0.5
        W = rng.standard_normal((n * p, rank)) * 0.5
        
        for inner in range(max_inner):
            if time.time() - start > 60:
                break
            
            U = _als_step(n, m, p, rank, T, V, W)
            V = _als_step_V(n, m, p, rank, T, U, W)
            W = _als_step_W(n, m, p, rank, T, U, V)
            
            # Check convergence
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_decomp - T))
            if err < 1e-8:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r, V_r, W_r, T, tol=1e-6):
                    return U_r, V_r, W_r
        
        if time.time() - start > 60:
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _compute_tensor(n, m, p)
    
    start_time = time.time()
    
    # Schoolbook fallback
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((n * p, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    rng = np.random.default_rng(12345)
    
    # Try ranks from lower bound
    min_rank = max(n * m, m * p, n * p)  # 32
    
    # Try a few ranks
    for rank in range(min_rank, min_rank + 15):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        result = _als_decompose(n, m, p, rank, T, rng, max_outer=10, max_inner=20)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
