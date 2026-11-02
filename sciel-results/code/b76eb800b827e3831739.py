"""Matrix multiplication 3x4x5 - rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x, y, z] where:
    x = i*m+j (A index)
    y = j*p+k (B index)
    z = k*n+i (C index, column-major convention)
    T[x,y,z] = 1 if indices are consistent, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[x,r]*V[y,r]*W[z,r] equals the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U_arr, V_arr, W_arr)
    
    # Build expected
    expected = _build_expected_tensor(n, m, p)
    
    return np.allclose(D, expected, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                U[x, col] = 1.0
                V[y, col] = 1.0
                W[z, col] = 1.0
                col += 1
    
    return U, V, W


def _try_als_decomposition(n, m, p, target_rank, seed=0, max_iters=50, deadline=None):
    """
    Try to find a rank-`target_rank` decomposition using Alternating Least Squares.
    Returns (U, V, W) if found, None otherwise.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    expected = _build_expected_tensor(n, m, p)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            return None
        
        # Solve for U: For each x, U[x,:] @ (V ⊙ W).reshape(-1, R).T = T[x,:,:].ravel()
        VW = np.einsum('yr,zr->yzr', V, W)  # (m*p, p*n, R)
        VW_flat = VW.reshape(-1, target_rank)  # (m*p*p*n, R)
        
        # For each x, solve least squares: VW_flat.T @ U[x,:].T = T[x,:,:].ravel()
        # Actually: sum_r U[x,r] * VW_flat[idx, r] = T_x[idx]
        # This is: U[x,:] @ VW_flat.T = T_x
        # Or: VW_flat @ U[x,:].T = T_x (since T_x is 1D)
        for x in range(n * m):
            T_x = expected[x, :, :].ravel()
            try:
                sol, _, _, _ = np.linalg.lstsq(VW_flat, T_x.reshape(-1, 1), rcond=None)
                U[x, :] = sol.flatten()
            except:
                pass
        
        # Solve for V: For each y, V[y,:] @ (U ⊙ W).reshape(-1, R).T = T[:,y,:].ravel()
        UW = np.einsum('xr,zr->x zr', U, W)  # (n*m, p*n, R)
        UW_flat = UW.reshape(-1, target_rank)  # (n*m*p*n, R)
        
        for y in range(m * p):
            # T[:,y,:] has shape (n*m, p*n)
            T_y = expected[:, y, :].ravel()
            try:
                sol, _, _, _ = np.linalg.lstsq(UW_flat, T_y.reshape(-1, 1), rcond=None)
                V[y, :] = sol.flatten()
            except:
                pass
        
        # Solve for W: For each z, W[z,:] @ (U ⊙ V).reshape(-1, R).T = T[:,:,z].ravel()
        UV = np.einsum('xr,yr->xyr', U, V)  # (n*m, m*p, R)
        UV_flat = UV.reshape(-1, target_rank)  # (n*m*m*p, R)
        
        for z in range(p * n):
            # T[:,:,z] has shape (n*m, m*p)
            T_z = expected[:, :, z].ravel()
            try:
                sol, _, _, _ = np.linalg.lstsq(UV_flat, T_z.reshape(-1, 1), rcond=None)
                W[z, :] = sol.flatten()
            except:
                pass
        
        # Check convergence
        if it % 5 == 0:
            D = np.einsum('xr,yr,zr->xyz', U, V, W)
            err = np.linalg.norm(D - expected)
            if err < 1e-8:
                return U, V, W
    
    # Final check
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    if np.allclose(D, expected, atol=1e-8):
        return U, V, W
    
    return None


def _try_sparse_search(n, m, p, target_rank, deadline):
    """
    Try a structured search for lower rank decompositions.
    Uses the idea that certain sub-blocks might have lower rank.
    """
    # For 3x4x5, the schoolbook rank is 60.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 20
    # But matmul rank is generally believed to be close to n*m*p for small matrices.
    
    # Try a few ranks below 60
    for r in range(target_rank, 55, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            result = _try_als_decomposition(n, m, p, r, seed=seed, max_iters=20, deadline=deadline)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U, V, W
    
    return None


def construct():
    """
    Constructs the rank-minimal tensor decomposition for 3x4x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # We'll try a few candidate ranks below schoolbook
    for target_rank in range(best_rank - 1, max(best_rank - 20, 40), -1):
        if time.time() > deadline - 30:
            break
        
        result = _try_sparse_search(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists of lists with integer entries
    # For exact integer/half-integer, we round to nearest 0.5
    U_list = np.round(best_U * 2).astype(int).tolist()
    V_list = np.round(best_V * 2).astype(int).tolist()
    W_list = np.round(best_W * 2).astype(int).tolist()
    
    # Scale back: since we rounded to nearest 0.5, we need to divide by 2
    # But the return type expects integer/half-integer entries.
    # Let's return as floats that are multiples of 0.5
    U_list = [[val / 2.0 for val in row] for row in U_list]
    V_list = [[val / 2.0 for val in row] for row in V_list]
    W_list = [[val / 2.0 for val in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
