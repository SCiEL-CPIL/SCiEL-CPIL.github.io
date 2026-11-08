"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] for A(n x m) * B(m x p) = C(n x p).
    T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', j'==j, k==k'
    We flatten: U index = i*m+j, V index = j*p+k, W index = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T"""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    reconstructed = np.zeros_like(T)
    for c in range(rank):
        reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return np.allclose(reconstructed, T, atol=tol)

def _check_half_int(U, V, W, tol=1e-9):
    """Check if all entries are half-integers (i.e., 2*x is integer)"""
    for M in [U, V, W]:
        for val in M.flatten():
            if abs(2*val - round(2*val)) > tol:
                return False
    return True

def _round_to_half_int(M):
    """Round matrix to nearest half-integer"""
    return np.round(2*M) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Build target tensor
    T = _build_tensor(n, m, p)
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Try to find lower rank decompositions
    # Strategy: start from various initializations and use iterative refinement
    
    def try_rank(r, max_iter=500, seed=0):
        """Try to find a rank-r decomposition"""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Scale to match magnitude
        U *= 1.0 / (r ** 0.25)
        V *= 1.0 / (r ** 0.25)
        W *= 1.0 / (r ** 0.25)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W, solve for U
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For each (j,k): T[:,j,k] = U * (V[j,:] * W[k,:])
            # So U = T_reshaped * (VW)^+ 
            VW = np.einsum('jc,kc->jkc', V, W)  # (m*p, p*n)
            # Reshape T to (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, -1)
            VW_flat = VW.reshape(m*p, -1)
            # U = T_flat @ VW_flat^T @ (VW_flat @ VW_flat^T)^-1
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V
            UW = np.einsum('ic,kc->ick', U, W)  # (n*m, p*n)
            UW_flat = UW.reshape(n*m, -1)
            T_flat2 = T.transpose(1,0,2).reshape(m*p, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat2.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V)  # (n*m, m*p)
            UV_flat = UV.reshape(n*m, -1)
            T_flat3 = T.transpose(2,0,1).reshape(p*n, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat3.T, rcond=None)
                W = W_new.T
            except:
                pass
        
        # Check if we got close
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.max(np.abs(reconstructed - T))
        if err < 1e-6:
            # Round to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r
        return None
    
    # Try ranks from low to high
    for r in range(10, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_iter=300, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
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
