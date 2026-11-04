"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Exploit block structure. View B (3x9) as three 3x3 blocks.
    Compute C[:,0:3] = A*B0, C[:,3:6] = A*B1, C[:,6:9] = A*B2.
    Find a low-rank decomposition for 2x3x3 matmul, then replicate 3 times.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def schoolbook_full():
        """Standard schoolbook decomposition for 2x3 x 3x9."""
        rank = n * m * p
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
        return U, V, W
    
    def find_rank_r_2x3x3(r, max_iters=300, seed=42):
        """
        Find a rank-r decomposition for 2x3 x 3x3 matrix multiplication.
        Returns (U, V, W) of shapes (6, r), (9, r), (6, r) or None.
        """
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Target tensor for 2x3x3: T[i,j,k] = 1 if indices align
        # T shape: (n*m, m*p3, p3*n) = (6, 9, 6) where p3=3
        p3 = 3
        T = np.zeros((n * m, m * p3, p3 * n))
        for i in range(n):
            for j in range(m):
                for k in range(p3):
                    T[i * m + j, j * p3 + k, k * n + i] = 1.0
        
        def tensor_norm_error(U, V, W, T_target):
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_decomp - T_target)
        
        # Try multiple random initializations
        best_err = float('inf')
        best_UVW = None
        
        for attempt in range(5):
            U = rng.randn(n * m, r)
            V = rng.randn(m * p3, r)
            W = rng.randn(p3 * n, r)
            
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p3*r].reshape(m*p3, r)
                W_flat = params[n*m*r + m*p3*r:].reshape(p3*n, r)
                T_decomp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                err = T_decomp - T
                return np.sum(err**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            result = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-11})
            
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r:n*m*r + m*p3*r].reshape(m*p3, r)
            W_opt = result.x[n*m*r + m*p3*r:].reshape(p3*n, r)
            
            err = tensor_norm_error(U_opt, V_opt, W_opt, T)
            
            if err < best_err:
                best_err = err
                best_UVW = (U_opt, V_opt, W_opt)
            
            if err < 1e-8:
                # Try to snap to rational values
                for denom in [1, 2, 3, 4]:
                    U_r = np.round(U_opt * denom) / denom
                    V_r = np.round(V_opt * denom) / denom
                    W_r = np.round(W_opt * denom) / denom
                    err_r = tensor_norm_error(U_r, V_r, W_r, T)
                    if err_r < 1e-10:
                        return U_r, V_r, W_r
                return U_opt, V_opt, W_opt
        
        if best_UVW is not None and best_err < 1e-6:
            return best_UVW
        
        return None
    
    # Try to find a low-rank decomposition for 2x3x3
    # Schoolbook for 2x3x3 is rank 9
    # Try ranks 5, 6, 7, 8
    
    block_decomp = None
    block_rank = None
    
    for r in [5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        for seed in [42, 123, 456]:
            if time.time() > deadline - 25:
                break
            result = find_rank_r_2x3x3(r, max_iters=200, seed=seed)
            if result is not None:
                block_decomp = result
                block_rank = r
                break
        if block_decomp is not None:
            break
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        # Replicate for 3 blocks: total rank = 3 * block_rank
        total_rank = 3 * block_rank
        
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        for block in range(3):
            # U is the same for all blocks (it's the A matrix)
            U_full[:, block * block_rank : (block + 1) * block_rank] = U_block
            
            # V: for block b, rows correspond to B[:, 3*b : 3*b+3]
            # V_block has shape (9, r) where row index = j*3 + k_local
            # In V_full, row index = j*p + k_global = j*9 + (3*b + k_local)
            for j in range(m):
                for k_local in range(3):
                    v_row = j * 3 + k_local  # row in V_block
                    v_full_row = j * p + (3 * block + k_local)  # row in V_full
                    V_full[v_full_row, block * block_rank : (block + 1) * block_rank] = \
                        V_block[v_row, :]
            
            # W: for block b, columns correspond to C[:, 3*b : 3*b+3]
            # W_block has shape (6, r) where row index = k_local*n + i
            # In W_full, row index = k_global*n + i = (3*b + k_local)*n + i
            for k_local in range(3):
                for i in range(n):
                    w_row = k_local * n + i  # row in W_block
                    w_full_row = (3 * block + k_local) * n + i  # row in W_full
                    W_full[w_full_row, block * block_rank : (block + 1) * block_rank] = \
                        W_block[w_row, :]
        
        # Verify the decomposition
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_decomp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        err = np.linalg.norm(T_decomp - T)
        
        if err < 1e-8:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook_full()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
