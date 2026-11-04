"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction


def _verify(U, V, W, n, m, p):
    """Verify decomposition using matrix multiplication check."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        prod = u_col @ v_col
        if not np.allclose(prod, w_col.T, atol=1e-9):
            return False
    return True


def _construct_schoolbook():
    n, m, p = DIMENSIONS
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


def _iht_search(n, m, p, deadline):
    """
    Iterative Hard Thresholding (IHT) search for sparse low-rank tensor decomposition.
    
    The tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication index.
    We seek U (nm x R), V (mp x R), W (pn x R) such that
        U[i,j] * V[j,k] * W[k,i] summed over R equals T[i,j,k].
    
    We start with an over-parameterized random initialization and iteratively
    minimize the residual while enforcing sparsity via hard thresholding.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor as a 3D array
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Start with a rank slightly above what we hope to achieve
    # Schoolbook is 105. Let's try to find something lower.
    # IHT works best when the true rank is much lower than the initial parameterization.
    # Let's start with R_init = 60 and try to prune down.
    
    best_rank = nm * mp * pn  # Schoolbook rank
    best_result = None
    
    # Try different initial ranks
    for R_init in [80, 60, 50, 40]:
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        np.random.seed(R_init)
        U = np.random.randn(nm, R_init) * 0.1
        V = np.random.randn(mp, R_init) * 0.1
        W = np.random.randn(pn, R_init) * 0.1
        
        # Learning rate for gradient steps
        lr = 0.01
        # Threshold for hard thresholding
        threshold = 1e-4
        
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
                
            # Compute current tensor estimate
            # T_est[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is expensive. Let's use a smarter update.
            
            # Gradient descent on U, V, W
            # Loss = ||T - T_est||^2
            # dL/dU[i,j,r] = -2 * T[i,j,k] * V[j,k,r] * W[k,i,r] + 2 * T_est[i,j,k] * V[j,k,r] * W[k,i,r]
            # This is O(nmp * R) per step, which is 105 * 80 = 8400 ops. Fast enough.
            
            # Compute T_est
            T_est = np.zeros((n, m, p), dtype=float)
            for r in range(R_init):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # T_est += u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is an outer product-like operation but with index permutation
                for i in range(n):
                    for j in range(m):
                        if abs(u_r[i, j]) > 1e-12:
                            T_est[i, j, :] += u_r[i, j] * v_r[j, :] * w_r[:, i]
            
            residual = T - T_est
            loss = np.sum(residual ** 2)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for r in range(R_init):
                        # Gradient of loss w.r.t. U[i,j,r]
                        # dL/dU[i,j,r] = -2 * sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
                        grad = -2.0 * np.sum(residual[i, j, :] * V[j, :, r] * W[:, i, r])
                        U[i, j, r] -= lr * grad
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for r in range(R_init):
                        # dL/dV[j,k,r] = -2 * sum_i residual[i,j,k] * U[i,j,r] * W[k,i,r]
                        grad = -2.0 * np.sum(residual[:, j, k] * U[:, j, r] * W[k, :, r])
                        V[j, k, r] -= lr * grad
            
            # Update W
            for k in range(p):
                for i in range(n):
                    for r in range(R_init):
                        # dL/dW[k,i,r] = -2 * sum_j residual[i,j,k] * U[i,j,r] * V[j,k,r]
                        grad = -2.0 * np.sum(residual[i, :, k] * U[i, :, r] * V[:, k, r])
                        W[k, i, r] -= lr * grad
            
            # Hard thresholding: zero out small entries
            U[np.abs(U) < threshold] = 0.0
            V[np.abs(V) < threshold] = 0.0
            W[np.abs(W) < threshold] = 0.0
            
            # Check if we've converged
            if loss < 1e-10:
                break
        
        # After optimization, try to snap to half-integers and verify
        # First, prune columns that are entirely zero
        U_mask = np.any(np.abs(U) > 1e-12, axis=0)
        V_mask = np.any(np.abs(V) > 1e-12, axis=0)
        W_mask = np.any(np.abs(W) > 1e-12, axis=0)
        valid_cols = U_mask & V_mask & W_mask
        
        if not np.any(valid_cols):
            continue
            
        U_pruned = U[:, valid_cols]
        V_pruned = V[:, valid_cols]
        W_pruned = W[:, valid_cols]
        
        R_eff = U_pruned.shape[1]
        
        if R_eff >= best_rank:
            continue
            
        # Try to snap to half-integers
        # Scale to make entries closer to integers/half-integers
        # First, check if entries are already close to half-integers
        def _snap_to_halfint(arr):
            snapped = np.round(arr * 2.0) / 2.0
            diff = np.max(np.abs(arr - snapped))
            return snapped, diff
        
        U_snapped, diff_U = _snap_to_halfint(U_pruned)
        V_snapped, diff_V = _snap_to_halfint(V_pruned)
        W_snapped, diff_W = _snap_to_halfint(W_pruned)
        
        max_diff = max(diff_U, diff_V, diff_W)
        
        if max_diff < 1e-6:
            # Convert to list of lists with exact half-integers
            U_list = U_snapped.tolist()
            V_list = V_snapped.tolist()
            W_list = W_snapped.tolist()
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                if R_eff < best_rank:
                    best_rank = R_eff
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Start with schoolbook
    best_result = _construct_schoolbook()
    best_rank = 3 * 5 * 7
    
    # Try IHT search
    if time.time() < deadline - 10:
        iht_result, iht_rank = _iht_search(3, 5, 7, deadline)
        if iht_result is not None and iht_rank < best_rank:
            best_result = iht_result
            best_rank = iht_rank
    
    return best_result
# EVOLVE-BLOCK-END
