import time
import numpy as np
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p  # 64
    
    # Schoolbook decomposition as fallback (Rank 64, all 1s/0s, valid half-integers)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Precompute index arrays for efficient matrix construction
    # For equation (i,j,k) at index eq_idx = i*m*p + j*p + k
    eq_indices = np.arange(total_eq)
    i_idx = (eq_indices // (m * p)).astype(int)
    j_idx = ((eq_indices // p) % m).astype(int)
    k_idx = (eq_indices % p).astype(int)
    
    # Precompute index mappings
    # U index: i*m + j
    u_row_idx = i_idx * m + j_idx
    # V index: j*p + k  
    v_row_idx = j_idx * p + k_idx
    # W index: k*n + i
    w_row_idx = k_idx * n + i_idx
    
    def snap_to_half_int(X):
        """Snap values to the nearest half-integer (k/2)."""
        return np.round(2 * X) / 2.0

    def compute_residual_snapped(U, V, W):
        """Compute residual using snapped values to ensure validity."""
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        
        U_eq = U_s[u_row_idx, :]
        V_eq = V_s[v_row_idx, :]
        W_eq = W_s[w_row_idx, :]
        vals = np.sum(U_eq * V_eq * W_eq, axis=1)
        return np.sum((vals - 1)**2)

    def solve_for_U(V, W, target_rank):
        """Fix V and W, solve linear system for U"""
        A_mat = np.zeros((total_eq, nm * target_rank))
        # Vectorized construction of A
        # A[eq, u_row*rank + r] = V[v_row, r] * W[w_row, r]
        # We can iterate over eq to fill rows, which is 64 iterations.
        for eq in range(total_eq):
            u_r = u_row_idx[eq]
            v_r = v_row_idx[eq]
            w_r = w_row_idx[eq]
            # The coefficients for U[u_r, :] are V[v_r, :] * W[w_r, :]
            coeffs = V[v_r, :] * W[w_r, :]
            A_mat[eq, u_r*target_rank : (u_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return U_flat.reshape(nm, target_rank)
    
    def solve_for_V(U, W, target_rank):
        """Fix U and W, solve linear system for V"""
        A_mat = np.zeros((total_eq, mp * target_rank))
        for eq in range(total_eq):
            v_r = v_row_idx[eq]
            u_r = u_row_idx[eq]
            w_r = w_row_idx[eq]
            coeffs = U[u_r, :] * W[w_r, :]
            A_mat[eq, v_r*target_rank : (v_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return V_flat.reshape(mp, target_rank)
    
    def solve_for_W(U, V, target_rank):
        """Fix U and V, solve linear system for W"""
        A_mat = np.zeros((total_eq, pn * target_rank))
        for eq in range(total_eq):
            w_r = w_row_idx[eq]
            u_r = u_row_idx[eq]
            v_r = v_row_idx[eq]
            coeffs = U[u_r, :] * V[v_r, :]
            A_mat[eq, w_r*target_rank : (w_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return W_flat.reshape(pn, target_rank)

    # Try to find a lower-rank decomposition
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # We want to find rank < 64.
    # Known lower bounds for 2x4x8 multiplication?
    # Schoolbook is 64.
    # Let's try ranks from 32 upwards.
    for target_rank in range(32, 64):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_iters = 20
        for init in range(max_iters):
            if time.time() > deadline:
                break
            
            np.random.seed(init * 1000 + target_rank)
            
            # Initialize randomly
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            prev_residual = float('inf')
            for it in range(200):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, target_rank)
                V = solve_for_V(U, W, target_rank)
                W = solve_for_W(U, V, target_rank)
                
                # Check residual with continuous values first for speed
                U_eq = U[u_row_idx, :]
                V_eq = V[v_row_idx, :]
                W_eq = W[w_row_idx, :]
                vals = np.sum(U_eq * V_eq * W_eq, axis=1)
                residual = np.sum((vals - 1)**2)
                
                if residual < 1e-8:
                    # Check if snapping to half-integers preserves exactness
                    snapped_res = compute_residual_snapped(U, V, W)
                    if snapped_res < 1e-10:
                        # Valid solution found
                        U_s = snap_to_half_int(U)
                        V_s = snap_to_half_int(V)
                        W_s = snap_to_half_int(W)
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                if abs(prev_residual - residual) < 1e-15:
                    break
                prev_residual = residual
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
    
    # Return the best we found (Schoolbook is always valid)
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
