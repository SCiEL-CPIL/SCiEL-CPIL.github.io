import numpy as np
import time
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        
        i_idx = np.arange(n)
        j_idx = np.arange(m)
        k_idx = np.arange(p)
        
        I, J, K = np.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
        
        u_indices = I * m + J
        v_indices = J * p + K
        w_indices = K * n + I
        
        u_flat = u_indices.flatten()
        v_flat = v_indices.flatten()
        w_flat = w_indices.flatten()
        
        prod_sum = np.sum(U_arr[u_flat, :] * V_arr[v_flat, :] * W_arr[w_flat, :], axis=1)
        
        if np.max(np.abs(prod_sum - 1.0)) > tol:
            return False
        return True

    def get_strassen_decomposition():
        rank = 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        if col != rank:
            return get_schoolbook()
            
        return U, V, W

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_reduce_rank(U, V, W, max_iterations=100):
        """Try to reduce rank by finding and removing redundant columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        
        # For each column, check if it can be removed by adjusting others
        # This is a heuristic - try removing each column and see if we can
        # compensate with small adjustments to remaining columns
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if R <= 0:
                break
            
            removed = False
            for r in range(R):
                if time.time() > deadline:
                    break
                
                # Temporarily remove column r
                U_r = U[:, r].copy()
                V_r = V[:, r].copy()
                W_r = W[:, r].copy()
                
                U_temp = np.delete(U, r, axis=1)
                V_temp = np.delete(V, r, axis=1)
                W_temp = np.delete(W, r, axis=1)
                
                if U_temp.shape[1] == 0:
                    continue
                
                # Check if the removed column's contribution can be absorbed
                # by scaling other columns. This is a simplified check.
                # For exact rank reduction, we'd need to solve a system.
                # Here we just check if all entries in the removed column are zero
                if np.allclose(U_r, 0) and np.allclose(V_r, 0) and np.allclose(W_r, 0):
                    U = U_temp
                    V = V_temp
                    W = W_temp
                    R -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        return U, V, W, R

    # Primary strategy: Use the known Strassen decomposition of rank 54
    U, V, W = get_strassen_decomposition()
    
    # Verify the decomposition
    if check_decomposition(U, V, W, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
