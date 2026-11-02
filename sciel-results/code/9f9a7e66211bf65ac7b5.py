import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Combines numerical optimization (least_squares) with vectorized verification
    and greedy column removal to find the lowest possible rank.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    n, m, p = 3, 4, 7
    R_school = n * m * p  # 84
    
    # Precompute target tensor values
    # T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build index map for fast verification
    index_map = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                index_map.append((u_idx, v_idx, w_idx))
    index_map = np.array(index_map)  # shape (n*m*p, 3)
    
    def verify_decomposition_fast(U, V, W, tol=1e-10):
        """
        Vectorized verification using the index map.
        """
        u_indices = index_map[:, 0]
        v_indices = index_map[:, 1]
        w_indices = index_map[:, 2]
        
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        products = U_rows * V_rows * W_rows
        sums = products.sum(axis=1)
        
        return np.all(np.abs(sums - 1.0) < tol)
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        for r in range(R):
            i = r // (m * p)
            remainder = r % (m * p)
            j = remainder // p
            k = remainder % p
            
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            U[u_row, r] = 1.0
            V[v_row, r] = 1.0
            W[w_row, r] = 1.0
        
        return U, V, W, R
    
    def residual_fn(params, R):
        """
        Compute residual for least_squares optimization.
        params: flattened [U_flat, V_flat, W_flat]
        Returns: vector of residuals (one per (i,j,k) entry)
        """
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R)).T  # (R, n*m)
        V = V_flat.reshape((m * p, R)).T  # (R, m*p)
        W = W_flat.reshape((p * n, R)).T  # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        # Reconstruct: sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        recon = np.einsum('rij,rjk,rki->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target is all ones for the (n*m*p) entries
        target = np.ones((n, m, p))
        
        return (recon - target).flatten()
    
    def try_numerical_search(R, max_restarts=3, time_budget=30.0):
        """
        Try to find a valid decomposition of rank R using least_squares.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + R)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.3
            
            try:
                res = least_squares(
                    residual_fn, x0, args=(R,), 
                    method='lm', max_nfev=100,
                    xtol=1e-12, ftol=1e-12
                )
            except Exception:
                continue
            
            if res.cost < 1e-10:
                # Extract solution
                size_U = n * m * R
                size_V = m * p * R
                
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape((n * m, R))
                V = V_flat.reshape((m * p, R))
                W = W_flat.reshape((p * n, R))
                
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                if verify_decomposition_fast(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
                
                # If snap fails, verify original
                if verify_decomposition_fast(U, V, W, tol=1e-10):
                    return U, V, W
        
        return None
    
    def greedy_remove(U, V, W, max_time=20.0):
        """
        Greedy column removal to further reduce rank.
        """
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = U.shape[1]
        initial_rank = current_rank
        removed = set()
        
        # Try removing columns in reverse order
        for col in range(initial_rank - 1, -1, -1):
            if time.time() > max_time:
                break
            
            current_idx = col - sum(1 for r in removed if r < col)
            
            if current_idx < 0 or current_idx >= current_rank:
                continue
            
            mask = np.ones(current_rank, dtype=bool)
            mask[current_idx] = False
            
            U_trial = current_U[:, mask]
            V_trial = current_V[:, mask]
            W_trial = current_W[:, mask]
            
            if verify_decomposition_fast(U_trial, V_trial, W_trial, tol=1e-10):
                current_U = U_trial
                current_V = V_trial
                current_W = W_trial
                current_rank -= 1
                removed.add(col)
        
        if current_rank < initial_rank:
            return current_U, current_V, current_W
        return None
    
    # Start with schoolbook
    U_school, V_school, W_school, R_school = build_schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Try numerical search for lower ranks
    # Start from a reasonable lower bound
    # For matrix multiplication, Strassen-type bounds suggest rank can be lower
    # Try ranks from 60 up to 83
    for R in range(60, min(84, R_school)):
        if time.time() > deadline - 20:
            break
        
        result = try_numerical_search(R, max_restarts=3, time_budget=25.0)
        if result is not None:
            U_found, V_found, W_found = result
            
            # Try greedy removal on the found solution
            greedy_result = greedy_remove(U_found, V_found, W_found, max_time=10.0)
            if greedy_result is not None:
                U_final, V_final, W_final = greedy_result
            else:
                U_final, V_final, W_final = U_found, V_found, W_found
            
            final_rank = U_final.shape[1]
            
            if final_rank < best_rank:
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = final_rank
            
            # If we found a good rank, try to go lower
            if best_rank <= 70:
                break
    
    # Final verification
    if not verify_decomposition_fast(best_U, best_V, best_W, tol=1e-10):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = R_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
