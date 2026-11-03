"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
import random
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 100)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # 2. Try to find a lower rank decomposition (e.g., Rank 99)
    # We will attempt to solve for a rank R decomposition using least squares.
    # The tensor T is defined by T[i, j, k] = 1 if i,j,k match the multiplication indices.
    # Specifically, the non-zero entries are at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    
    # Generate the list of non-zero tensor coordinates
    tensor_coords = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_coords.append((i * m + j, j * p + k, k * n + i))
                
    num_equations = len(tensor_coords)
    
    def compute_error(params, R):
        """
        Compute the error of the tensor decomposition for a given set of parameters.
        params is a flat array of length R * (nm + mp + pn).
        """
        nm, mp, pn = n * m, m * p, p * n
        total_vars = R * (nm + mp + pn)
        
        # Reshape params into U, V, W
        # Order: U (R vectors of size nm), V (R vectors of size mp), W (R vectors of size pn)
        # Actually, let's structure it as:
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Flatten order: U[0,0], U[0,1]... U[nm-1, R-1], V[0,0]... V[mp-1, R-1], W[0,0]... W[pn-1, R-1]
        
        idx = 0
        U_flat = params[idx : idx + nm * R]
        idx += nm * R
        V_flat = params[idx : idx + mp * R]
        idx += mp * R
        W_flat = params[idx : idx + pn * R]
        idx += pn * R
        
        U = U_flat.reshape((nm, R))
        V = V_flat.reshape((mp, R))
        W = W_flat.reshape((pn, R))
        
        errors = np.zeros(num_equations)
        for eq_idx, (i_idx, j_idx, k_idx) in enumerate(tensor_coords):
            # Sum over r of U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
            # This is a dot product of the rows of U, V, W at the specified indices
            sum_val = np.dot(np.dot(U[i_idx, :], V[j_idx, :]), W[k_idx, :])
            # Target is 1.0
            errors[eq_idx] = sum_val - 1.0
            
        return errors

    # Try Rank 99
    target_rank = rank_school - 1
    if time.time() < deadline - 5:
        # Initialize with schoolbook terms (first target_rank terms)
        # This is a good starting point because it's close to a solution.
        init_params = np.zeros(target_rank * (n*m + m*p + p*n))
        
        # Fill init_params with the first target_rank schoolbook terms
        # Schoolbook term r corresponds to (i_r, j_r, k_r)
        # We need to map the flat index back to the matrix structure.
        
        # Let's just use random initialization scaled to 1.0, or schoolbook.
        # Schoolbook initialization:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    # Set U[i*m+j, col] = 1, etc.
                    # In the flat array:
                    # U part: index = (i*m+j) * R + col
                    # V part: index = nm*R + (j*p+k) * R + col
                    # W part: index = nm*R + mp*R + (k*n+i) * R + col
                    
                    u_idx = (i * m + j) * target_rank + col
                    v_idx = (n * m) * target_rank + (j * p + k) * target_rank + col
                    w_idx = (n * m) * target_rank + (m * p) * target_rank + (k * n + i) * target_rank + col
                    
                    init_params[u_idx] = 1.0
                    init_params[v_idx] = 1.0
                    init_params[w_idx] = 1.0
                    col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
                
        # Run least squares
        try:
            res = least_squares(compute_error, init_params, args=(target_rank,), method='lm', max_nfev=2000)
            if res.success or res.fun.max() < 1e-6:
                # Check if we can round to half-integers
                params = res.x
                nm, mp, pn = n * m, m * p, p * n
                U_flat = params[0 : nm * target_rank]
                V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
                W_flat = params[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
                
                U = U_flat.reshape((nm, target_rank))
                V = V_flat.reshape((mp, target_rank))
                W = W_flat.reshape((pn, target_rank))
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                errors = compute_error(
                    np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), 
                    target_rank
                )
                
                if np.max(np.abs(errors)) < 1e-9:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
        except Exception:
            pass

    # If we found a lower rank, return it. Otherwise, return schoolbook.
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
