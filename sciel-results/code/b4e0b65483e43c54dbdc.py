"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that (U, V, W) exactly decomposes the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[x, y, z] where x=i*m+j, y=j'*p+k, z=k'*n+i' is 1 iff j==j', i==i', k==k'.
    We check: sum_r U[x,r] * V[y,r] * W[z,r] == T[x,y,z] for all x,y,z.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build expected tensor: T[x, y, z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for consistent i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, else 0
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]
    # Using einsum for efficiency
    D = np.einsum('xr,yr,zr->xyz', U_arr, V_arr, W_arr)
    
    return np.allclose(D, expected, atol=tol)

def _build_tensor_slices(n, m, p):
    """
    Returns a list of vectors t_z for z in 0..pn-1.
    t_z is a vector of length (n*m * m*p) representing the slice of the tensor at z.
    t_z[x, y] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some j.
    """
    slices = []
    for z in range(p * n):
        k = z // n
        i = z % n
        t = np.zeros((n * m, m * p))
        for j in range(m):
            x = i * m + j
            y = j * p + k
            t[x, y] = 1.0
        slices.append(t.flatten())
    return slices

def _try_rank_reduction(n, m, p, target_rank, deadline, seed=42):
    """
    Attempt to find a decomposition of the given target rank.
    Strategy:
    1. Randomly initialize U and V with small integers.
    2. Construct K where K[(x,y), r] = U[x,r] * V[y,r].
    3. For each slice t_z, check if t_z is in the column space of K.
    4. If yes for all z, solve for W_z.
    5. Check if W entries are rational (close to integers or half-integers).
    """
    np.random.seed(seed)
    slices = _build_tensor_slices(n, m, p)
    num_slices = p * n
    dim_x = n * m
    dim_y = m * p
    total_dim = dim_x * dim_y
    
    max_iter = 500
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline - 1:
            break
            
        # Initialize U and V with small integers
        # Using -1, 0, 1
        U = np.random.choice([-1, 0, 1], size=(dim_x, target_rank))
        V = np.random.choice([-1, 0, 1], size=(dim_y, target_rank))
        
        # Construct K: shape (total_dim, target_rank)
        # K[(x,y), r] = U[x,r] * V[y,r]
        # We can use einsum or broadcasting
        # U is (dim_x, R), V is (dim_y, R)
        # We want K of shape (dim_x*dim_y, R)
        # K[x*dim_y + y, r] = U[x,r] * V[y,r]
        
        # Efficient construction:
        # Expand U to (dim_x, 1, R)
        # Expand V to (1, dim_y, R)
        # Multiply and reshape
        K = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(total_dim, target_rank)
        
        # Check if all slices are in the column space of K
        # This is equivalent to checking if rank([K | t_z]) == rank(K) for all z
        # Or solving K w = t_z and checking residual.
        
        # Compute rank of K
        # Using SVD is robust
        try:
            _, s, _ = np.linalg.svd(K, full_matrices=False)
            rank_K = np.sum(s > 1e-9)
        except:
            continue
            
        valid = True
        W_solutions = []
        
        for z in range(num_slices):
            t_z = slices[z]
            # Solve K w = t_z
            # Use least squares
            w_z, residuals, rank, _ = np.linalg.lstsq(K, t_z, rcond=1e-9)
            
            # Check if residual is close to 0
            if len(residuals) == 0:
                # Underdetermined? But K is total_dim x R.
                # If total_dim > R, residuals should have length total_dim - R.
                # If total_dim < R, it's underdetermined.
                # Here total_dim = 240. R <= 60. So overdetermined.
                pass
            elif residuals[0] > 1e-9:
                valid = False
                break
            else:
                W_solutions.append(w_z)
                
        if not valid:
            continue
            
        # If we got here, all t_z are in Col(K).
        # Now we need to check if W_solutions contain rational numbers.
        # We prefer integer or half-integer.
        # Let's check if w_z entries are close to k/2 for some integer k.
        
        W = np.zeros((num_slices, target_rank))
        for z in range(num_slices):
            w_z = W_solutions[z]
            # Check if w_z is close to half-integers
            # Multiply by 2
            w_z_2 = w_z * 2
            # Round to nearest integer
            w_z_rounded = np.round(w_z_2)
            # Check difference
            if np.allclose(w_z_2, w_z_rounded, atol=1e-9):
                W[z] = w_z_rounded / 2.0
            else:
                # Maybe integers?
                w_z_rounded_int = np.round(w_z)
                if np.allclose(w_z, w_z_rounded_int, atol=1e-9):
                    W[z] = w_z_rounded_int
                else:
                    valid = False
                    break
                    
        if not valid:
            continue
            
        # Construct result
        result_U = U.tolist()
        result_V = V.tolist()
        result_W = W.tolist()
        
        # Final verification
        if _verify_decomposition(result_U, result_V, result_W, n, m, p):
            return {"U": result_U, "V": result_V, "W": result_W}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        pass
        
    # Try to find lower rank
    # Start from best_rank - 1 down to 40
    # Give more time to higher ranks (closer to 60) as they are more likely to exist?
    # Actually, lower rank is harder.
    # Let's try a few ranks.
    
    # Time allocation:
    # 60: 0s (known)
    # 59: 5s
    # 58: 5s
    # ...
    
    ranks_to_try = list(range(best_rank - 1, best_rank - 11, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Use multiple seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = _try_rank_reduction(n, m, p, rank, deadline, seed=seed)
            if result is not None:
                best = result
                best_rank = rank
                break # Move to next lower rank
                
    return best
# EVOLVE-BLOCK-END
