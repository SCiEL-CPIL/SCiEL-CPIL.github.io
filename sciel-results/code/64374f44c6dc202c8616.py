import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    Returns {'U', 'V', 'W'} where U is (16, rank), V is (32, rank), W is (32, rank).
    The decomposition satisfies: sum_r U[:,r] * V[:,r] * W[:,r] = multiplication tensor.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # ... (existing Strassen construction code) ...
    
    # After constructing U, V, W of rank 98:
    
    # Attempt to reduce rank by finding linear dependencies among the rank-1 tensors.
    # Flatten U, V, W to a matrix of shape (16*32*32, 98)
    # If there is a non-trivial null space, we can eliminate a term.
    
    def try_reduce_rank(U, V, W):
        rank = U.shape[1]
        # Flatten
        # T_r is the outer product of U[:,r], V[:,r], W[:,r]
        # We want c such that sum c_r * T_r = 0
        # This is equivalent to finding null space of matrix A where A[:,r] = vec(T_r)
        
        # Constructing A explicitly is expensive (16384 * 98 floats)
        # 16384 * 98 * 8 bytes ~ 12 MB. Feasible.
        
        # vec(T_r)_{i,j,k} = U[i,r] * V[j,r] * W[k,r]
        
        # We can form A by looping r and filling columns.
        A = np.zeros((16*32*32, rank))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Outer product
            # np.einsum('i,j,k->ijk', u, v, w).flatten()
            A[:, r] = np.einsum('i,j,k->ijk', u, v, w).flatten()
            
        # SVD to find null space
        # A is 16384 x 98.
        # SVD will give Vt of shape (98, 98) if full_matrices=False?
        # Actually, we want the right null space.
        # A = U_svd * S * Vt
        # Null space of A is spanned by rows of Vt corresponding to zero singular values.
        
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        
        tol = 1e-10
        # Find indices where singular value is close to 0
        null_indices = np.where(s < tol)[0]
        
        if len(null_indices) == 0:
            return U, V, W
        
        # We have a null space.
        # Each row in Vt[null_indices] is a vector c in R^98 such that A @ c = 0.
        # This means sum_r c_r T_r = 0.
        
        # Try to eliminate one term for each null vector.
        # We pick a null vector c.
        # We want to remove term k.
        # T_k = - sum_{r!=k} (c_r/c_k) T_r
        # New T = sum_{r!=k} (1 + c_r/c_k) T_r
        
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        
        for c in Vt[null_indices]:
            # Find index k with largest |c_k|
            k = np.argmax(np.abs(c))
            if c[k] == 0:
                continue
            
            # Compute new coefficients
            new_coeffs = np.ones(rank)
            new_coeffs[k] = 0
            for r in range(rank):
                if r == k:
                    continue
                new_coeffs[r] = 1 + c[r] / c[k]
            
            # Construct new U, V, W
            # We can scale U[:, r] by new_coeffs[r]
            new_U = U.copy()
            new_V = V.copy()
            new_W = W.copy()
            
            valid = True
            for r in range(rank):
                if new_coeffs[r] == 0:
                    continue
                # Scale U[:, r]
                new_U[:, r] *= new_coeffs[r]
                
                # Check if entries are integer or half-integer
                # 2 * entry must be integer
                scaled_U = new_U[:, r]
                # Check if 2 * scaled_U is close to integer
                two_scaled = 2 * scaled_U
                if not np.allclose(two_scaled, np.round(two_scaled), atol=1e-5):
                    valid = False
                    break
            
            if valid:
                # Remove column k
                mask = np.ones(rank, dtype=bool)
                mask[k] = False
                new_U = new_U[:, mask]
                new_V = new_V[:, mask]
                new_W = new_W[:, mask]
                
                new_rank = new_U.shape[1]
                if new_rank < best_rank:
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    best_rank = new_rank
                    
                # We can only do one reduction per null vector easily.
                # Repeating would require re-computing SVD.
                # Given time constraints, one pass is good.
                
        return best_U, best_V, best_W

    # Call the reduction
    U, V, W = try_reduce_rank(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
