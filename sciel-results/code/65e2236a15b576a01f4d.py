import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # The matrix multiplication tensor T has entries T_{ijk} = 1 for all i,j,k
    # in the standard basis where:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # We want T = sum_c U[:,c] * V[:,c] * W[:,c] (Hadamard product of outer products)
    # i.e., T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # where T_{a,b,c} = 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0.
    # Actually, the tensor is supported on the "multiplication" indices.
    # Let's define the target tensor T explicitly.
    
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    
    # We are looking for a decomposition of rank r < 60.
    # Since the tensor is sparse and structured, we can try to exploit this.
    
    # A key insight: The tensor is "separable" in a certain sense.
    # However, for 2x3x10, the exact rank is not trivially known to be < 60.
    # Let's try to find a rank < 60 using a more structured search.
    
    # Strategy:
    # 1. Try to find a rank r decomposition for r < 60.
    # 2. Use the fact that the tensor is supported on a specific set of indices.
    # 3. We can set up a system of linear equations if we fix two of the matrices.
    
    # Let's try to fix U and V to have a specific structure.
    # For example, let U and V be identity-like or sparse.
    
    # Alternative: Use the "subspace" method.
    # If the rank is r, then the image of the tensor in each mode is at most r-dimensional.
    
    # Let's try a randomized search with a better initialization.
    # We will try to solve for W given U and V, as before, but with more attempts
    # and better verification.
    
    # We will also try to enforce that the entries are close to integers or half-integers
    # by adding a regularization term or by rounding and re-solving.
    
    # For now, let's stick to the linear solve approach but with more robustness.
    
    def try_rank(r, seed=0):
        if r >= 60:
            return None
        
        np.random.seed(seed)
        
        # Generate random U and V
        # We can try to make them sparse or with small entries to help with rounding
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        
        W = np.zeros((p * n, r))
        
        # For each C_idx (k*n+i), solve for W[C_idx, :]
        for k in range(p):
            for i in range(n):
                c_idx = k * n + i
                # Build Y matrix of size m x r
                Y = np.zeros((m, r))
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    Y[j, :] = U[a_idx, :] * V[b_idx, :]
                
                rhs = np.ones(m)
                
                # Solve Y * x = rhs
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(Y, rhs, rcond=None)
                except:
                    return None
                
                # Check if the solution is exact
                if np.allclose(Y @ x, rhs, atol=1e-8):
                    W[c_idx, :] = x
                else:
                    return None
        
        # Verify the full decomposition
        # We need to check that for all (i,j,k), the sum is 1
        # And for all other indices, the sum is 0?
        # Actually, the tensor is only defined on the support.
        # The decomposition must match the tensor on the support.
        # What about outside the support?
        # The tensor T is 0 outside the support.
        # So we must also ensure that the decomposition gives 0 outside the support.
        # This is a much stronger condition.
        
        # Let's check a few random points outside the support.
        # If the decomposition gives non-zero values there, it's invalid.
        
        # But checking all points is expensive.
        # Let's check all points for small r.
        
        # Actually, the problem statement says "exactly decompose the matrix multiplication tensor".
        # This implies the tensor is the full 3D array, not just the support.
        # So T_{a,b,c} is 1 if (a,b,c) is in the support, 0 otherwise.
        
        # So we must ensure that the decomposition matches T everywhere.
        
        # Let's check all entries.
        # T has shape (n*m, m*p, p*n) = (6, 30, 20) = 3600 entries.
        # For each entry, we compute sum_r U_{a,r} V_{b,r} W_{c,r}.
        # This is 3600 * r operations. For r=59, that's ~212k operations. Fast.
        
        # But we need to construct T first.
        
        # Let's construct T as a sparse representation or a dense array.
        # Dense array of size 6x30x20 is small.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        
        # Now check the decomposition
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute this as a tensor contraction.
        
        # Efficient way:
        # For each c, T_recon[:, :, c] = sum_r U[:, r] V[:, r] W[c, r]
        # = sum_r W[c, r] (U[:, r] * V[:, r])
        # = sum_r W[c, r] outer(U[:, r], V[:, r])
        
        # Let's compute T_recon for all c.
        T_recon = np.zeros_like(T)
        for c in range(p*n):
            # T_recon[:, :, c] = sum_r W[c, r] (U[:, r] * V[:, r])
            # This is a linear combination of rank-1 matrices.
            # We can compute it as:
            # M = U * W[c, :]  # (n*m, r) * (r,) -> (n*m, r)
            # T_recon[:, :, c] = M @ V.T  # (n*m, r) @ (r, m*p) -> (n*m, m*p)
            
            M = U * W[c, :]  # (n*m, r)
            T_recon[:, :, c] = M @ V.T
        
        # Check if T_recon matches T
        if np.allclose(T_recon, T, atol=1e-8):
            # Check if entries are close to integers or half-integers
            def is_half_int(x):
                return abs(x * 2 - round(x * 2)) < 1e-6
            
            if all(is_half_int(x) for x in U.flatten()) and \
               all(is_half_int(x) for x in V.flatten()) and \
               all(is_half_int(x) for x in W.flatten()):
                
                U_frac = np.round(U * 2) / 2
                V_frac = np.round(V * 2) / 2
                W_frac = np.round(W * 2) / 2
                
                # Convert to lists
                U_list = U_frac.tolist()
                V_list = V_frac.tolist()
                W_list = W_frac.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 59 down to 50
    # We will try multiple seeds for each rank
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
        for seed in range(10):  # 10 attempts per rank
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                return result
    
    # Fallback: Schoolbook
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
