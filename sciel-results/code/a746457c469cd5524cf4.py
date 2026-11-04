"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: try to find a decomposition with rank < n*m*p = 54
    # We'll use a randomized local search / gradient-free optimization approach
    # to minimize the rank of the tensor decomposition.
    
    # The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to write T as sum over c of U[:,c] * V[:,c] * W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: Use a known lower bound and try to construct a valid decomposition.
    # For 3x3x6, the schoolbook rank is 54. 
    # We can potentially do better by exploiting structure.
    
    # Let's try a direct sum approach: split the 6 columns into two groups of 3.
    # T = T1 + T2 where T1 handles k=0,1,2 and T2 handles k=3,4,5
    # Each Ti is a 3x3x3 multiplication tensor.
    
    # For 3x3x3, Strassen's algorithm gives rank 7.
    # So T1 and T2 each have rank 7, total rank 14.
    # But we can potentially share some factors.
    
    # Let's implement Strassen for 3x3x3 first.
    # Strassen's algorithm for 3x3 matrix multiplication uses 7 multiplications.
    # The standard Strassen decomposition for 2x2x2 has rank 7.
    # For 3x3, we can use a 3x3 version.
    
    # Actually, let's use a simpler approach: try to find a rank-54 solution first 
    # (schoolbook), then try to reduce it.
    
    # Let's implement a randomized search for lower rank decompositions.
    
    def schoolbook_decomposition():
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
        return U, V, W, rank
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        rank = U.shape[1]
        # Build the tensor T
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        
        # Compute the decomposition
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check exactness
        if not np.allclose(T, T_decomp, atol=1e-10):
            return False
        return True
    
    def try_rank(r, max_iters=1000, seed=42):
        """Try to find a rank-r decomposition using random initialization and local search."""
        np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Compute current error
            T_decomp = np.zeros((n * m, m * p, p * n))
            for r_idx in range(r):
                T_decomp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            err = np.linalg.norm(T - T_decomp)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            # Gradient-free step: perturb one factor
            factor = np.random.choice(3)
            r_idx = np.random.randint(r)
            if factor == 0:
                U[:, r_idx] += np.random.randn(n * m) * 0.1
            elif factor == 1:
                V[:, r_idx] += np.random.randn(m * p) * 0.1
            else:
                W[:, r_idx] += np.random.randn(p * n) * 0.1
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, True
        return best_U, best_V, best_W, False
    
    # First, get the schoolbook decomposition
    U, V, W, rank = schoolbook_decomposition()
    
    # Check if it's valid
    if check_decomposition(U, V, W):
        # Try to find a lower rank decomposition
        # Theoretical lower bounds suggest rank >= 27 for 3x3x3, so 54 for 3x3x6
        # But with sharing, we might do better.
        
        # Try ranks from 40 down to 30
        for r in range(min(53, n*m*p - 1), 25, -1):
            U_r, V_r, W_r, success = try_rank(r, max_iters=200, seed=42 + r)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # If no lower rank found, return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Should not happen
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
