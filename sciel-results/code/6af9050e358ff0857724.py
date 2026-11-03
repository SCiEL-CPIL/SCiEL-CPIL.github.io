import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 100  # 20s safety margin
    
    def build_decomposition(U_list, V_list, W_list, rank):
        """Build a decomposition from factor lists.
        U_list: list of rank vectors for each of n*m indices
        V_list: list of rank vectors for each of m*p indices
        W_list: list of rank vectors for each of p*n indices
        """
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for idx, vec in enumerate(U_list):
            U[idx, :] = vec
        for idx, vec in enumerate(V_list):
            V[idx, :] = vec
        for idx, vec in enumerate(W_list):
            W[idx, :] = vec
        return U, V, W
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
        # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    def try_strassen_like(rank):
        """Try to find a decomposition of the given rank using structured patterns."""
        # For 3x4*4x6, we can think of it as block matrices
        # A is 3x4, B is 4x6, C = A*B is 3x6
        
        # Try different rank values
        for r in range(rank, 35):  # Search from rank down to 35
            if time.time() > deadline:
                break
            
            # Use a structured approach: partition the computation
            # Think of A as [A1 A2] where A1 is 3x2, A2 is 3x2
            # B as [B1; B2] where B1 is 2x6, B2 is 2x6
            # C = A1*B1 + A2*B2
            
            # Each A_i*B_i is a 3x2 * 2x6 product
            # We can use rank-4 algorithms for 3x2*2x6 if available, but let's be more systematic
            
            # Direct approach: try to find U, V, W of size (12,r), (24,r), (18,r)
            # such that the tensor equation holds
            
            # Use random search with structure
            np.random.seed(42)
            
            for trial in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize with small integer values
                U_init = np.random.randint(-1, 2, size=(n * m, r))
                V_init = np.random.randint(-1, 2, size=(m * p, r))
                
                # For each (i,j,k), we need:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is a system of 72 equations in 18*r unknowns (W values)
                # For each c, the W values that appear are W[k*n+i, c] for various (i,k)
                
                # Let's set up the linear system for W
                # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
                # That's p*n*r = 18*r unknowns
                # Equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                
                # This is a sparse linear system. Let's build it.
                # Row index: (i,j,k) -> i*m*p + j*p + k = 72 rows
                # Column index: (k,n+i,c) -> k*n*r + i*r + c = 18*r columns
                
                A_mat = np.zeros((n * m * p, p * n * r))
                b_vec = np.ones(n * m * p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            for c in range(r):
                                val = U_init[i * m + j, c] * V_init[j * p + k, c]
                                if val != 0:
                                    col = k * n * r + i * r + c
                                    A_mat[row, col] += val
                
                # Try to solve the linear system
                try:
                    # Use least squares
                    W_flat, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    
                    # Check if the solution is exact (residuals should be 0)
                    if len(residuals) > 0 and residuals[0] < 1e-10:
                        # Check if W has small integer/half-integer values
                        W_check = W_flat.reshape(p * n, r)
                        # Check if close to half-integers
                        W_scaled = W_check * 2
                        if np.allclose(W_scaled, np.round(W_scaled), atol=1e-6):
                            W_result = np.round(W_scaled) / 2.0
                            # Verify
                            if check_decomposition(U_init, V_init, W_result, r):
                                return U_init, V_init, W_result, r
                except:
                    continue
        
        return None
    
    # Try Strassen-like or other structured decompositions
    # For 3x4*4x6, the lower bound is max(3*4, 4*6, 3*6) = max(12,24,18) = 24
    # Upper bound from schoolbook is 72
    # Known results: for 3x3*3x3, rank 23 (Strassen-like)
    # For our case, let's try ranks from 30 upward
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, create the schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
    
    # Try to find better decompositions
    for target_rank in range(35, min(50, n * m * p)):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_strassen_like(target_rank)
        if result is not None:
            U, V, W, r = result
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    
    # If we found a better result, use it; otherwise use schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_sb, V_sb, W_sb
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
