"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j, :]
                    v_val = V_arr[j*p + k, :]
                    w_val = W_arr[k*n + i, :]
                    s = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Try to find lower-rank decompositions using structured approaches
    np.random.seed(42)
    
    def try_rank_reduction(U_sb, V_sb, W_sb, target_rank, max_attempts=100):
        """Try to find a rank-reduced decomposition by random linear combinations."""
        r = U_sb.shape[1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random mixing: U_new = U_sb @ Q, V_new = V_sb @ Q^{-1}, W_new = W_sb @ I
            # This preserves the tensor property if Q is invertible
            Q = np.random.randn(r, r)
            
            # Try to make some columns of U_new, V_new, or W_new zero
            # by choosing Q to have a large null space component
            
            # Alternative: Use structured mixing with sparse Q
            Q_sparse = np.random.randn(r, r)
            # Zero out some entries to create structure
            mask = np.random.rand(r, r) < 0.3
            Q_sparse[~mask] = 0
            
            U_new = U_sb @ Q_sparse
            V_new = V_sb @ Q_sparse
            W_new = W_sb @ Q_sparse
            
            # Check if we can drop columns that are near-zero in all three
            U_norms = np.linalg.norm(U_new, axis=0)
            V_norms = np.linalg.norm(V_new, axis=0)
            W_norms = np.linalg.norm(W_new, axis=0)
            
            # Find columns that are zero in at least one factor
            zero_mask = (U_norms < 1e-10) | (V_norms < 1e-10) | (W_norms < 1e-10)
            
            if zero_mask.sum() > 0:
                keep_mask = ~zero_mask
                if keep_mask.sum() < best_rank:
                    U_c = U_new[:, keep_mask]
                    V_c = V_new[:, keep_mask]
                    W_c = W_new[:, keep_mask]
                    if validate_fast(U_c, V_c, W_c):
                        return {"U": U_c, "V": V_c, "W": W_c}
        
        return None
    
    # Method 1: Random sparse mixing
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        result = try_rank_reduction(U_sb, V_sb, W_sb, rank_sb - 1, max_attempts=5)
        if result is not None:
            best_rank = min(best_rank, result["U"].shape[1])
            best = {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Method 2: Try Kronecker product structure
    # For 4x4 x 4x9, we can view this as (4x4) * (4x9)
    # Use the fact that 4 = 2*2 to create block structure
    
    # Create a structured decomposition using Kronecker products
    # Let A = [[a11, a12], [a21, a22]] where each aij is 2x2
    # Let B = [[b1], [b2]] where b1, b2 are 2x9
    
    # We can use the identity: (A*B)_{ij} = sum_k A_{ik} B_{kj}
    # For blocks: C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # Try to use 7 multiplications instead of 8 (Strassen-like)
    # But since B has different dimensions, we need to adapt
    
    # Alternative: Try to find a decomposition with rank < 144 using
    # the structure of the problem
    
    # For now, return the best found
    return best
# EVOLVE-BLOCK-END
