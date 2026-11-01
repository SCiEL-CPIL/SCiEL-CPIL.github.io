"""
Strategy: Global Tensor Rank Minimization via Alternating Least Squares (ALS) with Exact Rounding.

The current program uses a block-decomposition strategy (Strassen blocks), which is structurally
limited to ranks that are sums of block ranks (e.g., multiples of 7). This new strategy treats the
matrix multiplication tensor T_{ijk} as a single object and optimizes a rank-R decomposition
T = sum_r u_r v_r w_r directly.

Method:
1. Initialize U, V, W with a valid schoolbook decomposition (rank 48) or random noise.
2. Use Alternating Least Squares (ALS) to minimize the Frobenius norm ||T - sum_r u_r v_r w_r||^2
   for a target rank R.
3. After convergence (or time limit), round the entries of U, V, W to the nearest integer or half-integer.
4. Verify if the rounded decomposition is exact (integer arithmetic check).
5. If exact, return. If not, try a lower R or different initialization.

We start from R=36 (known achievable via 2x3x2 blocks) and try to go lower.
The ALS step is performed in floating point, but the final verification is in exact arithmetic
(using fractions or high-precision integers if necessary, but numpy float64 is usually enough
for verification if the solution is well-conditioned or integers are small).
"""

import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i,j,k are valid indices for matrix multiplication
    # T is the "matrix multiplication tensor"
    # Shape: (n, m, p)
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Reshape T to matrix form for ALS
    # We want to solve: T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # This is a 3-way tensor.
    # ALS updates:
    # U = (T_x1 * (V W^T)) / ((V W^T) * (V W^T)^T)
    # V = (T_x2 * (U W^T)) / ((U W^T) * (U W^T)^T)
    # W = (T_x3 * (U V^T)) / ((U V^T) * (U V^T)^T)
    # Where T_x1 is the matricization of T along mode 1, etc.
    
    # Matricizations:
    # T_x1: shape (n, m*p). T_x1[i, j*p+k] = T[i, j, k]
    # T_x2: shape (m, n*p). T_x2[j, i*p+k] = T[i, j, k]
    # T_x3: shape (p, n*m). T_x3[k, i*m+j] = T[i, j, k]
    
    T_x1 = T.transpose(0, 1, 2).reshape(n, m * p)
    T_x2 = T.transpose(1, 0, 2).reshape(m, n * p)
    T_x3 = T.transpose(2, 0, 1).reshape(p, n * m)
    
    def matmul_3way(U, V, W):
        """Compute tensor from factors."""
        # U: (n, R), V: (m, R), W: (p, R)
        # Tensor[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Efficient: (U V^T) is (n, m). Then multiply with W?
        # Actually, we can compute T_x1 = U * (V W^T).
        # V W^T is (m, p).
        # So T_x1 = U @ (V @ W.T)
        T_x1_rec = U @ (V @ W.T)
        return T_x1_rec.reshape(n, m, p)

    def check_exactness(U, V, W, T):
        """Check if decomposition is exact within floating point tolerance."""
        # Compute T_rec
        T_rec = matmul_3way(U, V, W)
        diff = np.abs(T_rec - T)
        return np.max(diff) < 1e-9

    def round_to_half_ints(M):
        """Round matrix to nearest half-integer."""
        return np.round(M * 2) / 2.0

    def try_rank(R, init_type='schoolbook', iterations=500):
        """
        Attempt to find a rank-R decomposition.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline - 10:
            return None
            
        # Initialize
        if init_type == 'schoolbook':
            # Construct schoolbook rank n*m*p
            # U[i*m+j, r] = 1 if r == i*m+j
            # V[j*p+k, r] = 1 if r == j*p+k
            # W[k*n+i, r] = 1 if r == k*n+i
            # This is rank n*m*p. If R < n*m*p, we can't use this directly.
            # Instead, we use random initialization for R < n*m*p.
            # Or we can use a block structure initialization if R matches.
            # For now, random init is more general.
            pass
            
        # Random Initialization
        # Scale to match norm of T
        norm_T = np.linalg.norm(T)
        scale = norm_T / (R * np.sqrt(n*m*p)) # Rough heuristic
        U = np.random.randn(n, R) * scale
        V = np.random.randn(m, R) * scale
        W = np.random.randn(p, R) * scale
        
        # ALS Optimization
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T_x1 = U @ (V @ W.T)
            # U_new = T_x1 @ (V @ W.T).T / ((V @ W.T) @ (V @ W.T).T)
            VW_t = V @ W.T  # (m, p)
            VW_t_T = VW_t.T # (p, m)
            # Denominator: (V W^T) (V W^T)^T = V (W W^T) V^T
            # Or simply (VW_t @ VW_t_T) is (m, m)? No.
            # We need to solve: U * (V W^T) = T_x1
            # Let M = V @ W.T (m, p)
            # U M = T_x1
            # U = T_x1 M^T (M M^T)^{-1}
            M = V @ W.T
            MM_t = M @ M.T
            # Add regularization
            MM_t += 1e-8 * np.eye(m)
            U = T_x1 @ M.T @ np.linalg.inv(MM_t)
            
            # Update V
            # T_x2 = V @ (W @ U.T)
            # Let M = W @ U.T (p, n)
            # V M = T_x2
            M = W @ U.T
            MM_t = M @ M.T
            MM_t += 1e-8 * np.eye(p)
            V = T_x2 @ M.T @ np.linalg.inv(MM_t)
            
            # Update W
            # T_x3 = W @ (U @ V.T)
            # Let M = U @ V.T (n, m)
            # W M = T_x3
            M = U @ V.T
            MM_t = M @ M.T
            MM_t += 1e-8 * np.eye(n)
            W = T_x3 @ M.T @ np.linalg.inv(MM_t)
            
            # Check convergence (optional, just check time)
            
        # Round to half-integers
        U_r = round_to_half_ints(U)
        V_r = round_to_half_ints(V)
        W_r = round_to_half_ints(W)
        
        # Verify exactness
        if check_exactness(U_r, V_r, W_r, T):
            # Convert to list of lists
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
        # Try rounding to integers only?
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exactness(U_i, V_i, W_i, T):
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
            
        return None

    # List of ranks to try, from low to high
    # Known lower bound is roughly max(n*m, m*p, p*n) / something?
    # Actually, rank >= max(n,m,p) is a weak bound.
    # Rank 24 is a common lower bound estimate for small matrices.
    # We know 42 is possible. 36 is possible.
    # Let's try 36, 35, 34, 33, 32, 31, 30...
    
    # We will try a few random restarts for each rank to increase chance of finding a solution.
    
    best_result = None
    
    # Fallback: Schoolbook rank 48
    # This is guaranteed to work.
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
                idx += 1
                
    # We will return schoolbook if nothing better is found.
    # But we want to find something better.
    
    # Try ranks
    # 36 is a known achievable rank (2x3x2 blocks).
    # Let's try 36 first with a few restarts.
    # Then 35, 34, etc.
    
    ranks_to_try = [36, 35, 34, 33, 32, 31, 30, 29, 28]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try 5 random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            res = try_rank(R, iterations=1000)
            if res:
                U, V, W = res
                # Verify one last time
                if check_exactness(np.array(U), np.array(V), np.array(W), T):
                    return {"U": U, "V": V, "W": W}
                    
        # If we found a valid decomposition for R, we return it.
        # The loop above returns immediately if found.
        
    # If we get here, we didn't find a better rank.
    # Return schoolbook.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
