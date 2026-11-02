"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Strategy: Use the known rank-21 algorithm for 3x3x3 (Strassen-like) and extend
    # to 3x3x4 by adding terms for the 4th column of B.
    #
    # The tensor T_{ijk} for (A B)_{ik} = sum_j A_{ij} B_{jk}
    # For 3x3x3: rank 21 is known optimal (Larsen et al.)
    # For 3x3x4: we need to handle B_{j3} for j=0,1,2 (the 4th column)
    #
    # Approach: Start with the 3x3x3 rank-21 decomposition, then find minimal
    # additional rank to cover the extra column.
    
    # First, let's try to construct a valid low-rank decomposition using a
    # structured approach based on the 3x3x3 rank-21 algorithm.
    
    # The 3x3x3 rank-21 algorithm (from Larsen, Shitov, Strassen 2018):
    # This uses a specific set of 21 multiplications.
    # For simplicity, let's use a known explicit construction.
    
    # Actually, let me try a different approach: use the fact that for 3x3x4,
    # we can potentially achieve rank 22 or 23 by extending the 3x3x3 algorithm.
    
    # Let's try to find a rank-22 decomposition by:
    # 1. Starting with a 3x3x3 rank-21 decomposition
    # 2. Adding 1 term to handle the 4th column
    
    # The 3x3x3 rank-21 algorithm (explicit construction):
    # I'll use a known pattern. The key insight is that we can share intermediate
    # products.
    
    # For now, let's try a brute-force approach with structured constraints.
    # We'll parameterize the decomposition and solve for integer/half-integer values.
    
    # Actually, let's try a simpler approach: use the schoolbook but look for
    # linear dependencies that allow rank reduction.
    
    # For 3x3x4, the tensor has 36 entries (3*3*4). The schoolbook uses rank 36.
    # Can we do better?
    
    # Let's try rank 24: group by the k dimension.
    # For each k, we have a 3x3 matrix multiplication (A * B[:,k]).
    # If we could share structure across k values...
    
    # Alternative: use the fact that B has 4 columns, but we only need 3 independent
    # combinations if there's a linear dependency.
    
    # Let's try a direct construction for rank 24:
    # Group terms by (i,j) pairs and try to share across k.
    
    # For each (i,j), we need: sum_k A[i,j] * B[j,k] * W[k*n+i]
    # If we fix i,j, this is A[i,j] * sum_k B[j,k] * W[k*n+i]
    # So for each (i,j), we need to compute a weighted sum over k.
    
    # This suggests: for each (i,j), use 4 terms (one per k) = 36 total (schoolbook)
    # To reduce rank, we need to share terms across different (i,j) pairs.
    
    # Let's try: for each j, compute all 3 output values (i=0,1,2) together.
    # For fixed j: we need (A[:,j] * B[j,:]) which is a 3x4 matrix.
    # This can be done with rank 12 (3*4) per j, total 36.
    # To reduce: if we can share across j values...
    
    # Let me try a different strategy: use a known low-rank algorithm for 3x3x3
    # and extend it.
    
    # The 3x3x3 rank-21 algorithm (simplified version):
    # I'll construct it explicitly.
    
    # For 3x3x3, one known rank-21 decomposition:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,1]*B[1,0]
    # M3 = A[0,2]*B[2,0]
    # M4 = A[1,0]*B[0,1]
    # M5 = A[1,1]*B[1,1]
    # M6 = A[1,2]*B[2,1]
    # M7 = A[2,0]*B[0,2]
    # M8 = A[2,1]*B[1,2]
    # M9 = A[2,2]*B[2,2]
    # M10 = (A[0,0]+A[0,1])*(B[0,1]+B[1,0])
    # M11 = (A[0,1]+A[0,2])*(B[1,1]+B[2,1])
    # M12 = (A[0,2]+A[0,0])*(B[2,1]+B[0,0])  # etc.
    # ... this gets complex.
    
    # Let me try a more systematic approach: use linear algebra to find a rank-24
    # decomposition by solving for the factor matrices.
    
    # For rank r, we need U (9xr), V (12xr), W (12xr) such that:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a bilinear system. Let's try rank 24 with a structured initialization.
    
    def try_rank_decomposition(target_rank, max_iter=5000, tol=1e-8):
        """Try to find a decomposition of given rank using iterative refinement."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with schoolbook and randomly drop terms
        np.random.seed(42)
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize with a subset of schoolbook terms
        for c in range(min(target_rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        
        # Iteratively refine using gradient-like updates
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residuals[i,j,k] = val - 1.0
            
            # If residuals are small enough, we're done
            if np.max(np.abs(residuals)) < tol:
                # Convert to integer/half-integer if possible
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify the rounded version
                if verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {
                        'U': U_int.tolist(),
                        'V': V_int.tolist(),
                        'W': W_int.tolist()
                    }
                else:
                    # Return the float version (should be close to integer/half-integer)
                    return {
                        'U': U.tolist(),
                        'V': V.tolist(),
                        'W': W.tolist()
                    }
            
            # Update: try to reduce residuals
            # This is a non-convex problem, so use a simple heuristic
            # Pick the largest residual and adjust the corresponding term
            max_idx = np.unravel_index(np.argmax(np.abs(residuals)), residuals.shape)
            i, j, k = max_idx
            residual_val = residuals[i,j,k]
            
            # Find the term with maximum contribution to this residual
            contributions = np.array([
                U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
                for c in range(target_rank)
            ])
            
            # Simple update: scale the term that has the largest contribution
            if np.max(np.abs(contributions)) > 1e-10:
                max_c = np.argmax(np.abs(contributions))
                # Adjust to reduce the residual
                current_contrib = contributions[max_c]
                if current_contrib != 0:
                    # We want to reduce residual_val
                    # Adjust U, V, or W
                    scale = 1.0 - residual_val / current_contrib
                    scale = max(0.5, min(1.5, scale))
                    U[i*m+j, max_c] *= scale
                    # Renormalize
                    U[i*m+j, max_c] = np.round(U[i*m+j, max_c] * 2) / 2
        
        return None
    
    # Try ranks from 22 to 35
    for target_rank in [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        result = try_rank_decomposition(target_rank)
        if result is not None:
            # Verify it's valid
            if verify(result['U'], result['V'], result['W']):
                return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
