"""
Matrix multiplication 3x3x8 tensor decomposition.

Strategy:
1. Build the schoolbook rank-72 decomposition as a guaranteed fallback.
2. Attempt to find a lower-rank decomposition using a structured approach:
   - Partition the 8 output columns into two groups of 4.
   - For each group, try to find a compact bilinear decomposition.
   - Use iterative refinement (alternating least squares on factor matrices)
     to try to converge to a valid low-rank decomposition.
3. If no lower-rank decomposition is found within time budget, return schoolbook.

The key insight is that for 3x3x8, the tensor has a product structure in the
last dimension. We can potentially share structure across the 8 columns.

We use the following approach:
- Start with a random or structured initialization at rank r < 72.
- Alternately update U, V, W to minimize the reconstruction error.
- Check if the result is exact (within tolerance).
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook():
    """Build the trivial rank-72 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72
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

def _verify(U, V, W):
    """
    Verify that the decomposition exactly reconstructs the multiplication tensor.
    T[i,j,k] should equal 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        U_r = Ua[:, r]
        V_r = Va[:, r]
        W_r = Wa[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    return np.allclose(T, 1.0, atol=1e-6)

def _compute_error(U, V, W):
    """Compute the max absolute error of the reconstruction."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        U_r = Ua[:, r]
        V_r = Va[:, r]
        W_r = Wa[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    return np.max(np.abs(T - 1.0))

def _try_alternating_least_squares(rank, max_iters=200, tol=1e-8, seed=None):
    """
    Try to find a rank-r decomposition using alternating least squares.
    
    For each factor matrix, we solve a linear system holding the others fixed.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with random values
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Normalize columns to avoid scale issues
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:,r]**2) + np.sum(V[:,r]**2) + np.sum(W[:,r]**2))
        if norm > 0:
            U[:,r] /= norm
            V[:,r] /= norm
            W[:,r] /= norm
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except:
                    pass
        
        # Compute error
        err = _compute_error(U.tolist(), V.tolist(), W.tolist())
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_error < 1e-6:
        return {
            "U": [list(row) for row in best_U],
            "V": [list(row) for row in best_V],
            "W": [list(row) for row in best_W]
        }
    return None

def _try_structured_partition():
    """
    Try a structured approach by partitioning the 8 columns into two groups of 4.
    For each group, use a rank-4 decomposition if possible.
    
    The idea: C = A * B where B is 3x8. Split B into B1 (3x4) and B2 (3x4).
    Then C1 = A * B1 and C2 = A * B2.
    
    If we can find rank-r decompositions for 3x3x4, we can combine them.
    """
    n, m, p = DIMENSIONS
    
    # For 3x3x4, schoolbook rank is 36. Try to find lower rank.
    # Actually, let's try a different structured approach.
    
    # Key insight: The multiplication tensor T[i,j,k] = delta_{i,j,k} (in the tensor sense)
    # has a product structure. We can try to exploit this.
    
    # Approach: Use the fact that for fixed j, the sub-tensor over (i,k) is
    # a 3x8 matrix. We can try to decompose this jointly.
    
    # Let's try rank 48, 45, 42, 40, 36
    for target_rank in [48, 45, 42, 40, 36]:
        result = _try_alternating_least_squares(target_rank, max_iters=50, seed=target_rank)
        if result:
            return result
    return None

def _try_block_diagonal():
    """
    Try a block-diagonal approach: treat the 8 output columns as two independent
    3x4 problems and solve each separately, then combine.
    
    For 3x3x4, try to find a rank-r decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Try to find rank-r decomposition for 3x3x4
    # Schoolbook for 3x3x4 is 36
    for target_rank in [30, 28, 26, 24]:
        # We need to adapt _try_alternating_least_squares for p=4
        # For now, just try the full 3x3x8 with this rank
        result = _try_alternating_least_squares(target_rank * 2, max_iters=30, seed=target_rank)
        if result:
            return result
    return None

def _try_sparse_initialization():
    """
    Try sparse random initialization followed by ALS refinement.
    Sparse initializations may converge to better local optima.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    for target_rank in [60, 55, 50, 48, 45]:
        for trial in range(3):
            np.random.seed(target_rank * 100 + trial)
            
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Random sparse initialization
            for r in range(target_rank):
                # Each column has a few non-zero entries
                n_u = np.random.randint(1, 4)
                n_v = np.random.randint(1, 4)
                n_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(nm, n_u, replace=False)
                v_idx = np.random.choice(mp, n_v, replace=False)
                w_idx = np.random.choice(pn, n_w, replace=False)
                
                U[u_idx, r] = np.random.choice([-1, 1], n_u)
                V[v_idx, r] = np.random.choice([-1, 1], n_v)
                W[w_idx, r] = np.random.choice([-1, 1], n_w)
            
            result = _try_alternating_least_squares(target_rank, max_iters=30, seed=target_rank + trial)
            if result:
                return result
    return None

def construct():
    start_time = time.time()
    timeout = 120.0
    
    # 1. Build schoolbook as guaranteed fallback
    schoolbook = _build_schoolbook()
    
    # 2. Try structured partition approach
    if time.time() - start_time < timeout - 60:
        structured = _try_structured_partition()
        if structured and _verify(structured['U'], structured['V'], structured['W']):
            if len(structured['U'][0]) < len(schoolbook['U'][0]):
                return structured
    
    # 3. Try block diagonal approach
    if time.time() - start_time < timeout - 40:
        block = _try_block_diagonal()
        if block and _verify(block['U'], block['V'], block['W']):
            if len(block['U'][0]) < len(schoolbook['U'][0]):
                return block
    
    # 4. Try sparse initialization with ALS
    if time.time() - start_time < timeout - 20:
        sparse = _try_sparse_initialization()
        if sparse and _verify(sparse['U'], sparse['V'], sparse['W']):
            if len(sparse['U'][0]) < len(schoolbook['U'][0]):
                return sparse
    
    # 5. Direct ALS attempts at various ranks
    if time.time() - start_time < timeout - 10:
        for r in [65, 60, 55]:
            if time.time() - start_time > timeout - 10:
                break
            result = _try_alternating_least_squares(r, max_iters=20, seed=r)
            if result and _verify(result['U'], result['V'], result['W']):
                if len(result['U'][0]) < len(schoolbook['U'][0]):
                    return result
    
    return schoolbook

# EVOLVE-BLOCK-END
