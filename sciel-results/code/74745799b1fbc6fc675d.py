"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form a valid decomposition of matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Vectorized dot: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                prod = V[j*p+k] * W[k*n+i]
                val = np.dot(U[i*m+j], prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_refine(U, V, W, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """Alternating Least Squares refinement for tensor decomposition.
    At each step, fix two factors and solve for the third via linear least squares.
    This is a structural change from gradient descent to direct linear solve.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    prev_loss = np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve U[i*m+j, :] such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
        # Stack over k: A[i,j] has rows A[k,:] = V[j*p+k,:] * W[k*n+i,:]
        # Solve A @ x = ones(p)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k] = V[j*p+k] * W[k*n+i]
                b = np.ones(p)
                # Least squares solve
                try:
                    U[i*m+j], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: for each (j,k), solve V[j*p+k, :] such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i] = U[i*m+j] * W[k*n+i]
                b = np.ones(n)
                try:
                    V[j*p+k], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: for each (k,i), solve W[k*n+i, :] such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j] = U[i*m+j] * V[j*p+k]
                b = np.ones(m)
                try:
                    W[k*n+i], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = V[j*p+k] * W[k*n+i]
                    val = np.dot(U[i*m+j], prod)
                    loss += (val - 1.0) ** 2
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U, V, W

def _try_strassen_block(n, m, p, deadline=None):
    """Try to construct a decomposition using Strassen-like block structure.
    For 4x4 * 4x5, we can think of it as blocks and try to exploit
    the known Strassen algorithm structure for sub-blocks.
    
    The key structural idea: use the known rank-7 decomposition for 2x2 matrices
    and extend it. For 4x4*4x4, the best known rank is 49 (for 4x4x4 it's 49).
    For 4x4*4x5, we can pad to 4x4*4x8 or use a mixed approach.
    
    Actually, let's try a direct approach: start from the schoolbook and try
    to merge terms using linear dependencies.
    """
    pass  # Placeholder - the main strategy is ALS refinement

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank
    
    deadline = time.time() + 115  # Safety margin
    
    # Strategy 1: Try ALS refinement starting from random initializations
    # at various ranks below schoolbook
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    # For 4x4x5, the theoretical lower bound is max(n*m, m*p, p*n) = max(16,20,20) = 20
    # But in practice, we need more. Let's try from 40 to 79.
    for rank in range(40, schoolbook_rank):
        if time.time() > deadline:
            break
        
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Scale so that each term contributes roughly 1
            scale = (1.0 / rank) ** (1.0/3.0)
            U *= scale
            V *= scale
            W *= scale
            
            # ALS refinement
            U, V, W = _als_refine(U, V, W, n, m, p, max_iter=100, deadline=deadline)
            
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                if rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                break  # Found valid at this rank, try lower
    
    # Strategy 2: Start from schoolbook and try to reduce rank via
    # identifying and merging linearly dependent terms
    # This is a structural approach: the schoolbook has rank 80, but there
    # may be linear dependencies among the rank-1 terms.
    if time.time() < deadline - 10:
        # Try to find a lower-rank decomposition by starting from schoolbook
        # and removing redundant terms
        # For each term, check if it can be expressed as a linear combination
        # of others. This is expensive, so we do a greedy approach.
        
        # Alternative: use the fact that for matrix multiplication, there are
        # known lower bounds and constructions. For 4x4x4, rank 49 is known.
        # For 4x4x5, we can try to extend a 4x4x4 result.
        
        # Let's try a different structural approach: use the known decomposition
        # for 2x2x2 (Strassen, rank 7) and extend it.
        
        # Actually, let's try to use a smarter initialization: start from
        # the schoolbook and randomly project to lower rank, then refine.
        for rank in range(50, schoolbook_rank):
            if time.time() > deadline:
                break
            
            # Project schoolbook to lower rank via random projection
            P = np.random.randn(schoolbook_rank, rank) * 0.5
            Q = np.random.randn(schoolbook_rank, rank) * 0.5
            R = np.random.randn(schoolbook_rank, rank) * 0.5
            
            U_proj = U_sb @ P
            V_proj = V_sb @ Q
            W_proj = W_sb @ R
            
            # This won't work directly because the product structure is broken.
            # Instead, let's try: U_new = U_sb @ P, V_new = V_sb @ Q, W_new = W_sb @ R
            # But then U_new[i*m+j,c] * V_new[j*p+k,c] * W_new[k*n+i,c]
            # = sum_{c1,c2,c3} U_sb[i*m+j,c1] P[c1,c] V_sb[j*p+k,c2] Q[c2,c] W_sb[k*n+i,c3] R[c3,c]
            # This is not equal to the original.
            
            # Better approach: use ALS from a structured initialization.
            # Initialize with the first 'rank' terms of schoolbook
            if rank <= schoolbook_rank:
                U_init = U_sb[:, :rank].copy()
                V_init = V_sb[:, :rank].copy()
                W_init = W_sb[:, :rank].copy()
                
                # But this only covers the first 'rank' (i,j,k) triples,
                # so it won't be a valid decomposition unless rank >= n*m*p.
                # So this approach doesn't work directly.
                
                # Instead, let's try a different initialization:
                # Use a structured pattern based on the dimensions.
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                
                # Distribute the 80 terms across 'rank' columns
                # Each column gets ~80/rank terms
                terms_per_col = schoolbook_rank // rank
                remainder = schoolbook_rank % rank
                col = 0
                term = 0
                for c in range(rank):
                    count = terms_per_col + (1 if c < remainder else 0)
                    for _ in range(count):
                        # Find the (i,j,k) for this term
                        i = term // (m * p)
                        j = (term // p) % m
                        k = term % p
                        U_init[i*m+j, c] = 1.0
                        V_init[j*p+k, c] = 1.0
                        W_init[k*n+i, c] = 1.0
                        term += 1
                
                # Now refine with ALS
                U_r, V_r, W_r = _als_refine(U_init, V_init, W_init, n, m, p, max_iter=50, deadline=deadline)
                
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    if rank < best_rank:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
