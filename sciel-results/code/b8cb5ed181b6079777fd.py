"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum for fast computation
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=1e-6)

def _verify_slow(U, V, W, n, m, p):
    """Slow but reliable verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2_block_decomposition():
    """
    Build a decomposition for 4x5 * 5x5 using block structure.
    
    We partition:
    A (4x5) into 2x2 blocks: A = [[A11, A12], [A21, A22]] where Aij are 2x(5/2)
    But 5 is odd, so we pad or use different partitioning.
    
    Alternative: Use the fact that 4x5x5 can be seen as multiple 2x2x2 multiplications.
    
    Let's use a simpler approach: exploit that we can compute 4x5*5x5 by splitting
    the 5 columns into 2+3, and 5 rows into 2+3.
    
    Actually, let's use a known result: the rank of 4x5x5 matrix multiplication.
    
    For now, let's try to construct a decomposition with rank < 100 using
    a structured approach based on separating the dimensions.
    """
    n, m, p = 4, 5, 5
    
    # We'll try to use a decomposition that exploits the structure.
    # One approach: use the fact that matrix multiplication is bilinear.
    
    # Let's try a rank-40 decomposition based on the following idea:
    # Split the 5x5 matrix into blocks and use a reduced set of products.
    
    # Actually, let's use a simpler heuristic: try to find a decomposition
    # with rank 40 by using the structure of the problem.
    
    # For a 4x5 by 5x5 multiplication, the tensor has 4*5*5 = 100 entries.
    # The minimum rank is at least max(4*5, 5*5, 5*4) = max(20, 25, 20) = 25.
    
    # Let's try to construct a rank-25 or rank-30 decomposition.
    
    # Approach: use the "entry-wise" structure but with some sharing.
    
    # For each output entry (i,k), we need sum_j A[i,j]*B[j,k] = 1
    # This means for each (i,k), the coefficient of A[i,j]*B[j,k] should be 1
    # and all other products should have coefficient 0.
    
    # A rank-r decomposition means we have r terms, each term is a product
    # of a vector in A-space, a vector in B-space, and a vector in C-space.
    
    # Let's try to construct a decomposition with rank 25.
    
    # One idea: for each j (column of A / row of B), we can have 5 terms
    # that handle all (i,k) pairs for that j. But that would be 5*5 = 25 terms.
    
    # Actually, let's think about it differently.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Wait, that's not right. The tensor is T[i,j,k] which represents the coefficient
    # of A[i,j] * B[j,k] in C[i,k]. So T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For the schoolbook, we have r = n*m*p = 100, with U[i,j,c]=V[j,k,c]=W[k,i,c]=1
    # only when (i,j,k) corresponds to term c.
    
    # To get lower rank, we need to find U,V,W with fewer columns such that
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the 3-tensor T.
    
    # Let's try a simple approach: use the fact that we can factor through the middle dimension.
    
    # For each j, we can write the contribution as:
    # sum_j (sum_i A[i,j] * something) * (sum_k B[j,k] * something)
    
    # Actually, let's just try to use ALS with a better initialization.
    
    # For now, return None to indicate we need to use ALS.
    return None

def _als_decomposition(n, m, p, rank, iters=50, seed=0):
    """Attempt to find a decomposition of given rank using ALS."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(iters):
        # Update U: for each (i,j), solve U[i,j,:] @ M[i,j] = ones(p)
        # where M[i,j][c,k] = V_r[j,k,c] * W_r[k,i,c]
        for i in range(n):
            for j in range(m):
                # M[c,k] = V_r[j,k,c] * W_r[k,i,c]
                # V_r[j] is (p, rank), W_r[:,i] is (p, rank)
                M = (V_r[j, :, :] * W_r[:, i, :]).T  # (rank, p)
                try:
                    U_r[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve V[j,k,:] @ B[j,k] = ones(n)
        # where B[j,k][c,i] = U_r[i,j,c] * W_r[k,i,c]
        for j in range(m):
            for k in range(p):
                # B[c,i] = U_r[i,j,c] * W_r[k,i,c]
                # U_r[:,j] is (n, rank), W_r[k] is (n, rank)
                B = (U_r[:, j, :] * W_r[k, :, :]).T  # (rank, n)
                try:
                    V_r[j, k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve W[k,i,:] @ C[k,i] = ones(m)
        # where C[k,i][c,j] = U_r[i,j,c] * V_r[j,k,c]
        for k in range(p):
            for i in range(n):
                # C[c,j] = U_r[i,j,c] * V_r[j,k,c]
                # U_r[i] is (m, rank), V_r[:,k] is (m, rank)
                C = (U_r[i, :, :] * V_r[:, k, :]).T  # (rank, m)
                try:
                    W_r[k, i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Resync
        U = U_r.reshape(n * m, rank)
        V = V_r.reshape(m * p, rank)
        W = W_r.reshape(p * n, rank)
    
    # Verify
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # But for exact integer decompositions, likely higher
    
    # First, try a few structured approaches
    # Then use ALS with various ranks and seeds
    
    # Try a range of ranks, starting from lower bounds
    ranks_to_try = list(range(25, 60)) + list(range(60, 100, 2))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple seeds
        max_seeds = 10 if target_rank < 40 else 5
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            result = _als_decomposition(n, m, p, target_rank, iters=30, seed=seed)
            if result is not None:
                # Attempt to round to integer/half-integer
                U_rounded = [[round(2 * x) / 2 for x in row] for row in result['U']]
                V_rounded = [[round(2 * x) / 2 for x in row] for row in result['V']]
                W_rounded = [[round(2 * x) / 2 for x in row] for row in result['W']]
                
                # Verify the rounded version
                if _verify_slow(U_rounded, V_rounded, W_rounded, n, m, p):
                    if target_rank < best_rank:
                        best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        if best_rank <= 30:
                            return best
                else:
                    # Try the unrounded version if it's close
                    if _verify_slow(result['U'], result['V'], result['W'], n, m, p):
                        # Check if values are close to half-integers
                        all_close = True
                        for mat in [result['U'], result['V'], result['W']]:
                            for row in mat:
                                for x in row:
                                    if abs(x * 2 - round(x * 2)) > 1e-4:
                                        all_close = False
                                        break
                            if not all_close:
                                break
                        if all_close:
                            if target_rank < best_rank:
                                best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                                best_rank = target_rank
    
    # If we found something better, use it
    if best_rank < n * m * p:
        return best
    
    # Otherwise, return schoolbook
    return best
# EVOLVE-BLOCK-END
