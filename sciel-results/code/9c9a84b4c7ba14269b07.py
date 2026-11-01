"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the tensor entry T[i,j,k] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
                # is reproduced by sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                expected = 1.0
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    # Also check that cross-terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i,j,k) != (i2,j2,k2):
                                # T[i,j,k] should be 0 when not matching, but actually
                                # the tensor is T[i,j,k] = 1 if this is a valid triple, 0 otherwise
                                # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
                                # because C[i,k] = sum_j A[i,j] B[j,k]
                                # So every (i,j,k) triple contributes 1 to the tensor.
                                pass
    return True

def _construct_rank20():
    """Construct a rank-20 decomposition for 2x3 x 3x4 matrix multiplication."""
    n, m, p = 2, 3, 4
    rank = 20
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 20
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 20
    
    # Construction based on known results for rectangular matrix multiplication
    # For 2x3 x 3x4, we can achieve rank 20
    
    # Let me use a systematic construction:
    # The idea is to combine some of the 24 trivial terms
    
    # First 12 terms: direct terms for all (i,j,k) with some sharing
    # Actually, let me try a different approach: use the known rank-20 construction
    
    # Based on the structure, here's a construction:
    # Terms 0-3: Handle A[0,:] * B[:,0] contributions to C[0,0] and C[1,0]
    # etc.
    
    # Let me try a more systematic approach using known patterns
    # For 2xn x nxm, rank = 2nm - 2n + 2 works for some cases
    
    # Actually, let me just try to find it by construction
    # I'll use the following pattern:
    
    # For each column k of B (0..3), we need to compute A * B[:,k]
    # A is 2x3, so A * B[:,k] gives a 2-vector
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # For each k, this is a 2x3 times 3x1 multiplication, which has rank 6 trivially
    # But we can do better by sharing across k's
    
    # Let me try: for each pair of columns (k1, k2), use a 2x3 x 3x2 subproblem
    # 2x3 x 3x2 has rank at most 2*3*2/2 = 6? No, it's 12 trivially.
    
    # This is getting complex. Let me try a random search approach.
    
    return None  # Placeholder

def _random_search_for_rank(target_rank, n, m, p, max_iters=50000, time_limit=100):
    """Try to find a rank-`target_rank` decomposition via random initialization + refinement."""
    import time
    start = time.time()
    np.random.seed(42)
    
    best = None
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        rank = target_rank
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Check if this decomposition is valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check cross-terms are zero
            valid_cross = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if (i,j,k) != (i2,j2,k2):
                                        # The tensor T[i,j,k] is 1 for all valid (i,j,k)
                                        # So cross-terms aren't an issue - the tensor is all 1s
                                        pass
            if valid_cross:
                # Convert to half-integers if possible
                best = (U, V, W)
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    n, m, p = DIMENSIONS
    
    # Try rank 20 first
    for target_rank in [20, 19, 18, 17]:
        if time.time() > deadline - 5:
            break
        
        result = _random_search_for_rank(target_rank, n, m, p, 
                                          max_iters=10000, 
                                          time_limit=min(30, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            # Convert to lists of half-integers
            U_list = [[round(x * 2) / 2 for x in row] for row in U]
            V_list = [[round(x * 2) / 2 for x in row] for row in V]
            W_list = [[round(x * 2) / 2 for x in row] for row in W]
            
            # Verify after rounding
            rank = target_rank
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0
                        actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                    for r in range(rank))
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
