"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    """Build the known rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    
    # Test with random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B  # shape (n, p)
        
        # Compute using decomposition
        # C[i,k] = sum_r U[i*? + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # U index: i*m + j (A[i,j])
        # V index: j*p + k (B[j,k])
        # W index: k*n + i (C[k,i] in trace convention, i.e., C[i,k])
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _try_rank_reduction(base_result, target_rank, time_limit=100):
    """Try to reduce rank by merging/removing terms via linear algebra."""
    n, m, p = DIMENSIONS
    U0 = np.array(base_result['U'], dtype=float)
    V0 = np.array(base_result['V'], dtype=float)
    W0 = np.array(base_result['W'], dtype=float)
    rank0 = U0.shape[1]
    
    start_time = time.time()
    
    # Try to find a lower-rank decomposition by looking for linear dependencies
    # or by using a rank-reduction algorithm
    
    # Strategy: try to express the tensor as a sum of fewer rank-1 terms
    # by finding a rank-r factorization that approximates the tensor
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A[i//m, i%m], j indexes B[j//p, j%p], k indexes C[k%n, k//n]
    
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank0):
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T[i, j, k] += U0[i, r] * V0[j, r] * W0[k, r]
    
    # Now try to find a rank-r decomposition for r < rank0
    # This is NP-hard in general, but we can try heuristic approaches
    
    # Approach: use the fact that the tensor has a specific structure
    # For M_{2,2,8}, each output C[i,k] depends on only 2 A entries and 2 B entries
    
    # Let's try a structured search: assume the decomposition has the form
    # where U vectors come from a small set of patterns
    
    # For now, try random restarts with rank = target_rank
    best_result = None
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a rank-r decomposition using alternating least squares
        # or a structured approach
        
        # Structured approach: for M_{2,2,8}, try rank-24 construction
        # where we use 6 terms per 2-column block but share 1 term between blocks
        
        if target_rank == 24:
            # Try the "shared Strassen" construction
            # For blocks q=0,1,2,3 (columns 2q, 2q+1):
            # Standard Strassen per block uses 7 terms
            # Can we share? The A-side coefficients are the same across blocks
            # So terms like (A00+A11)*X_q where X_q depends on block q
            # These are distinct rank-1 terms unless X_q are linearly dependent
            
            # Alternative: use a 2x2x4 block decomposition
            # M_{2,2,4} has rank 14 (2 Strassen blocks)
            # M_{2,2,8} = M_{2,2,4} + M_{2,2,4} shifted, so rank <= 28
            # But can we do better?
            
            # Try: use 3 "super-Strassen" blocks
            # Block 0: columns 0,1 -> 7 terms
            # Block 1: columns 2,3 -> 7 terms  
            # Block 2: columns 4,5 -> 7 terms
            # Block 3: columns 6,7 -> 7 terms
            # Total: 28
            
            # To get 24, we need to save 4 terms
            # Idea: merge some terms across blocks
            
            # Let's try a different structure:
            # Use 4 terms for "diagonal" contributions and 20 for "off-diagonal"
            # This is getting too speculative
            
            pass
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, build the known rank-28 solution
    result_28 = _build_strassen_28()
    
    # Verify it works
    if not _verify_decomposition(result_28):
        # Fallback: schoolbook rank-32
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # Check time budget
    elapsed = time.time() - start_time
    remaining = 140 - elapsed  # 10s safety margin
    
    if remaining > 5:
        # Try rank 24
        best = _try_rank_reduction(result_28, 24, time_limit=min(remaining - 2, 100))
        if best is not None and _verify_decomposition(best):
            return best
    
    # Return the rank-28 solution
    return result_28
# EVOLVE-BLOCK-END
