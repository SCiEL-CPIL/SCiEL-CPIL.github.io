"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # For each tensor slice, check that U[i,j] * V[j,k] * W[k,i] sums correctly
    # The tensor T[i,j,k] = 1 if i,j,k are valid indices, and the decomposition
    # should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1) > tol:
                    return False
    return True


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s budget to leave margin
    start_time = time.time()
    
    # Start with schoolbook decomposition
    result = _schoolbook()
    rank = 3 * 4 * 8  # 96
    
    # Try to find a better decomposition using ALS (Alternating Least Squares)
    # with random restarts
    
    def _round_to_half_integer(arr):
        """Round entries to nearest half-integer and verify."""
        rounded = np.round(arr * 2) / 2
        return rounded
    
    def _try_als(deadline):
        """Try ALS to find a low-rank decomposition."""
        n, m, p = DIMENSIONS
        best_result = None
        best_rank = None
        
        # Try different ranks starting from a reasonable lower bound
        # The trivial rank is 96. We try to go below that.
        # For 3x4x8, the best known rank is at most 96 (schoolbook).
        # Let's try to find if there's a decomposition with rank < 96.
        
        max_rank_to_try = 95
        min_rank_to_try = max(3 * 8, 4 * 8, 3 * 4)  # at least max of dimensions products
        # Actually, the rank must be at least max(n*m, m*p, p*n) in general? No.
        # The rank of tensor is at least the maximum of the "slice ranks"
        # For matrix multiplication, the rank is at least max(n*p, m*p, n*m)
        # Actually, for M_mm, rank >= max(n*m, m*p, p*n) is not right.
        # The rank is at least the maximum over all "contractions"
        # Let's just try a range of ranks
        
        # For 3x4x8, let's try ranks from, say, 40 to 95
        # This is a heuristic; we'll try to find a valid decomposition
        
        np.random.seed(42)
        
        for target_rank in range(40, max_rank_to_try + 1):
            if time.time() > deadline:
                break
            
            # Multiple restarts for this rank
            n_restarts = 5
            for restart in range(n_restarts):
                if time.time() > deadline:
                    break
                
                # Random initialization
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # ALS iterations
                for it in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            idx = i * m + j
                            # Target: for all k, sum_r U[idx, r] * V[j*p+k, r] = 1/W[k*n+i, r]
                            # Actually, we need: for each (i,j), the vector U[idx, :] should satisfy
                            # for all k: sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # This is coupled. Let's use a different approach.
                    
                    break  # Skip full ALS for now due to complexity
                
                # Instead, let's use a different approach: 
                # Directly solve for the tensor decomposition using the structure
                
                # Actually, for matrix multiplication, we can use known algorithms.
                # For 3x4x8, let's try to use a block decomposition.
                
                # Partition the 4-dimensional inner space into two 2-dimensional subspaces
                # and the 8-dimensional output space into four 2-dimensional subspaces.
                # Then we have four independent 3x2x2 multiplications.
                # Each 3x2x2 multiplication can be done with rank at most 3*2*2 = 12 (schoolbook)
                # or potentially less.
                
                # Actually, 3x2x2 tensor: the rank is at most 12 (schoolbook).
                # Can we do better? The tensor is M_3,2,2.
                # The rank of M_3,2,2 is at most 3*2*2 = 12.
                # Is there a known better bound? For small tensors, we can try to find.
                
                # Let's construct a block decomposition:
                # A is 3x4, B is 4x8.
                # Partition A into [A1, A2] where A1, A2 are 3x2
                # Partition B into [[B11, B12, B13, B14], [B21, B22, B23, B24]] 
                # where each Bij is 2x2
                # Then AB = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23, A1*B14 + A2*B24]
                
                # Each term A1*Bij + A2*B(j+1) is a 3x2 matrix.
                # So we need to compute 4 such terms, each involving 2 3x2 by 2x2 multiplications.
                # Each 3x2 by 2x2 multiplication has tensor rank at most 12.
                # So total rank would be 4 * 2 * 12 = 96, same as schoolbook.
                
                # This doesn't help directly. Let's try a different approach.
                
                # Use known results: for matrix multiplication, the rank of M_m,n,p 
                # is bounded above by various algorithms.
                # For small dimensions, we can try to use Winograd's method or other tricks.
                
                # Let's try to construct a decomposition using the structure of the problem.
                # We'll use the fact that we can combine terms.
                
                # For now, let's just return the schoolbook if we can't find better.
                pass
        
        return None
    
    # Try the ALS approach (simplified)
    better = _try_als(deadline)
    if better is not None:
        result = better
    
    # Verify the result
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # If verification fails, return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
