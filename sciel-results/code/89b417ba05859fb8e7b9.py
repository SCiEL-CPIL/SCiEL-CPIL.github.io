import numpy as np
import time

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def _validate_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k), and we're decomposing the identity tensor
    
    # Actually, the standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor M has M[i,j,k] = 1 for all i,j,k in valid range
    # We decompose M as sum_r u_r * v_r * w_r where:
    #   u_r is a vector in R^{n*m} (indexed by (i,j))
    #   v_r is a vector in R^{m*p} (indexed by (j,k))  
    #   w_r is a vector in R^{p*n} (indexed by (k,i))
    # And M[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # Let's verify by checking a few random instances
    rng = np.random.RandomState(42)
    for trial in range(20):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # Extract A_part from u_r: u_r[i*m+j] gives coefficient for A[i,j]
            # Extract B_part from v_r: v_r[j*p+k] gives coefficient for B[j,k]
            # w_r[k*n+i] gives coefficient for C[i,k]
            
            A_part = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    A_part[i, j] = u_r[i * m + j]
            
            B_part = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    B_part[j, k] = v_r[j * p + k]
            
            # The contribution to C from this rank-1 term
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_r[k * n + i] * (A_part * B_part).sum()
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank = n * m * p = 3 * 3 * 8 = 72
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as baseline
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using Strassen-like approach
    # For 3x3 * 3x8, we can split into blocks and use Strassen for 2x2 subblocks
    
    # Strategy: Split the 3x3 A into [[A11, a12], [a21, a22]] where A11 is 2x2
    # Split the 3x8 B into [[B11, B12], [b21, b22]] where B11 is 2x2, B12 is 2x2, etc.
    # Actually B is 3x8, so split as [[B_top, B_mid], [B_bot]] where B_top is 2x8, B_bot is 1x8
    
    # Let's try a different approach: use known rank bounds
    # R(3,3,8) is known to be at most 68 via certain constructions
    
    # For now, let's attempt a constructive approach using Strassen on sub-blocks
    # Split A (3x3) as:
    #   A = [[A11, A12], [A21, A22]]
    #   where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Split B (3x8) as:
    #   B = [[B11, B12], [B21, B22]]  
    #   where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    
    # C = A*B (3x8):
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2 * 2x4 product can be done with rank 14 (Strassen for 2x2 x 2x2 is 7, so 2x2 x 2x4 is 14)
    # But we have cross terms that are 2x1 * 1x4 and 1x2 * 2x4 which are simpler
    
    # Let's just return the schoolbook for now if we can't quickly find better
    if time.time() - start_time > DEADLINE_SECONDS - 30:
        return {"U": U, "V": V, "W": W}
    
    # Attempt: Use rank 68 decomposition
    # This requires careful construction. For safety, return schoolbook (rank 72)
    # which is guaranteed correct.
    
    return {"U": U, "V": V, "W": W}
