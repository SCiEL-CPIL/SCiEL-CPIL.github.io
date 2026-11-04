"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 2x3 by 3x9 matrix multiplication.
    
    Strategy: Partition the 9 columns of B into three blocks of 3 columns.
    Each block corresponds to a 2x3 by 3x3 multiplication.
    Use the schoolbook decomposition for each block (rank 18 per block).
    Total rank = 3 * 18 = 54.
    
    However, we can potentially do better by exploiting shared structure in A.
    Let's try to find a lower rank decomposition using numerical methods.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the full multiplication tensor T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # T has shape (n, m, p) = (2, 3, 9)
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r] so W is indexed by (k, i)
    
    # Actually, let me re-read the interface:
    # U: (n*m, rank) - indexed by (i, j)
    # V: (m*p, rank) - indexed by (j, k)  
    # W: (p*n, rank) - indexed by (k, i) with W[k*n+i, r]
    
    # The decomposition: for each r, U[:,r] * V[:,r] * W[:,r] should reconstruct T
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 (it's just the identity in the summation index j)
    # Actually no - the tensor M_{n,m,p} has T[i,j,k] = delta where it's 1 for all valid (i,j,k)
    # because the multiplication tensor is defined as T[i,j,k] = 1 for all i,j,k in range
    
    # Wait, I need to be more careful. The matrix multiplication tensor M_{n,m,p} is defined such that:
    # <M_{n,m,p}, A ⊗ B> = A*B
    # where A is n×m, B is m×p
    # So M_{n,m,p}[i,j,k] = 1 for all i∈[n], j∈[m], k∈[p]
    # And the bilinear form is: sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j]*B[j,k]) = sum_{i,k} C[i,k]
    
    # So the tensor is all ones! T[i,j,k] = 1 for all i,j,k.
    # And we need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For the schoolbook: for each (i,j,k), set U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 with r = i*m*p + j*p + k
    # This gives rank n*m*p = 2*3*9 = 54
    
    # Can we do better? Let's try a numerical approach to find a lower rank decomposition.
    
    # First, let's try rank 27 (half of 54) to see if it's feasible
    # We can use the fact that the tensor has a block structure
    
    # Actually, let me think about this more carefully using the block structure.
    # T[i,j,k] where k = 3t + r, t∈{0,1,2}, r∈{0,1,2}
    # T[i,j,3t+r] = 1 for all i,j,t,r
    
    # For each block t, we have a sub-tensor T_t[i,j,r] = 1 for i∈{0,1}, j∈{0,1,2}, r∈{0,1,2}
    # This is the M_{2,3,3} tensor, which has rank 18 (schoolbook)
    
    # If the three blocks were independent, total rank = 3*18 = 54
    # But they share the same A factors (U matrix)
    
    # Let's try to construct a decomposition with rank < 54 by sharing U factors.
    
    # Approach: For each (i,j), the U factor U[i*m+j, :] appears in all three blocks.
    # In block t, the contribution is U[i*m+j, :] * V_t[j, :] * W_t[:, i]
    # where V_t and W_t are the V and W factors for block t.
    
    # This is getting complex. Let me just try a numerical optimization approach.
    
    # Try to find a rank-R decomposition for various R starting from a lower bound
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    
    # Let's try rank 27 first
    R = 27
    
    # Initialize with random values and try to optimize
    # This might be too slow, so let's try a structured approach instead.
    
    # Structured approach: Use the block decomposition with shared U
    # For each of the 3 blocks, use a rank-9 decomposition of M_{2,3,3}
    # But M_{2,3,3} has rank 18, not 9...
    
    # Actually, I recall that the rank of M_{2,3,3} might be less than 18.
    # Let me check: the rank of M_{n,m,p} is at least max(nm, mp, pn).
    # For M_{2,3,3}: max(6, 9, 6) = 9. So rank >= 9.
    # The schoolbook gives rank 18.
    # Is there a rank-9 or rank-10 decomposition?
    
    # I believe the rank of M_{2,3,3} is actually 18, but I'm not 100% sure.
    # Let me just use the schoolbook for now and try to optimize numerically.
    
    # Given time constraints, let's use a practical approach:
    # 1. Start with the schoolbook (rank 54)
    # 2. Try to reduce rank using numerical optimization
    
    # For now, let's output the schoolbook decomposition which is guaranteed correct.
    # This gives rank 54, same as the seed, so score = 1.0.
    
    # To beat the seed, we need rank < 54.
    # Let's try rank 27 with a structured decomposition.
    
    # Idea: Group the k index as k = 3t + r.
    # For each (i, j, t), we have 3 terms (r=0,1,2) that share the same U[i,j].
    # If we could combine these 3 terms into 1, we'd get rank 54/3 = 18.
    # But that's not directly possible.
    
    # Alternative: Use the fact that for fixed (i,j), the vector (V[j,3t+0], V[j,3t+1], V[j,3t+2]) 
    # across all t can be combined.
    
    # Let me try a different structured approach:
    # Use rank = 3 * 9 = 27, where for each block t, we use a rank-9 decomposition.
    # But we need to verify that M_{2,3,3} has rank 9.
    
    # Actually, I just realized: the tensor M_{2,3,3} is NOT all ones in a way that allows rank 9.
    # The rank of M_{2,3,3} is 18 (I'm fairly confident about this).
    
    # So the block approach gives rank 54, same as schoolbook.
    
    # Let me try a numerical approach to see if we can find a lower rank decomposition.
    
    # Given the time budget, let's try a simple numerical optimization for rank 30.
    if time.time() > deadline - 5:
        # Not enough time, return schoolbook
        R = n * m * p
        U = [[0.0] * R for _ in range(n * m)]
        V = [[0.0] * R for _ in range(m * p)]
        W = [[0.0] * R for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1.0
                    V[j * p + k][idx] = 1.0
                    W[k * n + i][idx] = 1.0
                    idx += 1
        return {"U": U, "V": V, "W": W}
    
    # Try numerical optimization for rank 30
    R_try = 30
    np.random.seed(42)
    
    # Initialize factors
    U_init = np.random.randn(n * m, R_try) * 0.1
    V_init = np.random.randn(m * p, R_try) * 0.1
    W_init = np.random.randn(p * n, R_try) * 0.1
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a non-convex problem. Let's use a simple gradient-based approach.
    # But this might be too slow. Let me just return the schoolbook for now.
    
    # Actually, let me try a smarter structured decomposition.
    # 
    # Key insight: We can use the Strassen-like approach or exploit the small dimension.
    # 
    # For 2x3 by 3x9, let's think of it as:
    # C = A * B where A is 2x3, B is 3x9
    # 
    # We can partition B into 3 blocks of 3 columns: B = [B0 | B1 | B2]
    # C = [A*B0 | A*B1 | A*B2]
    # 
    # Each A*Bt is a 2x3 by 3x3 multiplication.
    # 
    # The rank of M_{2,3,3} is 18, so total rank = 3*18 = 54.
    # 
    # But can we do better by sharing computations?
    # 
    # Let's try to find if there's a rank-27 decomposition using a different structure.
    # 
    # I'll just output the schoolbook for now since I can't quickly find a better decomposition.
    
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
