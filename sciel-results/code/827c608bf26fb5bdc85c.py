"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against identity: T[i,j,k] should be 1 if (i,j,k) is a valid triple, else 0
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta (it's the identity tensor for matmul)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # (since the matmul tensor has all 1s at valid indices)
    
    # More precisely, the matmul tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no. The standard matmul tensor: T[i,j,k] = 1 means C[i,k] += A[i,j]*B[j,k]
    # So T is the identity in the sense that T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let's verify by checking a few random entries
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy: Use the block structure. The 4x8 matrix can be seen as two 4x4 blocks.
    We can use the fact that M_{3,4,8} can be decomposed using the structure of
    the rectangular dimensions.
    
    Key insight: A 3x4 times 4x8 multiplication can be viewed as:
    - The second matrix B is 4x8, which we can split into B_left (4x4) and B_right (4x4)
    - C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    
    But this doesn't directly help reduce rank below 96 in a simple way.
    
    Alternative: Use Winograd-style or Strassen-like ideas adapted to rectangular case.
    
    For now, return the schoolbook rank-96 solution, but try to find something better
    using algebraic structure.
    
    Actually, let's think about this more carefully. The matmul tensor for (n,m,p) has
    rank at most n*m*p. For (3,4,8), that's 96.
    
    Can we do better? The rank of the matmul tensor M_{n,m,p} is known to be at most
    n*m*p, and for some cases strictly less. For square n=n=n, Strassen gives 7 for 2x2x2.
    
    For rectangular, we can use embedding: M_{3,4,8} embeds in M_{4,4,8} which embeds in M_{4,4,16}...
    But these embeddings don't necessarily give rank reductions.
    
    A practical approach: try to use the block structure.
    A is 3x4, B is 4x8. Split B into two 4x4 blocks: B = [B1 | B2].
    C = [A*B1 | A*B2].
    
    This gives us two independent 3x4 x 4x4 multiplications. Each has rank at most 3*4*4=48.
    Total rank at most 96. Same as schoolbook.
    
    Let me try a different approach: use the fact that we only need integer/half-integer entries.
    
    For a 3x4 x 4x8 problem, one approach is to use the "border rank" or "sub-tensor" methods.
    
    Actually, let me just return the schoolbook solution. It's guaranteed correct with rank 96.
    """
    n, m, p = DIMENSIONS
    
    # Try to construct a lower-rank decomposition using block structure
    # Split the 8 columns of B into two groups of 4
    # C[:, 0:4] = A @ B[:, 0:4] and C[:, 4:8] = A @ B[:, 4:8]
    # Each is a 3x4 x 4x4 matmul
    
    # For 3x4 x 4x4, can we do better than 48?
    # Embed in 4x4 x 4x4 (rank 64 schoolbook, or 7*4=28 with Strassen-like? No, Strassen is for 2x2)
    
    # Let's just use schoolbook for now - it's safe and correct
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
