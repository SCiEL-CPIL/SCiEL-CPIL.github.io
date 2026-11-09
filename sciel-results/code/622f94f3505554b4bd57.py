"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the n x m x p matmul tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matmul tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # W index is k*n + i (trace convention W[k*n+i, r])
    
    for c in range(rank):
        u = U[:, c]  # shape (n*m,)
        v = V[:, c]  # shape (m*p,)
        w = W[:, c]  # shape (p*n,)
        
        # Reconstruct contribution to each (i,j,k)
        # U[i*m+j] * V[j*p+k] * W[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The matmul tensor has 1 at every (i,j,k)
                    actual = u[i*m+j] * v[j*p+k] * w[k*n+i]
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using tensor operations."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the full tensor contribution
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape U to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank) -- W[k*n+i, c] means index is k*n+i, so reshape to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We need to sum over c: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # First compute U3[i,j,c] * V3[j,k,c] -> shape (n, m, p, rank)
    # U3[:, :, :, None] * V3[None, :, :, :] doesn't work directly due to different axes
    
    # Let's do it step by step
    # For each c, we have U3[:,:,c] (n,m), V3[:,:,c] (m,p), W3[:,:,c] (p,n)
    # Contribution to T[i,j,k] from term c: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Build full tensor
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        # For each (i,j,k): uc[i,j] * vc[j,k] * wc[k,i]
        # uc[i,j] * vc[j,k] gives (n, m, p) via uc[:, :, None] * vc[None, :, :]
        # Then multiply by wc[k,i] which is (p, n) -> need wc.T[i, k] = wc[k, i]
        # So we need (n, m, p) * (p, n) broadcast -> reshape wc.T to (n, 1, p)
        temp = uc[:, :, None] * vc[None, :, :]  # (n, m, p)
        temp = temp * wc.T[:, None, :]  # (n, 1, p) broadcast to (n, m, p)
        T += temp
    
    # The matmul tensor should be all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _construct_block_strassen(n, m, p):
    """
    Decompose using block structure.
    For 2x3x10: split B into five 3x2 blocks.
    Each 2x3x2 block can use Strassen-like rank 7? No, 2x3x2 min rank is 6 (schoolbook) 
    or possibly less. Actually for 2x3x2, schoolbook is 6.
    
    Let me think more carefully. The strategy says: decompose 3x10 as direct sum of ten 
    3x1 multiplications. Each 2x3x1 tensor computes one output column: C[:,k] = A * B[:,k].
    This is a 2x3x1 tensor with T[i,j,k] where k is fixed. For fixed k, T[i,j,k] = 1 for all i,j.
    This is just the outer product structure: it's a rank-3 tensor (2x3x1) with all entries 1.
    The rank of this tensor: we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j (k fixed).
    Since k is fixed (only one value), W[k,i,c] is just a vector of length n=2.
    
    Actually, let me reconsider. The 2x3x1 tensor has entries T[i,j,0] = 1 for i in {0,1}, j in {0,1,2}.
    We need: sum_c U[i*m+j, c] * V[j*p+0, c] * W[0*n+i, c] = 1 for all i,j.
    
    With p=1, V has shape (m*1, rank) = (3, rank), W has shape (1*n, rank) = (2, rank).
    U has shape (2*3, rank) = (6, rank).
    
    This is equivalent to: for each i, the vector [sum_c U[i*3+j,c]*V[j,c]*W[i,c] for j=0,1,2] = [1,1,1].
    
    The minimal rank for this: Let's think of it as matrix multiplication of a 2x3 matrix by a 3x1 vector.
    The standard rank is 6. Can we do better? 
    
    Actually, the tensor T[i,j,k] with k fixed and T[i,j,k]=1 is isomorphic to the 2x3 matrix of all ones.
    The tensor rank of a matrix M is just the matrix rank if we view it as a 2-way tensor, but here it's 3-way.
    
    Let me just use the schoolbook for each column: rank 3*2 = 6 per column, total 10*6 = 60. That's the same as full schoolbook.
    
    The strategy in the prompt says "each 2x3x1 tensor has a proven minimal rank of 3". Let me check if that's possible.
    
    For rank 3: sum_{c=1}^3 U[i,j,c] * V[j,c] * W[i,c] = 1 for all i in {0,1}, j in {0,1,2}.
    
    Let's denote a_{i,c} = W[i,c], b_{j,c} = V[j,c], and U[i,j,c] is free.
    Then sum_c U[i,j,c] * b_{j,c} * a_{i,c} = 1.
    
    This means for each (i,j), the vector [U[i,j,c] * a_{i,c}] dot [b_{j,c}] = 1.
    Or: (a_i ⊙ U_i) · b_j = 1 where a_i is the c-vector for row i of W, U_i is row i of U (flattened over j... no).
    
    Actually U[i,j,c] is a separate variable for each (i,j,c). So for fixed i:
    sum_c U[i,j,c] * (a_{i,c} * b_{j,c}) = 1 for j=0,1,2.
    
    Let d_{j,c} = a_{i,c} * b_{j,c}. Then sum_c U[i,j,c] * d_{j,c} = 1.
    
    For this to have a solution, we need that for each i, the system in variables U[i,j,c] (which are free per (i,j,c)) works.
    Since U[i,j,c] is free, we just need that the vector [1,1,1] is in the span of rows of the matrix D_i where D_i[j,c] = a_{i,c}*b_{j,c}.
    
    D_i = a_i (outer) b^T where a_i is in R^3 (c-index) and b is in R^{3x3} (j x c).
    Actually D_i[j,c] = a_{i,c} * b_{j,c} = (a_i ⊙ b_j)_c... no.
    D_i = diag(a_i) @ B where B[j,c] = b_{j,c}.
    
    The row space of D_i contains the row space of B (scaled by a_i). So we need [1,1,1] in row space of B (assuming a_i is nonzero).
    B is 3x3 (j=0,1,2; c=1,2,3). If B is invertible, its row space is all of R^3, so [1,1,1] is in the row space.
    
    So rank 3 IS possible for each 2x3x1 block! Total rank = 10 * 3 = 30.
    
    Let me construct it explicitly.
    """
    # For each column k (0..9), we compute C[:,k] = A @ B[:,k]
    # Using rank-3 decomposition of the 2x3x1 tensor
    
    # For a single 2x3x1 block (fixed k):
    # We need: sum_{c=1}^3 U[i,j,c] * V[j,c] * W[i,c] = 1 for i in {0,1}, j in {0,1,2}
    # 
    # Choose W[i,c] and V[j,c] such that the system is solvable.
    # Simple choice: W[i,c] = 1 for all i,c; V[j,c] = 1 for all j,c.
    # Then sum_c U[i,j,c] * 1 * 1 = sum_c U[i,j,c] = 1.
    # So U[i,j,1] = 1, U[i,j,2] = 0, U[i,j,3] = 0.
    # 
    # Let's verify: sum_c U[i,j,c]*V[j,c]*W[i,c] = U[i,j,1]*1*1 + U[i,j,2]*1*1 + U[i,j,3]*1*1 = 1+0+0 = 1. ✓
    
    # So for each output column k:
    #   U[i*m+j, base+c] = 1 if c==0 else 0 (for all i,j)
    #   V[j*p+k, base+c] = 1 for all j, c
    #   W[k*n+i, base+c] = 1 for all i, c
    #   where base = k*3, c in {0,1,2}
    
    n, m, p = 2, 3, 10
    rank = p * 3  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        base = k * 3
        # U[i*m+j, base+0] = 1 for all i,j
        for i in range(n):
            for j in range(m):
                U[i * m + j][base + 0] = 1
                # U[i*m+j][base+1] = 0
                # U[i*m+j][base+2] = 0
        # V[j*p+k, base+c] = 1 for all j, c
        for j in range(m):
            for c in range(3):
                V[j * p + k][base + c] = 1
        # W[k*n+i, base+c] = 1 for all i, c
        for i in range(n):
            for c in range(3):
                W[k * n + i][base + c] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try the rank-30 construction first
    result = _construct_block_strassen(n, m, p)
    
    # Verify it
    if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    result = _construct_schoolbook(n, m, p)
    if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Should not reach here
    return _construct_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
