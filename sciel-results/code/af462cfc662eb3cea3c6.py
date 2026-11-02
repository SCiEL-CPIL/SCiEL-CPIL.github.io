"""Matrix multiplication 4,4,8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] with the given index convention."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose T."""
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient check: for each r, add outer product
    recon = np.zeros(T.shape, dtype=float)
    for r in range(U.shape[1]):
        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(recon, T, atol=tol)

def _strassen_2x2():
    """
    Return the 7-rank decomposition for 2x2 * 2x2 -> 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Convention: A indexed by (i,j) -> i*2+j, B by (j,k) -> j*2+k, C by (k,i) -> k*2+i
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # For each of 7 factors, U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11=(0,0), A22=(1,1); B11=(0,0), B22=(1,1)
    # C11 += M1, C22 += M1
    # U: A11 and A22 -> indices 0, 3
    # V: B11 and B22 -> indices 0, 3
    # W: C11 and C22 -> indices 0, 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21=(1,0), A22=(1,1); B11=(0,0)
    # C21 += M2, C22 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    # A11=(0,0); B21=(1,0), B22=(1,1)
    # C12 += M3, C22 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    # A22=(1,1); B12=(0,1), B11=(0,0)
    # C11 += M4, C21 += M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=(0,0), A12=(0,1); B22=(1,1)
    # C11 -= M5, C12 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=(1,0), A11=(0,0); B11=(0,0), B12=(0,1)
    # C22 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=(0,1), A22=(1,1); B21=(1,0), B22=(1,1)
    # C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """
    Build a 49-rank decomposition for 4x4 * 4x4 -> 4x4 using Strassen recursively.
    4x4 can be split into four 2x2 blocks.
    This gives rank 7*7 = 49.
    """
    n, m, p = 4, 4, 4
    U, V, W = np.zeros((16, 49)), np.zeros((16, 49)), np.zeros((16, 49))
    
    # Split indices into 2x2 blocks
    # A: (i,j) where i,j in {0,1,2,3}
    # Block (bi, bj) where bi,bj in {0,1}
    # Within block, local indices li, lj in {0,1}
    # Global i = bi*2 + li, j = bj*2 + lj
    # A index = i*4 + j = (bi*2+li)*4 + (bj*2+lj) = bi*8 + li*4 + bj*2 + lj
    # B: (j,k) -> j*4+k = (bj*2+lj)*4 + (bk*2+lk) = bj*8 + lj*4 + bk*2 + lk
    # C: (k,i) -> k*4+i = (bk*2+lk)*4 + (bi*2+li) = bk*8 + lk*4 + bi*2 + li
    
    # For each pair of 2x2 block positions (block_A, block_B, block_C),
    # we apply Strassen 2x2.
    # block_A = (bi, bj), block_B = (bj, bk), block_C = (bk, bi)
    # For each such triple, we use 7 Strassen factors.
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # This triple of blocks contributes 7 rank-1 terms
                base = (bi * 2 + bj) * 2 + bk  # 0..7
                U2, V2, W2 = _strassen_2x2()
                
                for r in range(7):
                    col = base * 7 + r
                    # Map 2x2 local indices to 4x4 global indices
                    # A local (li, lj) -> global i=bi*2+li, j=bj*2+lj -> idx = i*4+j
                    # B local (lj, lk) -> global j=bj*2+lj, k=bk*2+lk -> idx = j*4+k
                    # C local (lk, li) -> global k=bk*2+lk, i=bi*2+li -> idx = k*4+i
                    
                    for li in range(2):
                        for lj in range(2):
                            a_idx = (bi*2+li)*4 + (bj*2+lj)
                            U[a_idx, col] += U2[li*2+lj, r]
                    
                    for lj in range(2):
                        for lk in range(2):
                            b_idx = (bj*2+lj)*4 + (bk*2+lk)
                            V[b_idx, col] += V2[lj*2+lk, r]
                    
                    for lk in range(2):
                        for li in range(2):
                            c_idx = (bk*2+lk)*4 + (bi*2+li)
                            W[c_idx, col] += W2[lk*2+li, r]
    
    return U, V, W

def _strassen_4x4_4x8():
    """
    Build a 98-rank decomposition for 4x4 * 4x8 -> 4x8.
    Split the 4x8 into two 4x4 blocks.
    Use two independent 4x4*4x4 Strassen decompositions (49 each).
    """
    n, m, p = 4, 4, 8
    U, V, W = np.zeros((16, 98)), np.zeros((32, 98)), np.zeros((32, 98))
    
    # Split k into two halves: k in {0,1,2,3} and k in {4,5,6,7}
    # For each half, we have a 4x4 * 4x4 -> 4x4 problem
    # A is the same 4x4 for both halves
    # B_half1: columns 0-3, B_half2: columns 4-7
    # C_half1: rows 0-3 of C (for k=0..3), C_half2: rows 4-7
    
    # C index convention: k*n + i where k in 0..7, i in 0..3
    # For k in 0..3: C index = k*4 + i
    # For k in 4..7: C index = k*4 + i = (4+lk)*4 + i = 16 + lk*4 + i
    
    U1, V1, W1 = _strassen_4x4_4x4()
    U2, V2, W2 = _strassen_4x4_4x4()
    
    # First half (k=0..3)
    U[:, 0:49] = U1
    # V: B indices j*8+k. For k=0..3, j in 0..3: idx = j*8 + k
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V1[j*4+k]
    # W: C indices k*4+i for k=0..3
    W[0:16, 0:49] = W1
    
    # Second half (k=4..7), let lk = k-4
    U[:, 49:98] = U2
    # V: B indices j*8+k where k=4+lk
    for j in range(4):
        for lk in range(4):
            V[j*8+(4+lk), 49:98] = V2[j*4+lk]
    # W: C indices k*4+i where k=4+lk
    for lk in range(4):
        for i in range(4):
            W[(4+lk)*4+i, 49:98] = W2[lk*4+i]
    
    return U, V, W

def _try_reduce_rank(U, V, W, T, current_rank, deadline, tol=1e-9):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    Uses SVD-based approach: check if any column can be expressed as a linear
    combination of others.
    """
    n, m, p = DIMENSIONS
    dim1, dim2, dim3 = n*m, m*p, p*n
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try to find if we can eliminate columns
    # A column r is redundant if there exist coefficients alpha_s (s != r)
    # such that for all a,b,c: U[a,r]*V[b,r]*W[c,r] = sum_s alpha_s * U[a,s]*V[b,s]*W[c,s]
    
    # This is complex. Instead, try a simpler approach:
    # Check if the set of rank-1 tensors is linearly dependent.
    # Flatten each rank-1 tensor to a vector of length dim1*dim2*dim3
    # Then check rank of the matrix whose columns are these vectors.
    
    # This is expensive for large dimensions, so only try for smaller ranks
    if current_rank > 60:
        return best_U, best_V, best_W, best_rank
    
    # Flatten rank-1 tensors
    # Each rank-1 tensor is U[:,r] outer V[:,r] outer W[:,r]
    # Flatten to 1D vector
    flat_tensors = []
    for r in range(current_rank):
        t = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        flat_tensors.append(t.flatten())
    
    flat_matrix = np.array(flat_tensors).T  # shape (dim1*dim2*dim3, current_rank)
    
    # Find the rank of this matrix
    try:
        svd_vals = np.linalg.svd(flat_matrix, compute_uv=False)
        # Count significant singular values
        rank_est = np.sum(svd_vals > tol * svd_vals[0] if len(svd_vals) > 0 else 0)
        if rank_est < current_rank:
            # We can potentially reduce rank
            # But we need to find the actual reduced decomposition
            # For now, just note the rank
            pass
    except:
        pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build target tensor
    T = _build_tensor(n, m, p)
    
    # Start with Strassen-based decomposition: rank 98
    U, V, W = _strassen_4x4_4x8()
    current_rank = 98
    
    # Verify
    if not _verify(U, V, W, T, tol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        current_rank = rank
    else:
        # Try to reduce rank
        if time.time() < deadline:
            U, V, W, current_rank = _try_reduce_rank(U, V, W, T, current_rank, deadline)
    
    # Try a different approach: use the known 4x4x4 decomposition with rank 49
    # and see if we can do better for 4x4x8
    if time.time() < deadline:
        # Try: can we do 4x4*4x8 with rank < 98?
        # Known: 4x4x4 has rank <= 49 (Strassen)
        # 4x4x8 = two 4x4x4 blocks, so rank <= 98
        # But maybe there's a smarter decomposition
        
        # Try using the fact that we can share some computations
        # The A matrix is the same for both halves, so maybe we can
        # use a decomposition that exploits this
        
        # For now, return the 98-rank decomposition
        pass
    
    # Convert to list format
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
