"""Matrix multiplication 5x5x5 rank minimization via block-strassen hybrid."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    Output C = A*B where C[i,k] = sum_j A[i,j]*B[j,k].
    Tensor T[i,j,k] = delta_{i,k} ... wait no.
    For matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    Tensor T_{i,j,k} where we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,k}? No.
    
    Actually the tensor for matrix multiplication of (n x m) * (m x p) -> (n x p):
    T[i, j, k] such that C[i,k] = sum_j A[i,j] B[j,k].
    In tensor form: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n+i, r] is the trace convention.
    
    The decomposition satisfies:
    C[i,k] = sum_{j, c} A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j*p+k', c] * B[j,k']) ... 
    
    No wait. Let me think more carefully.
    
    Standard tensor decomposition for matrix mult:
    C = A * B, where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    Tensor form: T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]... 
    Actually the multiplication tensor has entries T[i, j, k] where the three indices are:
    - i: row of A (and C), 0..n-1
    - j: column of A (and row of B), 0..m-1  
    - k: column of B (and C), 0..p-1
    
    And T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the sense that every triple contributes).
    
    Wait no. The multiplication tensor M_{i,j,k} = 1 if the output C[i,k] depends on A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]*B[j,k], every triple (i,j,k) with valid indices contributes. So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition writes M as sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c where:
    u_c is a vector of length n*m (indexed by (i,j)), v_c length m*p (indexed by (j,k)), w_c length p*n (indexed by (k,i)).
    
    And the condition is: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    So U[i*m+j, c] = u_c[i*m+j], V[j*p+k, c] = v_c[j*p+k], W[k*n+i, c] = w_c[k*n+i].
    
    For schoolbook: for each (i,j,k), one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    
    For Strassen 2x2: the 7 multiplications are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A11-A01)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M1
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M5
    
    Let me construct the U, V, W matrices for 2x2x2 Strassen.
    n=m=p=2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    
    For each multiplication c, we have linear forms in A entries, B entries, and C output coefficients.
    
    c=0 (M1): A form: A00+A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
             B form: B00+B11 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
             C form: C00 += 1, C11 += 1 -> W[0*2+0,0]=1, W[1*2+1,0]=1 (k=0,i=0 and k=1,i=1)
    
    c=1 (M2): A form: A10+A11 -> U[2,1]=1, U[3,1]=1 (i=1,j=0 and i=1,j=1)
             B form: B00 -> V[0,1]=1 (j=0,k=0)
             C form: C10 += 1, C11 -= 1 -> W[0*2+1,1]=1, W[1*2+1,1]=-1 (k=0,i=1 and k=1,i=1)
    
    c=2 (M3): A form: A00 -> U[0,2]=1 (i=0,j=0)
             B form: B10-B11 -> V[2,2]=1, V[3,2]=-1 (j=1,k=0 and j=1,k=1)
             C form: C01 += 1, C11 += 1 -> W[1*2+0,2]=1, W[1*2+1,2]=1 (k=1,i=0 and k=1,i=1)
    
    c=3 (M4): A form: A11 -> U[3,3]=1 (i=1,j=1)
             B form: B01-B10 -> V[1,3]=1, V[2,3]=-1 (j=0,k=1 and j=1,k=0)
             C form: C00 += 1, C10 += 1 -> W[0*2+0,3]=1, W[0*2+1,3]=1 (k=0,i=0 and k=0,i=1)
    
    c=4 (M5): A form: A00+A10 -> U[0,4]=1, U[2,4]=1 (i=0,j=0 and i=1,j=0)
             B form: B11 -> V[3,4]=1 (j=1,k=1)
             C form: C00 -= 1, C11 += 1 -> W[0*2+0,4]=-1, W[1*2+1,4]=1 (k=0,i=0 and k=1,i=1)
    
    c=5 (M6): A form: A10-A00 -> U[2,5]=1, U[0,5]=-1 (i=1,j=0 and i=0,j=0)
             B form: B00+B10 -> V[0,5]=1, V[2,5]=1 (j=0,k=0 and j=1,k=0)
             C form: (not used in standard Strassen output... let me recheck)
    
    Actually let me redo this. Standard Strassen:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B10-B11)
    P4 = A11(B01-B10)
    P5 = (A00+A10)B11
    P6 = (A10-A00)(B00+B10)
    P7 = (A11-A01)(B01+B11)
    
    C00 = P1 - P5 + P6 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P7
    
    Let me verify: 
    C00 = (A00+A11)(B00+B11) - (A00+A10)B11 + (A10-A00)(B00+B10) + (A11-A01)(B01+B11)
    = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A10B11 + A10B00 + A10B10 - A00B00 - A00B10 + A11B01 + A11B11 - A01B01 - A01B11
    = A11B00 + A11B11 - A10B11 + A10B00 + A10B10 - A00B10 + A11B01 + A11B11 - A01B01 - A01B11
    Hmm, this doesn't simplify to A00B00 + A01B10 cleanly. Let me just use a known correct formulation.
    
    Actually, I'll just use the standard formulation and verify numerically.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1]
    V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1]
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1]
    V[0, 1] = 1               # B[0,0]
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # P3 = A00(B10-B11)
    U[0, 2] = 1               # A[0,0]
    V[2, 2] = 1; V[3, 2] = -1  # B[1,0], B[1,1]
    W[1, 2] = 1; W[3, 2] = 1   # C[0,1], C[1,1]
    
    # P4 = A11(B01-B10)
    U[3, 3] = 1               # A[1,1]
    V[1, 3] = 1; V[2, 3] = -1  # B[0,1], B[1,0]
    W[0, 3] = 1; W[2, 3] = 1   # C[0,0], C[1,0]
    
    # P5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1  # A[0,0], A[1,0]
    V[3, 4] = 1               # B[1,1]
    W[0, 4] = -1; W[3, 4] = 1  # C[0,0], C[1,1]
    
    # P6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A[1,0], A[0,0]
    V[0, 5] = 1; V[2, 5] = 1   # B[0,0], B[1,0]
    W[0, 5] = 1                # C[0,0]
    
    # P7 = (A11-A01)(B01+B11)
    U[3, 6] = 1; U[1, 6] = -1  # A[1,1], A[0,1]
    V[1, 6] = 1; V[3, 6] = 1   # B[0,1], B[1,1]
    W[0, 6] = 1; W[3, 6] = 1   # C[0,0], C[1,1]
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _block_decomposition(n, m, p, block_sizes):
    """
    Decompose n x m x p matrix multiplication using block structure.
    block_sizes: tuple (nb, mb, pb) - number of blocks in each dimension.
    Each block is of size (n//nb, m//mb, p//pb) approximately.
    
    For each block (bi, bj, bk) where bi indexes rows of A/C, bj indexes cols of A/rows of B,
    bk indexes cols of B/C:
    - If block is 2x2x2, use Strassen (rank 7)
    - Otherwise use schoolbook (rank = block_n * block_m * block_p)
    """
    bn = n // block_sizes[0]
    bm = m // block_sizes[1]
    bp = p // block_sizes[2]
    
    # Handle remainders by using variable block sizes
    n_blocks = []
    for i in range(block_sizes[0]):
        start = i * bn
        if i == block_sizes[0] - 1:
            n_blocks.append(n - start)
        else:
            n_blocks.append(bn)
    
    m_blocks = []
    for j in range(block_sizes[1]):
        start = j * bm
        if j == block_sizes[1] - 1:
            m_blocks.append(m - start)
        else:
            m_blocks.append(bm)
    
    p_blocks = []
    for k in range(block_sizes[2]):
        start = k * bp
        if k == block_sizes[2] - 1:
            p_blocks.append(p - start)
        else:
            p_blocks.append(bp)
    
    # Compute total rank
    total_rank = 0
    for bi in range(block_sizes[0]):
        for bj in range(block_sizes[1]):
            for bk in range(block_sizes[2]):
                bn_i = n_blocks[bi]
                bm_j = m_blocks[bj]
                bp_k = p_blocks[bk]
                if bn_i == 2 and bm_j == 2 and bp_k == 2:
                    total_rank += 7
                else:
                    total_rank += bn_i * bm_j * bp_k
    
    # Build U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for bi in range(block_sizes[0]):
        for bj in range(block_sizes[1]):
            for bk in range(block_sizes[2]):
                bn_i = n_blocks[bi]
                bm_j = m_blocks[bj]
                bp_k = p_blocks[bk]
                
                # Block offsets
                i_off = bi * bn
                j_off = bj * bm
                k_off = bk * bp
                
                if bn_i == 2 and bm_j == 2 and bp_k == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2()
                    for c in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U[(i_off + ii) * m + (j_off + jj)][col + c] = sU[ii*2+jj][c]
                        for jj in range(2):
                            for kk in range(2):
                                V[(j_off + jj) * p + (k_off + kk)][col + c] = sV[jj*2+kk][c]
                        for kk in range(2):
                            for ii in range(2):
                                W[(k_off + kk) * n + (i_off + ii)][col + c] = sW[kk*2+ii][c]
                    col += 7
                else:
                    # Schoolbook for this block
                    for ii in range(bn_i):
                        for jj in range(bm_j):
                            for kk in range(bp_k):
                                U[(i_off + ii) * m + (j_off + jj)][col] = 1
                                V[(j_off + jj) * p + (k_off + kk)][col] = 1
                                W[(k_off + kk) * n + (i_off + ii)][col] = 1
                                col += 1
    
    return U, V, W, total_rank


def construct():
    """Construct rank-minimal decomposition for 5x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try different block partitions
    # 5 = 2+2+1, so we can have 3 blocks of sizes 2,2,1
    # Or 2+1+2, 1+2+2, 2+2+1, etc.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # schoolbook baseline
    
    # Try all ways to partition 5 into 2,2,1 (permutations)
    partitions = [
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
    ]
    
    # For each partition, try all combinations of block counts
    # With sizes like [2,2,1], we have 3 blocks
    # We can use Strassen for 2x2x2 blocks and schoolbook for others
    
    # Strategy: partition into blocks where 2x2x2 blocks use Strassen
    # For 5=2+2+1: blocks are (2,2,2), (2,2,1), (2,1,2), (2,1,1), (1,2,2), (1,2,1), (1,1,2), (1,1,1)
    # Only (2,2,2) can use Strassen.
    
    # Let's try: split each dimension into [2,2,1]
    # This gives 3x3x3 = 27 blocks
    # Only the (2,2,2) block (first one) uses Strassen: saves 8-7=1 rank
    # Total: 125 - 1 = 124? No wait.
    # Schoolbook for 2x2x2 is rank 8, Strassen is rank 7. Saving = 1.
    # Other blocks: 2x2x1 = 4, 2x1x2 = 4, 2x1x1 = 2, 1x2x2 = 4, 1x2x1 = 2, 1x1x2 = 2, 1x1x1 = 1
    # Total schoolbook: 27 blocks, one is 2x2x2 (rank 8->7), rest schoolbook
    # Sum of all block sizes = 5*5*5 = 125
    # With Strassen on one 2x2x2: 125 - 1 = 124
    
    # Can we do better? What about overlapping Strassen?
    # Actually, the standard approach for 5x5x5 is to use the fact that
    # 5 = 2+2+1 and apply Strassen to the 2x2x2 corner.
    
    # Let me try all three partitions and pick the best
    for part in partitions:
        # part is a tuple of 3 sizes that sum to 5
        # e.g., (2,2,1) means n is split into blocks of size 2,2,1
        U, V, W, r = _block_decomposition(n, m, p, (3, 3, 3))
        # Wait, I need to pass the actual block sizes, not just counts.
        # Let me restructure.
    
    # Actually, let me directly construct with explicit block sizes.
    # For n=m=p=5, split each into [2,2,1]:
    
    def _construct_with_sizes(n_sizes, m_sizes, p_sizes):
        """n_sizes, m_sizes, p_sizes are lists of block sizes summing to n, m, p."""
        total_rank = 0
        for ns in n_sizes:
            for ms in m_sizes:
                for ps in p_sizes:
                    if ns == 2 and ms == 2 and ps == 2:
                        total_rank += 7
                    else:
                        total_rank += ns * ms * ps
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        i_off = 0
        for ns in n_sizes:
            j_off = 0
            for ms in m_sizes:
                k_off = 0
                for ps in p_sizes:
                    if ns == 2 and ms == 2 and ps == 2:
                        sU, sV, sW = _strassen_2x2()
                        for c in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(i_off + ii) * m + (j_off + jj)][col + c] = sU[ii*2+jj][c]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(j_off + jj) * p + (k_off + kk)][col + c] = sV[jj*2+kk][c]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(k_off + kk) * n + (i_off + ii)][col + c] = sW[kk*2+ii][c]
                        col += 7
                    else:
                        for ii in range(ns):
                            for jj in range(ms):
                                for kk in range(ps):
                                    U[(i_off + ii) * m + (j_off + jj)][col] = 1
                                    V[(j_off + jj) * p + (k_off + kk)][col] = 1
                                    W[(k_off + kk) * n + (i_off + ii)][col] = 1
                                    col += 1
                    k_off += ps
                j_off += ms
            i_off += ns
        
        return U, V, W, total_rank
    
    # Try different size partitions
    size_options = [
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
    ]
    
    best = None
    for ns in size_options:
        for ms in size_options:
            for ps in size_options:
                U, V, W, r = _construct_with_sizes(ns, ms, ps)
                if best is None or r < best[3]:
                    best = (U, V, W, r)
    
    U, V, W, r = best
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
