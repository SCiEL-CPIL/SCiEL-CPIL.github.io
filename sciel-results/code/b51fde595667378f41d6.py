"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x4():
    """Rank-10 decomposition for 2x2 * 2x4 using Strassen for first 2x2 block + 3 products for second."""
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: A[i*m+j], B[j*p+k], W[k*n+i]
    # A = [[a00, a01], [a10, a11]] -> indices 0,1,2,3
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]] -> indices 0-7
    # C = [[c00, c01, c02, c03], [c10, c11, c12, c13]] -> W indices: c00->0, c01->1, c02->2, c03->3, c10->4, c11->5, c12->6, c13->7
    
    # Strassen 7 products for B columns 0-1:
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[5, 0] = 1  # c00, c11
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[4, 1] = 1; W[5, 1] = -1  # c10, c11
    
    # M3 = a00(b10-b00)
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1  # c01, c11
    
    # M4 = a11(b11-b01)
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[4, 3] = 1  # c00, c10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[5, 5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # c00
    
    # 3 products for B columns 2-3:
    # M8 = (a00+a11)(b02+b13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[2, 7] = 1; W[7, 7] = 1  # c02, c13
    
    # M9 = (a10+a11)b02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[6, 8] = 1; W[7, 8] = -1  # c12, c13
    
    # M10 = a00(b12-b02)
    U[0, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[2, 9] = 1; W[6, 9] = 1  # c02, c12
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is correct."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check that for valid (i,j,k) triples, the sum equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(U[a_idx], V[b_idx] * W[c_idx])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that for invalid triples, the sum equals 0
    for a_idx in range(n * m):
        i, j = divmod(a_idx, m)
        for b_idx in range(m * p):
            jj, k = divmod(b_idx, p)
            for c_idx in range(p * n):
                kk, ii = divmod(c_idx, n)
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a_idx], V[b_idx] * W[c_idx])
                    if abs(val) > tol:
                        return False
    return True

def _try_rank_24():
    """
    Attempt to construct a rank-24 decomposition.
    
    For 2x2 * 2xp matrix multiplication, the rank is known to be 2p for p >= 2.
    For p=12, rank should be 24.
    
    Construction idea:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can write this as:
    C[i,k] = sum_r U[i*2+j, r] * V[j*12+k, r] * W[k*2+i, r]
    
    For rank 24, we need 24 terms. One natural construction:
    For each pair (i, k), we need to compute A[i,0]*B[0,k] + A[i,1]*B[1,k].
    
    Actually, let's think of it differently. The tensor is:
    T(i,j,k) = 1 for all i in {0,1}, j in {0,1}, k in {0,...,11}
    
    We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k).
    
    One rank-24 construction:
    For r = 0 to 11:
      U[0, r] = 1, U[2, r] = 1  (A[0,0] and A[1,0])
      V[0, r] = 1, V[12, r] = 1  (B[0, r] and B[1, r])
      W[r*2, r] = 1, W[r*2+1, r] = 1  (C[0,r] and C[1,r])
    
    Wait, that's not right. Let me reconsider.
    
    Actually, for rank 2p = 24, a known construction is:
    For each k in 0..11, use 2 terms:
    Term 2k: U[0, 2k]=1, U[2, 2k]=1; V[0, 2k]=1, V[12+k, 2k]=1; W[k*2, 2k]=1, W[k*2+1, 2k]=1
    This doesn't work directly.
    
    Let me try a different approach. The standard result is that rank(2x2 x 2xp) = 2p for p >= 2.
    
    A construction for rank 2p:
    For k = 0 to p-1:
      Term 2k:   U[0,2k]=1, U[1,2k]=0; V[0,2k]=1, V[p+2k,2k]=0; W[2k*2,2k]=1, W[2k*2+1,2k]=0
                 This gives A[0,0]*B[0,k] -> C[0,k]
      Term 2k+1: U[0,2k+1]=0, U[1,2k+1]=1; V[0,2k+1]=0, V[p+2k+1,2k+1]=1; W[2k*2,2k+1]=1, W[2k*2+1,2k+1]=0
                 This gives A[0,1]*B[1,k] -> C[0,k]
    
    Wait, I'm getting confused with the indexing. Let me be more careful.
    
    U has shape (n*m, rank) = (4, rank). Row index = i*m + j = 2i + j.
    V has shape (m*p, rank) = (24, rank). Row index = j*p + k = 12j + k.
    W has shape (p*n, rank) = (24, rank). Row index = k*n + i = 2k + i.
    
    We need: sum_r U[2i+j, r] * V[12j+k, r] * W[2k+i, r] = 1 for all i,j,k.
    
    For rank 24, let's try:
    For each k in 0..11:
      Term r = 2k:
        U[0, 2k] = 1  (i=0, j=0)
        U[2, 2k] = 1  (i=1, j=0)
        V[12*0+k, 2k] = 1  (j=0, k)
        V[12*1+k, 2k] = 1  (j=1, k)
        W[2k+0, 2k] = 1  (k, i=0)
        W[2k+1, 2k] = 1  (k, i=1)
      
      This would give for (i,j,k):
        If j=0: U[2i, 2k] * V[k, 2k] * W[2k+i, 2k] = 1 * 1 * 1 = 1 for i=0,1
        If j=1: U[2i+1, 2k] * V[12+k, 2k] * W[2k+i, 2k] = 0 * 1 * 1 = 0
      
      So this only handles j=0. We need another term for j=1.
    
    Let me try:
    For each k in 0..11:
      Term r = 2k:
        U[0, 2k] = 1  (i=0, j=0)
        U[2, 2k] = 1  (i=1, j=0)
        V[k, 2k] = 1  (j=0, k)
        W[2k, 2k] = 1  (k, i=0)
        W[2k+1, 2k] = 1  (k, i=1)
      
      Term r = 2k+1:
        U[1, 2k+1] = 1  (i=0, j=1)
        U[3, 2k+1] = 1  (i=1, j=1)
        V[12+k, 2k+1] = 1  (j=1, k)
        W[2k, 2k+1] = 1  (k, i=0)
        W[2k+1, 2k+1] = 1  (k, i=1)
    
    Check for (i=0, j=0, k):
      Term 2k: U[0,2k]*V[k,2k]*W[2k,2k] = 1*1*1 = 1
      Term 2k+1: U[0,2k+1]*V[k,2k+1]*W[2k,2k+1] = 0*0*1 = 0
      Other terms: 0
      Sum = 1 ✓
    
    Check for (i=0, j=1, k):
      Term 2k: U[1,2k]*V[k,2k]*W[2k,2k] = 0*1*1 = 0
      Term 2k+1: U[1,2k+1]*V[12+k,2k+1]*W[2k,2k+1] = 1*1*1 = 1
      Sum = 1 ✓
    
    Check for (i=1, j=0, k):
      Term 2k: U[2,2k]*V[k,2k]*W[2k+1,2k] = 1*1*1 = 1
      Sum = 1 ✓
    
    Check for (i=1, j=1, k):
      Term 2k+1: U[3,2k+1]*V[12+k,2k+1]*W[2k+1,2k+1] = 1*1*1 = 1
      Sum = 1 ✓
    
    Now check invalid triples. For (i, j, k) where the indices don't match:
    The only way to get a non-zero contribution is if the U, V, W indices align.
    
    For term 2k': U is non-zero at rows 0 and 2 (j=0). V is non-zero at row k' (j=0). W is non-zero at rows 2k' and 2k'+1 (i=0 and i=1).
    So for term 2k' to contribute, we need j=0, and the k in V must be k', and the i in W can be 0 or 1.
    The contribution is to (i, 0, k') for i=0,1.
    
    For term 2k'+1: U is non-zero at rows 1 and 3 (j=1). V is non-zero at row 12+k' (j=1). W is non-zero at rows 2k' and 2k'+1.
    So for term 2k'+1 to contribute, we need j=1, and the k in V must be k', and the i in W can be 0 or 1.
    The contribution is to (i, 1, k') for i=0,1.
    
    So the only non-zero contributions are for valid (i,j,k) triples. This should work!
    
    Let's verify with code.
    """
    n, m, p = 2, 2, 12
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k in range(p):
        # Term 2k: handles j=0
        r = 2*k
        U[0, r] = 1   # i=0, j=0
        U[2, r] = 1   # i=1, j=0
        V[k, r] = 1   # j=0, k
        W[2*k, r] = 1     # k, i=0
        W[2*k+1, r] = 1   # k, i=1
        
        # Term 2k+1: handles j=1
        r = 2*k + 1
        U[1, r] = 1   # i=0, j=1
        U[3, r] = 1   # i=1, j=1
        V[12+k, r] = 1  # j=1, k
        W[2*k, r] = 1     # k, i=0
        W[2*k+1, r] = 1   # k, i=1
    
    return U, V, W

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-24 construction first
    if time.time() < deadline:
        U24, V24, W24 = _try_rank_24()
        if _verify_decomposition(U24, V24, W24, n, m, p):
            return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
    
    # Fallback: rank-30 tiling of rank-10 2x2x4 decomposition
    if time.time() < deadline:
        U4, V4, W4 = _strassen_2x2x4()
        if _verify_decomposition(U4, V4, W4, 2, 2, 4):
            rank = 30
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for t in range(3):
                base_col = 4*t
                for r in range(10):
                    U[:, t*10+r] = U4[:, r]
                    for j in range(2):
                        for k in range(4):
                            V[j*p + base_col + k, t*10+r] = V4[j*4+k, r]
                    for k in range(4):
                        for i in range(2):
                            W[(base_col+k)*n+i, t*10+r] = W4[k*2+i, r]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: rank-42 Strassen on 6 pairs
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(6):
        if time.time() > deadline:
            break
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj][col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk][col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
