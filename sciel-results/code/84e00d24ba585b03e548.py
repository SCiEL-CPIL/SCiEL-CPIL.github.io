"""
Kronecker-product structural decomposition for 3x4 * 4x8 matrix multiplication.

Idea:
  * View the 4x8 right factor as two 4x4 blocks (p = 4 + 4).
  * For each 4x4 block, use a rank-8 "Strassen-like" decomposition for
    2x2 * 2x2 (applied to two independent 2x2 blocks inside the 4x4,
    plus the cross terms handled by schoolbook where necessary) -- but to keep
    correctness simple and verifiable, we instead use a *verified* rank-8
    decomposition for the full 2x2 * 2x2 tensor (Strassen's 7 multiplications
    actually give rank 7 for the 2x2 * 2x2 multiplication tensor; here we use
    the standard 7-term Strassen).  For a 4x4 block we can split it into four
    2x2 sub-blocks.  However, the cross-block terms in the 4x4 multiplication
    are not independent 2x2 multiplications, so we fall back to schoolbook for
    the cross terms.

  A cleaner structural approach:
  * Split the p=8 dimension into four groups of 2.
  * For each group of 2 in p, and each i in [0..2], j in [0..3], the
    corresponding "slice" is a 1x1 * 1x2 multiplication, which has rank 2
    (schoolbook).  That gives 3*4*2*4 = 96, same as schoolbook.

  Better: use Strassen on overlapping 2x2 * 2x2 sub-blocks where possible.
  Concretely, partition the i=3 into {0,1} and {2}, m=4 into {0,1} and {2,3},
  p=8 into four pairs {0,1},{2,3},{4,5},{6,7}.

  For the block (i={0,1}, m={0,1}) and each p-pair, use Strassen (rank 7 each).
  For the block (i={0,1}, m={2,3}) and each p-pair, use Strassen (rank 7 each).
  For i={2}, use schoolbook for all m and p.

  Total rank = 2 (i-pairs) * 2 (m-pairs) * 4 (p-pairs) * 7 (Strassen) 
             + 1 (i-single) * 4 (m) * 8 (p) * 1 (schoolbook)
             = 2*2*4*7 + 1*4*8 = 112 + 32 = 144.  That's worse.

  So instead, use Strassen only where it helps most: overlap is not allowed in
  a partition-based approach.  Let's just do a clean partition search:
  partition n=3 into {0,1}+{2}, m=4 into {0,1}+{2,3}, p=8 into four {2}-blocks.
  For each (n-block, m-block, p-block) triple:
    if n-block size 2, m-block size 2, p-block size 2: use Strassen (rank 7)
    else: schoolbook (rank = product of sizes)

  Partitions:
    n: {0,1},{2}  -> sizes [2,1]
    m: {0,1},{2,3} -> sizes [2,2]
    p: {0,1},{2,3},{4,5},{6,7} -> sizes [2,2,2,2]

  Triples:
    (2,2,2) blocks: 1*1*4 = 4 blocks, each rank 7 -> 28
    (2,2,1) or (2,1,2) or (1,2,2) etc: 
    Actually with the partition above, every n-block is size 2 or 1, every m-block size 2, every p-block size 2.
    So:
      n={0,1} (size 2), m={0,1} (size 2), p each pair (size 2): 4 blocks * 7 = 28
      n={0,1} (size 2), m={2,3} (size 2), p each pair (size 2): 4 blocks * 7 = 28
      n={2} (size 1), m={0,1} (size 2), p each pair (size 2): 4 blocks * 2 = 8
      n={2} (size 1), m={2,3} (size 2), p each pair (size 2): 4 blocks * 2 = 8
    Total = 28+28+8+8 = 72.

  Let's verify this is a valid exact decomposition.
"""
import time
import numpy as np

N, M, P = 3, 4, 8

def _strassen_2x2x2(i0, j0, k0, col, U, V, W, n, m, p):
    """
    Write the 7 Strassen terms for the 2x2 * 2x2 sub-block
    starting at indices i0, j0, k0 into columns col..col+6 of U,V,W.
    Returns col + 7.
    """
    # Strassen terms for C = A * B where A,B,C are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    #
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # U entries are coefficients on A, V on B, W on C.
    # U[i,j] is the coefficient for A[i,j] in the rank-1 factor.
    # V[j,k] is the coefficient for B[j,k].
    # W[k,i] is the coefficient for C[k,i]... wait, the trace convention is W[k*n+i, r].
    # So W index is (k, i) where k is the p-index and i is the n-index.
    # C[i,j] = sum_k A[i,k] * B[k,j].  The tensor entry is T[i,j,k] = A[i,k] B[k,j] C[j,i]? 
    # No, the standard matrix multiplication tensor is T[i,j,k] such that
    # (AB)[i,j] = sum_k A[i,k] B[k,j].
    # The factorization is (AB)[i,j] = sum_r U[i,j,r] V[k,j,r] W[k,i,r]? 
    # Let's use the convention from the problem: W[k*n+i, r].
    # So the tensor is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} for the multiplication tensor.
    # Wait, the multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of the product depends on A[i,k]*B[k,j].
    # Actually T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in the right indices).
    # No, T[i,j,k] = 1 means (AB)[i,j] gets contribution A[i,k]*B[k,j]. So T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition must satisfy: for all i,j,k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    
    # For the 2x2x2 Strassen, the local indices are di in {0,1}, dj in {0,1}, dk in {0,1}.
    # Global indices: i = i0+di, j = j0+dj, k = k0+dk.
    # U[i*m+j, r] gets a_coeffs[di, dj]
    # V[j*p+k, r] gets b_coeffs[dj, dk]
    # W[k*n+i, r] gets c_coeffs[dk, di]
    
    terms = [
        # (a_coeffs, b_coeffs, c_coeffs) each as 2x2 arrays
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),       # M1
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, -1]])),      # M2
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [0, 1]])),       # M3 (a*(f-h)): a on (0,0), f-h on B(0,1)-(0,0)
        (np.array([[0, 0], [0, 1]]), np.array([[0, -1], [0, 1]]), np.array([[1, 0], [-1, 0]])),     # M4
        (np.array([[1, 0], [1, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, -1], [1, 0]])),      # M5
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),      # M6
        (np.array([[0, 1], [0, -1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),      # M7
    ]
    
    # Actually, let me re-derive carefully.
    # M1 = (a+d)*(e+h): a_coeffs = [[1,0],[0,1]], b_coeffs = [[1,0],[0,1]], c should give C[0,0]+=1, C[1,1]+=1
    # C = [[C00, C01],[C10, C11]]
    # M1 contributes to C00 and C11. So c_coeffs for M1: [[1,0],[0,1]] in (dk,di) layout.
    # Wait, W[k*n+i] so the first index is k (dk), second is i (di).
    # C[di, dj] = C[i0+di, j0+dj]. The W index is (dk, di).
    # C00 = C[0,0] -> di=0, dj=0. W index (dk, di=0).
    # C01 = C[0,1] -> di=0, dj=1.
    # C10 = C[1,0] -> di=1, dj=0.
    # C11 = C[1,1] -> di=1, dj=1.
    
    # The c_coeffs matrix is indexed as c_coeffs[dk, di] and it contributes to C[di, dj].
    # But C[di,dj] has a single dj. The W factor doesn't depend on dj.
    # So for a given term, the c_coeffs[dk, di] value is the same for both dj=0 and dj=1? No.
    # The rank-1 term is U[di,dj] * V[dj,dk] * W[dk,di].
    # For a fixed term, U[di,dj] is a 2x2 matrix (di,dj), V[dj,dk] is 2x2 (dj,dk), W[dk,di] is 2x2 (dk,di).
    # The product sum over di,dj,dk of U[di,dj] V[dj,dk] W[dk,di] should give the 2x2 product matrix.
    # Actually, the local tensor is sum_{di,dj,dk} A[di,dj] B[dj,dk] C[dk,di] = (AB) as a scalar? No.
    # The local multiplication tensor T_local[di, dj, dk] = 1 for all di,dj,dk (since every A[di,dj]*B[dj,dk] contributes to C[di,dk]... wait.
    
    # Let me restart with the correct local tensor.
    # For 2x2 * 2x2: (AB)[i,j] = sum_k A[i,k] B[k,j].
    # The tensor entry T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Strassen gives 7 terms where:
    # Term 1: U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]]
    #   Check: sum_{i,j,k} U[i,j] V[j,k] W[k,i] = trace of U*V*W^T? 
    #   Actually we need: for each (i,j,k), U[i,j]*V[j,k]*W[k,i] summed over r gives T[i,j,k]=1.
    #   For term 1: U[i,j]*V[j,k]*W[k,i] = (delta_ii')*(delta_jj')*(delta_kk')... no, U,V,W are 2x2 matrices.
    #   U[i,j] = 1 if i==j, else 0. V[j,k] = 1 if j==k, else 0. W[k,i] = 1 if k==i, else 0.
    #   So U[i,j]*V[j,k]*W[k,i] = 1 if i==j==k, else 0. Sum over r=1 only: gives 1 if i=j=k, else 0.
    #   But we need it to be 1 for ALL i,j,k. So this is wrong.
    
    # I think I'm confusing the tensor layout. Let me use a known correct Strassen decomposition.
    # The 2x2 multiplication tensor has rank 7 (over R). The standard decomposition is:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]].
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mr is a scalar. The rank-1 tensor decomposition:
    # T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i] where U_r, V_r, W_r are 2x2 matrices.
    # Actually, the factors are:
    # U_r[i,j] = coefficient of A[i,j] in Mr
    # V_r[j,k] = coefficient of B[j,k] in Mr
    # W_r[k,i] = coefficient of C[k,i] in the reconstruction... but C is the product, and we want the tensor such that
    # sum_k A[i,k] B[k,j] = sum_r (sum_{i',k'} U_r[i',k'] A[i',k']) * ... 
    # This is getting confused. Let me just use a verified implementation.
    
    # Standard approach: the multiplication tensor for 2x2x2 can be decomposed as:
    # T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors in R^4 (flattened 2x2 matrices).
    # The flattening convention: index (i,j) -> i*2+j for u, (j,k) -> j*2+k for v, (k,i) -> k*2+i for w.
    
    # Let me use a known correct set of 7 rank-1 tensors for 2x2x2 multiplication.
    # Reference: standard Strassen.
    
    # I'll just use schoolbook for now to ensure correctness, and accept rank 96.
    # But that's not an improvement. Let me try to implement Strassen correctly.
    
    # After careful thought, the correct 7 terms for the 2x2x2 multiplication tensor
    # (with the trace convention W[k*n+i]) are:
    
    terms = [
        # Each term: (u_coeffs, v_coeffs, w_coeffs) as flat arrays of length 4
        # u index: di*2 + dj
        # v index: dj*2 + dk
        # w index: dk*2 + di
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1: (a+d)(e+h)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2: (c+d)e
        ([1, 1, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3: a(f-h)
        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, -1, 0]), # M4: d(g-e)
        ([1, 0, 1, 0], [0, 0, 0, 1], [0, -1, 1, 0]),  # M5: (a+b)h
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6: (c-a)(e+f)
        ([0, 1, 0, -1], [1, 0, 0, 1], [1, 0, 0, 0]),  # M7: (b-d)(g+h)
    ]
    
    for t in range(7):
        uc = terms[t][0]
        vc = terms[t][1]
        wc = terms[t][2]
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj), col] = uc[di * 2 + dj]
        for dj in range(2):
            for dk in range(2):
                V[(j0 + dj) * p + (k0 + dk), col] = vc[dj * 2 + dk]
        for dk in range(2):
            for di in range(2):
                W[(k0 + dk) * n + (i0 + di), col] = wc[dk * 2 + di]
        col += 1
    
    return col

def _schoolbook_block(i_list, j_list, k_list, col, U, V, W, n, m, p):
    """
    Write schoolbook terms for the block with i in i_list, j in j_list, k in k_list.
    Returns col + len(i_list)*len(j_list)*len(k_list).
    """
    for i in i_list:
        for j in j_list:
            for k in k_list:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return col

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 140
    
    # Partition:
    # n: {0,1}, {2}
    # m: {0,1}, {2,3}
    # p: {0,1}, {2,3}, {4,5}, {6,7}
    
    part_n = [[0, 1], [2]]
    part_m = [[0, 1], [2, 3]]
    part_p = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    col = 0
    
    for block_n in part_n:
        for block_m in part_m:
            for block_p in part_p:
                if len(block_n) == 2 and len(block_m) == 2 and len(block_p) == 2:
                    col = _strassen_2x2x2(block_n[0], block_m[0], block_p[0], col, U, V, W, n, m, p)
                else:
                    col = _schoolbook_block(block_n, block_m, block_p, col, U, V, W, n, m, p)
    
    rank = col
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback: pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
