"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication.
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 if and only if the (i,k) entry of A@B uses A[i,j]*B[j,k],
    i.e., T is the standard matrix multiplication tensor: T[i,j,k]=1 always for valid indices.
    Actually, the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k (it's the
    "multiplication" tensor where each entry is the product of one A element and one B element).
    So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the tensor: for each (i,j,k), compute the contraction
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as: for each c, outer product of U3[:,:,c], V3[:,:,c], W3[:,:,c]
    # But that's expensive. Instead, for each (i,j), compute the vector over k.
    # T[i,j,:] = sum_c U3[i,j,c] * V3[j,:,c] * W3[:,i,c]
    # Let's do it efficiently:
    # For fixed i,j: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = sum_c U3[i,j,c] * (V3[j,:,c] * W3[:,i,c])
    # We can precompute for each c: M_c[j,k] = V3[j,k,c] * W3[k,j... no, W3[k,i,c]
    # This is tricky. Let's just do it with broadcasting for small sizes.
    # n=3, m=5, p=6, rank up to ~90
    # T has shape (n, m, p) = (3, 5, 6) = 90 elements
    # For each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    # The matrix multiplication tensor should be all 1s
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2x2():
    """Return a rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4, M1-M2+M3+M6]]
    
    # We need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # such that sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = C[i,k]
    # 
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # Let's define the 7 terms and their contributions to each C[i,k]
    # Term c=0: M1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    #   U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    #   Contributes to C[0,0] and C[1,1]
    #   W[k*n+i]: for C[0,0]: k=0,i=0 -> W[0]; for C[1,1]: k=1,i=1 -> W[3]
    #   Actually W is indexed as W[k*2+i], so W[0*2+0]=W[0], W[1*2+1]=W[3]
    
    # Let me think more carefully.
    # U[i*2+j, c]: i in {0,1}, j in {0,1}
    # V[j*2+k, c]: j in {0,1}, k in {0,1}
    # W[k*2+i, c]: k in {0,1}, i in {0,1}
    
    # C[i,k] = sum_j sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    # But the tensor is defined such that T[i,j,k] = sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    # and we need T[i,j,k] to represent the multiplication tensor.
    
    # Actually, the matrix multiplication tensor for 2x2x2 has T[i,j,k]=1 for all i,j,k
    # because C[i,k] = sum_j A[i,j]*B[j,k].
    # So we need: for all i,j,k in {0,1}: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    # This is the standard 2x2x2 matrix multiplication tensor, which has rank 7 (Strassen).
    
    # Let me use a known Strassen decomposition.
    # One standard form:
    # c=0: U: (1,0,0,1) V: (1,0,0,1) W: (1,0,0,1) -> M1
    # Wait, let me use the explicit factor matrices from literature.
    
    # A common Strassen decomposition for 2x2x2:
    # U (4x7), V (4x7), W (4x7)
    #
    # Let me construct it from the 7 products:
    # M1 = (a+d)(e+h): U has 1 at (0,0) and (1,1) -> U[0]=1,U[3]=1; V has 1 at (0,0) and (1,1) -> V[0]=1,V[3]=1; W needs to contribute to C[0,0] and C[1,1]
    #   C[0,0] gets M1, C[1,1] gets M1
    #   W[k*2+i]: for (i=0,k=0): W[0]; for (i=1,k=1): W[3]
    #   So W[0]=1, W[3]=1 for this term.
    #
    # M2 = (c+d)e: U has 1 at (1,0) and (1,1) -> U[2]=1,U[3]=1; V has 1 at (0,0) -> V[0]=1; 
    #   C[1,0] gets M2: (i=1,k=0) -> W[0*2+1]=W[1]
    #   So W[1]=1
    #
    # M3 = a(f-h): U has 1 at (0,0) -> U[0]=1; V has 1 at (0,1) and (1,1) -> V[1]=1,V[3]=-1;
    #   C[0,1] gets M3: (i=0,k=1) -> W[1*2+0]=W[2]
    #   So W[2]=1
    #
    # M4 = d(g-e): U has 1 at (1,1) -> U[3]=1; V has 1 at (1,0) and (0,0) -> V[2]=1,V[0]=-1;
    #   C[0,0] gets M4: W[0]; C[1,0] gets M4: W[1]
    #   So W[0]=1, W[1]=1
    #
    # M5 = (a+b)h: U has 1 at (0,0) and (0,1) -> U[0]=1,U[1]=1; V has 1 at (1,1) -> V[3]=1;
    #   C[0,1] gets M5: W[2]; C[1,1] gets -M5: W[3]=-1
    #   So W[2]=1, W[3]=-1
    #
    # M6 = (c-a)(e+f): U has 1 at (1,0) and (0,0) -> U[2]=1,U[0]=-1; V has 1 at (0,0) and (0,1) -> V[0]=1,V[1]=1;
    #   C[1,1] gets M6: W[3]
    #   So W[3]=1
    #
    # M7 = (b-d)(g+h): U has 1 at (0,1) and (1,1) -> U[1]=1,U[3]=-1; V has 1 at (1,0) and (1,1) -> V[2]=1,V[3]=1;
    #   C[0,0] gets M7: W[0]
    #   So W[0]=1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = -1
    
    # M6: (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist()

def _block_strassen():
    """Use Strassen on 2x2 blocks where possible, schoolbook elsewhere.
    For 3x5x6, we can partition into blocks.
    A is 3x5, B is 5x6, C is 3x6.
    
    One approach: split into 2x2, 1x2, 2x1, 1x1 blocks for A and B.
    A: rows split as [2,1], cols as [2,2,1] or [2,3]
    B: rows split to match A's cols, cols split as [3,3] or [2,2,2]
    
    Let's try: A rows [2,1], A cols [2,3]; B rows [2,3], B cols [3,3]
    Then C rows [2,1], C cols [3,3]
    
    Block (0,0): 2x2 * 2x3 -> need Strassen-like for 2x2x3? Not standard.
    
    Alternative: use the fact that we can apply Strassen to 2x2x2 subproblems
    and handle the rest with schoolbook.
    
    For 3x5x6:
    - Split A into top 2 rows and bottom 1 row: A1 (2x5), A2 (1x5)
    - Split B into left 5 cols... B is 5x6.
    
    Let me think of it as: C = A@B where A is 3x5, B is 5x6.
    Split the 5 intermediate dimension: j = j1 + j2 where j1 has 2 elements, j2 has 3 elements.
    A = [A1 | A2] where A1 is 3x2, A2 is 3x3
    B = [B1; B2] where B1 is 2x6, B2 is 3x6
    C = A1@B1 + A2@B2
    
    A1@B1: 3x2 * 2x6 = 3x6, rank = 3*2*6 = 36 schoolbook
    A2@B2: 3x3 * 3x6 = 3x6, rank = 3*3*6 = 54 schoolbook
    Total: 90 (same as schoolbook)
    
    Now apply Strassen to sub-blocks:
    A1 (3x2): split rows into [2,1], cols [2]
    B1 (2x6): split rows [2], cols [3,3]
    
    A1_top (2x2) @ B1 (2x6): 
      Split B1 cols into [3,3]: B1a (2x3), B1b (2x3)
      A1_top @ B1a: 2x2*2x3, A1_top @ B1b: 2x2*2x3
      For 2x2*2x3: we can use Strassen on the 2x2 part of A and split B.
      Actually, 2x2*2x3 has rank 12 schoolbook. Can we do better?
      The tensor is 2x2x3. Its rank is... not sure. Let's just use schoolbook for non-square.
    
    This is getting complicated. Let me try a different approach.
    
    Actually, the known result from the cell finding says "Block Strassen decomposition reduces rank from 90 to ~84".
    Let me just implement a simple version: use Strassen for 2x2x2 sub-blocks and schoolbook for the rest.
    
    For 3x5x6, partition:
    A: 3x5 -> rows [2,1], cols [2,2,1]
    B: 5x6 -> rows [2,2,1], cols [2,2,2]
    C: 3x6 -> rows [2,1], cols [2,2,2]
    
    Block (i,j) of A has shape (row_size_i, col_size_j)
    Block (j,k) of B has shape (row_size_j, col_size_k)
    Product contributes to C block (i,k) of shape (row_size_i, col_size_k)
    
    For each (i,j,k) triple, we multiply A[i,j] (ri x cj) by B[j,k] (cj x ck) to get a contribution to C[i,k] (ri x ck).
    
    If ri=2, cj=2, ck=2: use Strassen (rank 7 instead of 8)
    Otherwise: schoolbook (rank ri*cj*ck)
    
    Let me enumerate:
    row_sizes = [2, 1]
    col_sizes_A = [2, 2, 1]  # these are the intermediate dimension splits
    col_sizes_B = [2, 2, 2]
    
    For i in range(2), j in range(3), k in range(3):
      ri = row_sizes[i], cj = col_sizes_A[j], ck = col_sizes_B[k]
      If ri==2 and cj==2 and ck==2: rank 7
      Else: rank ri*cj*ck
    
    Total rank = sum over all (i,j,k) of the rank for that block.
    
    Let's compute:
    i=0 (ri=2):
      j=0 (cj=2): k=0(ck=2): 7, k=1(ck=2): 7, k=2(ck=2): 7 -> 21
      j=1 (cj=2): k=0(ck=2): 7, k=1(ck=2): 7, k=2(ck=2): 7 -> 21
      j=2 (cj=1): k=0(ck=2): 2*1*2=4, k=1: 4, k=2: 4 -> 12
      i=0 total: 21+21+12 = 54
    i=1 (ri=1):
      j=0 (cj=2): k=0: 1*2*2=4, k=1: 4, k=2: 4 -> 12
      j=1 (cj=2): k=0: 4, k=1: 4, k=2: 4 -> 12
      j=2 (cj=1): k=0: 1*1*2=2, k=1: 2, k=2: 2 -> 6
      i=1 total: 12+12+6 = 30
    
    Total: 54+30 = 84. 
    
    Now I need to construct the actual factor matrices.
    """
    n, m, p = DIMENSIONS
    row_sizes = [2, 1]
    mid_sizes = [2, 2, 1]
    col_sizes = [2, 2, 2]
    
    # Precompute Strassen 2x2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Build the full factor matrices
    # First, compute total rank
    total_rank = 0
    block_ranks = {}
    for i in range(len(row_sizes)):
        for j in range(len(mid_sizes)):
            for k in range(len(col_sizes)):
                ri, cj, ck = row_sizes[i], mid_sizes[j], col_sizes[k]
                if ri == 2 and cj == 2 and ck == 2:
                    r = 7
                else:
                    r = ri * cj * ck
                block_ranks[(i, j, k)] = r
                total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i in range(len(row_sizes)):
        for j in range(len(mid_sizes)):
            for k in range(len(col_sizes)):
                ri, cj, ck = row_sizes[i], mid_sizes[j], col_sizes[k]
                r = block_ranks[(i, j, k)]
                
                # Determine the index ranges
                i_start = sum(row_sizes[:i])
                j_start = sum(mid_sizes[:j])
                k_start = sum(col_sizes[:k])
                
                if ri == 2 and cj == 2 and ck == 2:
                    # Use Strassen
                    for c in range(7):
                        # U indices: (i_start + ii, j_start + jj) for ii in 0,1; jj in 0,1
                        for ii in range(2):
                            for jj in range(2):
                                row_u = (i_start + ii) * m + (j_start + jj)
                                U[row_u][col_offset + c] += S_U[ii * 2 + jj][c]
                        # V indices: (j_start + jj, k_start + kk) for jj in 0,1; kk in 0,1
                        for jj in range(2):
                            for kk in range(2):
                                row_v = (j_start + jj) * p + (k_start + kk)
                                V[row_v][col_offset + c] += S_V[jj * 2 + kk][c]
                        # W indices: (k_start + kk, i_start + ii) for kk in 0,1; ii in 0,1
                        for kk in range(2):
                            for ii in range(2):
                                row_w = (k_start + kk) * n + (i_start + ii)
                                W[row_w][col_offset + c] += S_W[kk * 2 + ii][c]
                else:
                    # Schoolbook for this block
                    for ii in range(ri):
                        for jj in range(cj):
                            for kk in range(ck):
                                row_u = (i_start + ii) * m + (j_start + jj)
                                row_v = (j_start + jj) * p + (k_start + kk)
                                row_w = (k_start + kk) * n + (i_start + ii)
                                U[row_u][col_offset] = 1
                                V[row_v][col_offset] = 1
                                W[row_w][col_offset] = 1
                                col_offset += 1
                    continue
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the block Strassen approach
    try:
        result = _block_strassen()
        rank = len(result["U"][0])
        if _verify(result["U"], result["V"], result["W"], n, m, p, rank):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
