"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    Ur = U.reshape(n, m, r) if isinstance(U, np.ndarray) else np.array(U).reshape(n, m, r)
    Vr = V.reshape(m, p, r) if isinstance(V, np.ndarray) else np.array(V).reshape(m, p, r)
    Wr = W.reshape(p, n, r) if isinstance(W, np.ndarray) else np.array(W).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """
    Verified rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0] and C[1,1]
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1] and C[1,1]
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0] and C[0,1]
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _schoolbook_3x3():
    """
    Schoolbook rank-27 decomposition for 3x3x3 matrix multiplication.
    U: (9,27), V: (9,27), W: (9,27)
    """
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[k * 3 + i, col] = 1.0
                col += 1
    return U, V, W


def _liu_3x3():
    """
    Rank-15 decomposition for 3x3 matrix multiplication using Winograd's F(3,3) framework.
    
    The key insight: 3x3 matrix multiplication can be viewed as a 1D convolution problem.
    For each pair of rows/cols, we need to compute dot products. Winograd's F(3,3) 
    convolution algorithm computes y = conv(x, h) for len(x)=3, len(h)=3 using only 
    5 multiplications instead of 3*3=9.
    
    However, matrix multiplication is NOT simply a 1D convolution - it's a 2D operation.
    The correct approach: decompose 3x3 matmul into 3 independent 1D convolutions 
    (one per column of B / row of A), each of which uses F(3,3) with rank 5.
    Total rank = 3 * 5 = 15.
    
    Wait - this doesn't quite work directly because matrix multiplication mixes 
    rows and columns in a way that's not separable into independent 1D convolutions.
    
    Let me reconsider. The correct Winograd approach for 3x3 matmul:
    - Pad A and B to 5x5 (or use a different embedding)
    - Actually, the standard approach is to note that C = A*B where each element
      C[i,k] = sum_j A[i,j]*B[j,k]
    - This is a 3-term dot product for each (i,k) pair
    - Winograd F(3,3) can compute 3-term dot products with 5 multiplications
    - But we need 9 such dot products (one for each i,k pair)
    - If they share structure, we can do better
    
    Actually, the correct known result: rank(3x3) = 19 (Liu's algorithm).
    Rank-15 is NOT known to exist for 3x3 matrix multiplication over reals.
    
    Let me try a different approach: use the fact that 6x6 can be decomposed
    differently. Instead of Kronecker(2x2 rank 7, 3x3 rank 27) = 189,
    maybe I can find a better 6x6 decomposition directly.
    
    For now, return None to indicate we don't have a valid rank-15 3x3 decomposition.
    """
    return None


def _winograd_f33_3x3():
    """
    Attempt a rank-15 decomposition for 3x3 matrix multiplication.
    
    Strategy: Use Winograd's F(3,3) convolution on each "slice" of the matrix product.
    
    For 3x3 matmul C = A @ B:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    This is a 3-point dot product for each (i,k) pair. There are 9 such pairs.
    
    Winograd F(3,3) computes y = x * h (3-point convolution) with 5 multiplies:
    m0 = (x0+x1+x2)(h0+h1+h2)
    m1 = (x0-x1+x2)(h0-h1+h2) 
    m2 = (x0-x2)(h0-h2)
    m3 = (x1-x2)(h1-h2)
    m4 = x2*h2
    
    y0 = m0 + m1 + m2 + m3
    y1 = m0 - m1 + 2*m3
    y2 = m0 + m1 + m2 - m3 + m4
    
    But matrix multiplication is not a simple 1D convolution - the indices are coupled.
    
    Alternative: embed 3x3 matmul into a larger convolution problem.
    
    Actually, let me try a direct construction. The tensor for 3x3 matmul has
    27 non-zero entries. We want to write it as sum of 15 rank-1 terms.
    
    This is equivalent to finding 15 triples of vectors (u_c, v_c, w_c) in R^9
    such that sum_c u_c ⊗ v_c ⊗ w_c = T where T[i,j,k] = 1 if j=k, else 0... 
    No, T[i,j,k] = delta(j,k) is wrong.
    
    T[i,j,k] = 1 if the (i,j) element of A multiplies with (j,k) element of B 
    to contribute to (i,k) element of C. So T[i,j,k] = 1 for all i,j,k? No.
    
    The multiplication tensor T satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2}.
    Wait no - T is a 3-tensor of size 3x3x3 where T[i,j,k] = 1 if the 
    (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    That's T[i,j,k] = 1 for all i,j,k! That's the all-ones tensor, which has rank 1.
    
    I'm confusing myself. Let me be precise.
    
    The matrix multiplication tensor M_{n,n,n} is defined by:
    M[i,j,k] = 1 if j is the "middle" index connecting A[i,j] and B[j,k] to C[i,k]
    
    More precisely, M is a 3-tensor where:
    M[i,j,k] = 1 for all i,j,k in {0,...,n-1}
    
    No wait. The tensor is:
    M[i,j,k] = delta(j, j) which is always 1? That can't be right.
    
    OK let me think again. The bilinear map is:
    (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a trilinear form when we include C:
    sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} (sum_l A[i,l]*B[l,k]) * A[i,j] * B[j,k]
    
    No, that's not right either. The tensor decomposition is:
    
    C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    where the sum over j is implicit in the tensor structure. The tensor T is:
    T[i,j,k] = 1 for all i,j,k (it's the "all ones" 3-tensor of size 3x3x3)
    
    And we want: sum_c U[:, :, c] ⊗ V[:, :, c] ⊗ W[:, :, c] = T (all ones)
    
    Wait, that's not right either. Let me look at the verify function:
    
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And we want T[i,j,k] = 1 for all i,j,k.
    
    So yes, we want to decompose the all-ones 3x3x3 tensor into a sum of 15 
    rank-1 tensors. But the all-ones tensor has rank 1! So rank-15 is trivially 
    achievable (just use rank 1).
    
    I'm clearly misunderstanding the tensor structure. Let me re-read the verify function.
    
    Ah I see - the indices are:
    U has shape (n*m, rank) = (9, rank), indexed as U[i*m+j, c]
    V has shape (m*p, rank) = (9, rank), indexed as V[j*p+k, c]  
    W has shape (p*n, rank) = (9, rank), indexed as W[k*n+i, c]
    
    And T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We want T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    
    So the tensor to decompose is indeed the all-ones 3x3x3 tensor.
    The all-ones tensor has rank 1, so we can trivially achieve rank 1!
    
    But that's clearly wrong because matrix multiplication has rank 19 for 3x3.
    
    OH WAIT. I think I see the issue. The tensor is NOT all ones. Let me re-examine.
    
    For matrix multiplication C = A @ B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition should give:
    C[i,k] = sum_c (sum_j U[i,j,c] * V[j,k,c]) * W[k,i,c]... no that's not right.
    
    Actually, looking at the verify function more carefully:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This means the tensor T has indices (i, j, k) and we want T[i,j,k] = 1 for ALL i,j,k.
    
    But that IS the all-ones tensor, which has rank 1. So something is wrong with my understanding.
    
    Let me look at the schoolbook example:
    For each (i,j,k), we set U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    
    So T[i,j,k] = sum over cols where all three are 1. In schoolbook, for each (i,j,k) 
    there's exactly one col where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 simultaneously.
    So T[i,j,k] = 1 for all i,j,k.
    
    This confirms the tensor is all-ones. But then why does 3x3 matmul have rank 19?
    
    I think the issue is that the "tensor" in tensor decomposition theory for matrix 
    multiplication is different from what's being verified here. The standard 
    multiplication tensor M_{n,n,n} has entries:
    M[i,j,k] = 1 if j=k (the "middle" index matches)
    
    No wait, that's not right either.
    
    Let me think about this differently. The bilinear map is:
    f(A, B) = A @ B
    
    This can be written as:
    f(A, B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor notation, if we flatten A, B, C:
    C = M(A, B)
    
    The tensor M has entries M[i,k,j] = delta(j,j) = 1? No...
    
    OK I think the correct formulation is:
    The multiplication tensor T has T[i,j,k] = 1 for all i,j,k, and the 
    bilinear map is:
    C[i,k] = sum_j (sum_c U[i,j,c] * V[j,k,c]) * W[k,i,c]
    
    No, that's still not matching the verify function.
    
    Let me just trust the verify function and the schoolbook implementation.
    The verify function checks that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    And schoolbook achieves this with rank n*m*p by having one column per (i,j,k) triple.
    
    So the tensor to decompose is indeed the all-ones tensor of size (n*m) x (m*p) x (p*n)
    with a specific index structure.
    
    For n=m=p=3, this is a 9x9x9 tensor with T[i*3+j, j*3+k, k*3+i] = 1 for all i,j,k,
    and 0 otherwise? No, the verify function sums over c and checks if the result is 1 
    for all i,j,k. It doesn't say anything about other entries.
    
    Actually, I think the tensor is defined on the "structured" indices only:
    T[i,j,k] = 1 for all i,j,k in {0,...,n-1}
    
    And the Kronecker product structure means that the full 9x9x9 tensor has 
    non-zero entries only at positions (i*3+j, j*3+k, k*3+i) for i,j,k in {0,1,2}.
    
    This is a 27-entry tensor (not all 729 entries are 1).
    
    So the tensor to decompose has 27 non-zero entries (all equal to 1) at specific 
    positions in a 9x9x9 space.
    
    For the Kronecker product approach:
    - 2x2 matmul tensor has 8 non-zero entries (all 1) at positions (i*2+j, j*2+k, k*2+i)
    - 3x3 matmul tensor has 27 non-zero entries (all 1) at positions (i*3+j, j*3+k, k*3+i)
    - Kronecker product has 8*27 = 216 non-zero entries
    
    And the 6x6 matmul tensor has 216 non-zero entries at positions (i*6+j, j*6+k, k*6+i).
    
    So the Kronecker product of two valid decompositions gives a valid decomposition 
    of the larger tensor.
    
    Now, for 3x3 matmul, we want to decompose the 27-entry tensor into rank-15 terms.
    This is a non-trivial problem. The known minimum rank is 19 (Liu's algorithm).
    
    So rank-15 is NOT possible for 3x3 matmul over the reals.
    
    Therefore, my strategy of using rank-15 for 3x3 is flawed. I should stick with 
    the known rank-19 algorithm or find another approach.
    
    Since implementing Liu's rank-19 algorithm is complex, and the current approach 
    with rank 189 is working, let me just return None here.
    """
    return None


def _kronecker_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compute the Kronecker product of two tensor decompositions.
    
    If (U1, V1, W1) decomposes n1 x m1 x p1 matrix multiplication with rank r1,
    and (U2, V2, W2) decomposes n2 x m2 x p2 matrix multiplication with rank r2,
    then the Kronecker product decomposes (n1*n2) x (m1*m2) x (p1*p2)
    matrix multiplication with rank r1 * r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    U = np.zeros((n1 * n2 * m1 * m2, rank), dtype=float)
    V = np.zeros((m1 * m2 * p1 * p2, rank), dtype=float)
    W = np.zeros((p1 * p2 * n1 * n2, rank), dtype=float)
    
    # U: index (i1*n2+i2)*(m1*m2) + (j1*m2+j2), column c1*r2+c2
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1_val = U1[i1 * m1 + j1, c1]
                if u1_val == 0:
                    continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u1_val != 0 and u2_val != 0:
                                row = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                                col = c1 * r2 + c2
                                U[row, col] = u1_val * u2_val
    
    # V: index (j1*m2+j2)*(p1*p2) + (k1*p2+k2), column c1*r2+c2
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1_val = V1[j1 * p1 + k1, c1]
                if v1_val == 0:
                    continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v1_val != 0 and v2_val != 0:
                                row = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                col = c1 * r2 + c2
                                V[row, col] = v1_val * v2_val
    
    # W: index (k1*p2+k2)*(n1*n2) + (i1*n2+i2), column c1*r2+c2
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1_val = W1[k1 * n1 + i1, c1]
                if w1_val == 0:
                    continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w1_val != 0 and w2_val != 0:
                                row = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                col = c1 * r2 + c2
                                W[row, col] = w1_val * w2_val
    
    return U, V, W


def _block_strassen_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    
                    col += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_3x3():
    """
    Attempt to construct a rank-19 decomposition for 3x3 matrix multiplication
    using Liu's algorithm or a similar approach.
    
    For now, return None if we can't construct a verified rank-19 decomposition.
    """
    return None


def _kronecker_strassen2_lu3():
    """
    Kronecker product of Strassen 2x2 (rank 7) and a 3x3 decomposition.
    If a rank-19 3x3 decomposition is available, total rank = 7*19 = 133.
    Otherwise, fall back to schoolbook 3x3 (rank 27), total rank = 7*27 = 189.
    """
    sU, sV, sW = _strassen_2x2()
    
    # Try to get a low-rank 3x3 decomposition
    lu3 = _strassen_3x3()
    if lu3 is not None:
        kU, kV, kW = lu3
        return _kronecker_product(sU, sV, sW, kU, kV, kW, 2, 2, 2, 3, 3, 3)
    
    # Fall back to schoolbook 3x3
    kU, kV, kW = _schoolbook_3x3()
    return _kronecker_product(sU, sV, sW, kU, kV, kW, 2, 2, 2, 3, 3, 3)


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27).
    Total rank = 7 * 27 = 189.
    
    If a lower-rank 3x3 decomposition is available, the total rank will be lower.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: Kronecker product of Strassen 2x2 and best available 3x3
    U_k, V_k, W_k = _kronecker_strassen2_lu3()
    
    # Verify quickly
    if _verify(U_k, V_k, W_k, 6, 6, 6):
        return {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    
    # Fallback: block Strassen
    U_b, V_b, W_b = _block_strassen_6x6x6()
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
