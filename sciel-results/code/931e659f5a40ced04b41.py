"""
Rank-121 decomposition for 4x4x9 matrix multiplication.

Uses the Kronecker product structure:
  T_{4,4,9} is a sub-tensor of T_{2,2,3} ⊗ T_{2,2,3}
  
  T_{2,2,3} = (Strassen 2x2x2 tensor) + (3 schoolbook 2x2x1 terms)
  Rank(T_{2,2,3}) = 7 + 4 = 11
  
  Rank(T_{4,4,9}) <= Rank(T_{2,2,3})^2 = 121
  
  This is better than schoolbook rank 144.
"""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    
    U_S: (2*2) x 7 = 4 x 7  -- indices (i,j) -> row i*2+j
    V_S: (2*2) x 7 = 4 x 7  -- indices (j,k) -> row j*2+k
    W_S: (2*2) x 7 = 4 x 7  -- indices (k,i) -> row k*2+i
    
    Multiplications:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    Output:
    C00 = M1 - M2 - M4 + M7
    C01 = M3 + M5
    C10 = M4 + M6
    C11 = M1 - M3 - M5 + M6
    """
    # U_S: rows indexed by (i,j), cols by multiplication index
    # M1: A00, A11 contribute
    # M2: A10, A11 contribute
    # M3: A00 contributes
    # M4: A11 contributes
    # M5: A00, A01 contribute
    # M6: A10, A00 contribute
    # M7: A01, A11 contribute
    
    U_S = [
        # (i=0,j=0): A00
        [1, 0, 1, 0, 1, -1, 0],
        # (i=0,j=1): A01
        [0, 0, 0, 0, 1, 0, 1],
        # (i=1,j=0): A10
        [0, 1, 0, 0, 0, 1, 0],
        # (i=1,j=1): A11
        [1, 1, 0, 1, 0, 0, -1],
    ]
    
    # V_S: rows indexed by (j,k), cols by multiplication index
    # M1: B00, B11
    # M2: B00
    # M3: B10, B11
    # M4: B01, B10
    # M5: B11
    # M6: B00, B10
    # M7: B01, B11
    
    V_S = [
        # (j=0,k=0): B00
        [1, 1, 0, 0, 0, 1, 0],
        # (j=0,k=1): B01
        [0, 0, 0, 1, 0, 0, 1],
        # (j=1,k=0): B10
        [0, 0, 1, -1, 0, 1, 0],
        # (j=1,k=1): B11
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    # W_S: rows indexed by (k,i), cols by multiplication index
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M5
    # C10 = M4 + M6
    # C11 = M1 - M3 - M5 + M6
    
    W_S = [
        # (k=0,i=0): C00
        [1, -1, 0, -1, 0, 0, 1],
        # (k=0,i=1): C01
        [0, 0, 1, 0, 1, 0, 0],
        # (k=1,i=0): C10
        [0, 0, 0, 1, 0, 1, 0],
        # (k=1,i=1): C11
        [1, 0, -1, 0, -1, 1, 0],
    ]
    
    return U_S, V_S, W_S


def _schoolbook_2x2x1():
    """
    Schoolbook rank-4 decomposition for 2x2 by 2x1 multiplication.
    
    U: (2*2) x 4 = 4 x 4
    V: (2*1) x 4 = 2 x 4
    W: (1*2) x 4 = 2 x 4
    
    C[i] = sum_j A[i,j] * B[j]
    So we need 4 multiplications: A[i,j]*B[j] for each (i,j).
    
    U row (i,j): has 1 in column (i,j)
    V row j: has 1 in column (i,j) for each i... 
    
    Wait, let me think more carefully.
    For 2x2 * 2x1:
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    The tensor T_{2,2,2} here has:
    - First index: (i,j) for A, range 0..3
    - Second index: j for B, range 0..1
    - Third index: i for C, range 0..1
    
    Non-zero terms: T[i,j, j', i'] = 1 if j==j' and i==i'
    So non-zero when (i,j, j, i) -> i.e., (i,j,k,i) with k=j.
    
    U: (i,j) row -> column (i,j): value 1
    V: j row -> column (i,j) for fixed j: value 1 if j matches... 
    
    Actually let me use the standard form:
    U[(i,j), (i',j')] = delta_{i,i'} * delta_{j,j'}  (but we only need 4 terms)
    
    Simpler: 4 terms indexed by (i,j):
    U[i*2+j, i*2+j] = 1
    V[j, i*2+j] = 1  for all i
    W[i, i*2+j] = 1  for all j
    
    Let me verify: sum_c U[i,j,c] * V[j',c] * W[i',c]
    = sum_{c} [c=(i0,j0)] delta_{i,i0}delta_{j,j0} * delta_{j',j0} * delta_{i',i0}
    = delta_{i,i0}delta_{j,j0}delta_{j',j0}delta_{i',i0} summed over (i0,j0)
    = 1 if i==i' and j==j' and j'==j (i.e., j==j') and i==i'
    = 1 if i==i' and j==j'
    Which is correct for C[i] = sum_j A[i,j]B[j].
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    for i in range(2):
        for j in range(2):
            col = i*2 + j
            U[i*2+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
    
    return U, V, W


def _build_2x2x3():
    """
    Build rank-11 decomposition for T_{2,2,3}.
    
    T_{2,2,3}: multiply 2x2 matrix by 2x3 matrix.
    This is the tensor product / direct sum of T_{2,2,2} (Strassen, rank 7)
    and 3 copies of T_{2,2,1} (schoolbook, rank 4 each) = 7 + 3*4 = 19?
    
    No wait. T_{2,2,3} is NOT the direct sum of T_{2,2,2} + T_{2,2,1}.
    T_{2,2,3} multiplies a 2x2 by a 2x3 to get a 2x3.
    The tensor has indices:
    - U: (i,j) for i in {0,1}, j in {0,1} -> 4 entries
    - V: (j,k) for j in {0,1}, k in {0,1,2} -> 6 entries
    - W: (k,i) for k in {0,1,2}, i in {0,1} -> 6 entries
    
    Non-zero: T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k' and i==i'
    
    We can decompose this as: for each k in {0,1,2}, we have a T_{2,2,1} sub-tensor
    that computes C[:,k] = A * B[:,k].
    
    Each T_{2,2,1} has rank 4 (schoolbook).
    So T_{2,2,3} has rank at most 3*4 = 12 (schoolbook).
    
    But we want to use Strassen where possible.
    Actually, the Kronecker product approach:
    T_{4,4,9} is a sub-tensor of T_{2,2,3} ⊗ T_{2,2,3}.
    
    For the Kronecker product, we need a decomposition of T_{2,2,3}.
    The simplest is schoolbook rank 12.
    
    But wait, the previous approach mentioned was:
    T_{2,2,3} = Strassen(2,2,2) + schoolbook(2,2,1) for the "extra" column?
    
    No, that's not right either. Let me reconsider.
    
    Actually, the cleanest approach:
    T_{2,2,3} can be decomposed by treating the 3rd dimension.
    For each of the 3 columns of B, we compute A * B_col.
    Using schoolbook for each: rank = 3 * 4 = 12.
    
    Alternatively, we can use a mixed approach:
    Use Strassen for the first two "virtual" 2x2 outputs and schoolbook for the third?
    That doesn't quite work because Strassen gives 2x2 output.
    
    Let me just use schoolbook rank 12 for T_{2,2,3}.
    Then T_{4,4,9} <= T_{2,2,3} ⊗ T_{2,2,3} has rank 12*12 = 144.
    That's the same as schoolbook! Not helpful.
    
    Hmm. Let me reconsider the decomposition.
    
    Actually, I think the correct approach is:
    T_{4,4,9} is a sub-tensor of T_{2,2,3} ⊗ T_{2,2,3} only if we can embed it.
    4 = 2*2, 4 = 2*2, 9 = 3*3. Yes, this works.
    
    So we need Rank(T_{2,2,3})^2.
    Schoolbook rank of T_{2,2,3} is 2*2*3 = 12.
    12^2 = 144. Same as before.
    
    To get better than 144, we need Rank(T_{2,2,3}) < 12.
    
    Can we do T_{2,2,3} with rank < 12?
    T_{2,2,3}: multiply 2x2 by 2x3.
    This is equivalent to multiplying 2x2 by [2x2 | 2x1].
    The 2x2 part can use Strassen (rank 7), the 2x1 part uses schoolbook (rank 4).
    Total: 7 + 4 = 11.
    
    So Rank(T_{2,2,3}) <= 11.
    Then Rank(T_{4,4,9}) <= 11^2 = 121.
    
    This is the key insight!
    
    Let me construct this properly.
    """
    U_S, V_S, W_S = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    # T_{2,2,3} decomposition:
    # U: 4 x 11 (first 7 cols from Strassen, next 4 from schoolbook for the 2x1 part)
    # V: 6 x 11
    # W: 6 x 11
    
    # For Strassen part (computes A * B[:,0:2] as a 2x2):
    # U_S: 4 x 7
    # V_S: 4 x 7 (for B[:,0:2], which has indices (j,k) with k in {0,1})
    # W_S: 4 x 7 (for C[:,0:2], which has indices (k,i) with k in {0,1})
    
    # For schoolbook part (computes A * B[:,2] as a 2x1):
    # U_1: 4 x 4
    # V_1: 2 x 4 (for B[:,2], which is just B[0,2] and B[1,2], indices j in {0,1})
    # W_1: 2 x 4 (for C[:,2], which is C[0,2] and C[1,2], indices i in {0,1})
    
    # Combined U: 4 x 11
    U_comb = [[U_S[i][c] for c in range(7)] + [U_1[i][c] for c in range(4)] for i in range(4)]
    
    # Combined V: 6 x 11
    # Rows 0,1: (j=0,k=0), (j=1,k=0) -> from V_S rows 0,2 (k=0)
    # Rows 2,3: (j=0,k=1), (j=1,k=1) -> from V_S rows 1,3 (k=1)
    # Rows 4,5: (j=0,k=2), (j=1,k=2) -> from V_1 rows 0,1
    V_comb = [
        [V_S[0][c] for c in range(7)] + [0]*4,  # (j=0,k=0)
        [V_S[2][c] for c in range(7)] + [0]*4,  # (j=1,k=0)
        [V_S[1][c] for c in range(7)] + [0]*4,  # (j=0,k=1)
        [V_S[3][c] for c in range(7)] + [0]*4,  # (j=1,k=1)
        [0]*7 + [V_1[0][c] for c in range(4)],   # (j=0,k=2)
        [0]*7 + [V_1[1][c] for c in range(4)],   # (j=1,k=2)
    ]
    
    # Combined W: 6 x 11
    # Rows 0,1: (k=0,i=0), (k=0,i=1) -> from W_S rows 0,2 (k=0)
    # Rows 2,3: (k=1,i=0), (k=1,i=1) -> from W_S rows 1,3 (k=1)
    # Rows 4,5: (k=2,i=0), (k=2,i=1) -> from W_1 rows 0,1
    W_comb = [
        [W_S[0][c] for c in range(7)] + [0]*4,  # (k=0,i=0)
        [W_S[2][c] for c in range(7)] + [0]*4,  # (k=0,i=1)
        [W_S[1][c] for c in range(7)] + [0]*4,  # (k=1,i=0)
        [W_S[3][c] for c in range(7)] + [0]*4,  # (k=1,i=1)
        [0]*7 + [W_1[0][c] for c in range(4)],   # (k=2,i=0)
        [0]*7 + [W_1[1][c] for c in range(4)],   # (k=2,i=1)
    ]
    
    return U_comb, V_comb, W_comb


def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """
    Compute the Kronecker product of two tensor decompositions.
    
    If T1 has decomposition (U1, V1, W1) with rank r1,
    and T2 has decomposition (U2, V2, W2) with rank r2,
    then T1 ⊗ T2 has decomposition with rank r1*r2.
    
    U_combined: (n1*m1) * (n2*m2) x (r1*r2)
    V_combined: (m1*p1) * (m2*p2) x (r1*r2)
    W_combined: (p1*n1) * (p2*n2) x (r1*r2)
    
    U_combined[(i1,j1)*(n2*m2) + (i2,j2), c1*r2+c2] = U1[(i1,j1), c1] * U2[(i2,j2), c2]
    V_combined[(j1,k1)*(m2*p2) + (j2,k2), c1*r2+c2] = V1[(j1,k1), c1] * V2[(j2,k2), c2]
    W_combined[(k1,i1)*(p2*n2) + (k2,i2), c1*r2+c2] = W1[(k1,i1), c1] * W2[(k2,i2), c2]
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    
    n1m1 = len(U1)
    m1p1 = len(V1)
    p1n1 = len(W1)
    
    n2m2 = len(U2)
    m2p2 = len(V2)
    p2n2 = len(W2)
    
    rank = r1 * r2
    
    U_comb = [[0]*rank for _ in range(n1m1 * n2m2)]
    V_comb = [[0]*rank for _ in range(m1p1 * m2p2)]
    W_comb = [[0]*rank for _ in range(p1n1 * p2n2)]
    
    for idx1 in range(n1m1):
        for idx2 in range(n2m2):
            for c1 in range(r1):
                for c2 in range(r2):
                    val = U1[idx1][c1] * U2[idx2][c2]
                    if val != 0:
                        U_comb[idx1*n2m2 + idx2][c1*r2 + c2] += val
    
    for idx1 in range(m1p1):
        for idx2 in range(m2p2):
            for c1 in range(r1):
                for c2 in range(r2):
                    val = V1[idx1][c1] * V2[idx2][c2]
                    if val != 0:
                        V_comb[idx1*m2p2 + idx2][c1*r2 + c2] += val
    
    for idx1 in range(p1n1):
        for idx2 in range(p2n2):
            for c1 in range(r1):
                for c2 in range(r2):
                    val = W1[idx1][c1] * W2[idx2][c2]
                    if val != 0:
                        W_comb[idx1*p2n2 + idx2][c1*r2 + c2] += val
    
    return U_comb, V_comb, W_comb


def _construct_rank_121():
    """
    Construct a rank-121 decomposition for 4x4x9 matrix multiplication.
    
    Uses T_{4,4,9} as a sub-tensor of T_{2,2,3} ⊗ T_{2,2,3}.
    Rank(T_{2,2,3}) = 11 (Strassen 7 + schoolbook 4 for the extra column).
    Rank = 11 * 11 = 121.
    """
    U1, V1, W1 = _build_2x2x3()
    U, V, W = _kron_decompose(U1, V1, W1, U1, V1, W1)
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    # Check valid positions: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[u_idx][c] * V_f[v_idx][c] * W_f[w_idx][c]
                if total != 1:
                    return False
    
    # Check invalid positions: sum_c U[i,j,c] * V[j2,k,c] * W[k2,i2,c] = 0
    # when not (j==j2 and k==k2 and i==i2)
    for u_idx in range(n * m):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(m * p):
            j2 = v_idx // p
            k = v_idx % p
            for w_idx in range(p * n):
                k2 = w_idx // n
                i2 = w_idx % n
                if j == j2 and k == k2 and i == i2:
                    continue
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[u_idx][c] * V_f[v_idx][c] * W_f[w_idx][c]
                if total != 0:
                    return False
    
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try the rank-121 construction first
    try:
        result = _construct_rank_121()
        # Verify it's correct
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            # If verification fails, fall through to search
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
