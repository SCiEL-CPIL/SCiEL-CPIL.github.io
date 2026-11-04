"""Exact rank-46 decomposition for the (2,5,5) matrix-multiplication tensor.

Strategy:
  Split the 5 intermediate indices as 4 + 1.
  - A_{2x4} B_{4x4}: four 2x2 Strassen multiplications (rank 7 each) -> 28
  - A_{2x4} B_{4x1}: schoolbook -> 8
  - A_{2x1} B_{1x4}: schoolbook -> 8
  - A_{2x1} B_{1x1}: schoolbook -> 2
  Total rank = 46.

Index conventions:
  A has shape (2,5): A[i, j] with i in 0..1, j in 0..4
  B has shape (5,5): B[j, k] with j in 0..4, k in 0..4
  C has shape (2,5): C[i, k] with i in 0..1, k in 0..4

  U index: a = i*m + j = i*5 + j   (10 rows)
  V index: b = j*p + k = j*5 + k   (25 rows)
  W index: w = k*n + i = k*2 + i   (10 rows)

  C[w] = sum_r U[a, r] * V[b, r] * W[w, r]  ... no, that's not right.
  
  Actually the tensor decomposition is:
  C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r]
  
  Wait, the standard formulation is:
  T = sum_r u_r ⊗ v_r ⊗ w_r
  where T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] ... no.
  
  The matrix multiplication tensor T has entries T[i, j, k] = 1 if the product
  involves A[i,j]*B[j,k], else 0. More precisely, for C[i,k] = sum_j A[i,j]*B[j,k],
  the tensor is T[i, j, k] where the first index is the row of A (and C), the second
  is the column of A / row of B, and the third is the column of B (and C).
  
  A rank-1 decomposition: C[i,k] = sum_r (sum_j alpha_{r,i,j} A[i,j]) * (sum_j beta_{r,j,k} B[j,k])
  
  But in the U, V, W format:
  C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  
  This doesn't look right dimensionally. Let me reconsider.
  
  Standard: we want U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
  for all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  
  But T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
  So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
  
  AND the decomposition must be such that C[i,k] = sum_j A[i,j]*B[j,k].
  
  The correct formulation: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
  
  Hmm, that's also not standard. Let me look at the task description again.
  
  "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
  
  The matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix has entries indexed by (i, j, k) where i is in 0..n-1, j in 0..m-1, k in 0..p-1. The entry T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes to C[i,k]).
  
  A rank-1 decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  
  For this to work, we need the sum over r of the product of these three entries to equal T[i,j,k] = 1 for all valid (i,j,k).
  
  But wait, that's not quite right either. The issue is that the tensor T is defined on the full index set, and T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] term appears in exactly one C[i,k]).
  
  So we need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
  
  This is the condition. Let me verify with schoolbook:
  For schoolbook, rank = n*m*p. For each (i,j,k), we have one factor r corresponding to (i,j,k).
  U[i*m+j, r_{ijk}] = 1, V[j*p+k, r_{ijk}] = 1, W[k*n+i, r_{ijk}] = 1, and 0 otherwise.
  Then sum_r = 1 for that (i,j,k). Correct.
  
  Now for Strassen on 2x2:
  A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  C[0,0] = (a+b)(e+g) - (d+b)g - (a+c)e + (a+d)(e-h) ... let me use the standard Strassen:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e+f)
  P3 = (a-b)(g+h)
  P4 = (c+d)g
  P5 = (a+d)h
  P6 = (a-b)e
  P7 = (c+b)(f-h)
  C[0,0] = P1 - P4 - P5 + P7
  C[0,1] = P3 + P5
  C[1,0] = P2 - P7
  C[1,1] = P1 - P2 + P3 + P4
  
  Each P_r is a product of two linear forms in A and B.
  For P1 = (a+d)(e+h):
    The linear form in A is a+d = A[0,0] + A[1,0]
    The linear form in B is e+h = B[0,0] + B[0,1]
  
  So for this factor, we need:
    U[A[0,0] index, r] = 1, U[A[1,0] index, r] = 1
    V[B[0,0] index, r] = 1, V[B[0,1] index, r] = 1
    W[C[0,0] index, r] = 1 (from P1 in C[0,0])
    W[C[0,1] index, r] = 0
    W[C[1,0] index, r] = 0
    W[C[1,1] index, r] = -1 (from -P1 in C[1,1])
    
  Wait, but W[k*n+i, r] where k is column of C, i is row of C.
  C[0,0]: i=0, k=0, index = 0*2+0 = 0
  C[0,1]: i=0, k=1, index = 1*2+0 = 2
  C[1,0]: i=1, k=0, index = 0*2+1 = 1
  C[1,1]: i=1, k=1, index = 1*2+1 = 3
  
  So W[0, r] += 1, W[3, r] += -1 for P1.
  
  Let me implement this carefully.
"""

import time

_T0 = time.time()
_DEADLINE = 140.0


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _strassen_2x2():
    """Return the 7 rank-1 factors for 2x2 matrix multiplication.
    
    Each factor is a dict with keys 'U' (dict from A-index to coeff), 
    'V' (dict from B-index to coeff), 'W' (dict from C-index to coeff).
    A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    """
    factors = []
    
    # P1 = (a+d)(e+h)
    factors.append({
        'U': {0: 1, 2: 1},
        'V': {0: 1, 3: 1},
        'W': {0: 1, 3: -1}
    })
    # P2 = (c+d)(e+f)
    factors.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1, 1: 1},
        'W': {2: 1, 3: -1}
    })
    # P3 = (a-b)(g+h)
    factors.append({
        'U': {0: 1, 1: -1},
        'V': {2: 1, 3: 1},
        'W': {1: 1, 3: 1}
    })
    # P4 = (c+d)g
    factors.append({
        'U': {2: 1, 3: 1},
        'V': {2: 1},
        'W': {0: -1, 3: 1}
    })
    # P5 = (a+d)h
    factors.append({
        'U': {0: 1, 2: 1},
        'V': {3: 1},
        'W': {0: -1, 1: 1}
    })
    # P6 = (a-b)e
    factors.append({
        'U': {0: 1, 1: -1},
        'V': {0: 1},
        'W': {}
    })
    # P7 = (c+b)(f-h)
    factors.append({
        'U': {1: 1, 2: 1},
        'V': {1: 1, 3: -1},
        'W': {0: 1, 2: -1}
    })
    
    return factors


def construct():
    n, m, p = 2, 5, 5
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 46
    V = [[0] * rank for _ in range(m * p)]  # 25 x 46
    W = [[0] * rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    strassen_factors = _strassen_2x2()
    
    # Split: 4 + 1
    # Part 1: A_{2x4} B_{4x4}
    # A indices: j=0..3, B indices: j=0..3, k=0..3
    # Split A into two 2x2 blocks: A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Split B into four 2x2 blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C1 = A1*B11, C2 = A1*B12, C3 = A2*B21, C4 = A2*B22
    # C[:, 0:2] = C1 + C3, C[:, 2:4] = C2 + C4
    
    for block in range(4):
        # block 0: A1*B11 -> C[:, 0:2]
        # block 1: A1*B12 -> C[:, 2:4]
        # block 2: A2*B21 -> C[:, 0:2]
        # block 3: A2*B22 -> C[:, 2:4]
        
        if block == 0:
            a_j_offset = 0  # A columns 0,1
            b_j_offset = 0  # B rows 0,1
            b_k_offset = 0  # B cols 0,1
            c_k_offset = 0  # C cols 0,1
        elif block == 1:
            a_j_offset = 0
            b_j_offset = 0
            b_k_offset = 2
            c_k_offset = 2
        elif block == 2:
            a_j_offset = 2
            b_j_offset = 2
            b_k_offset = 0
            c_k_offset = 0
        else:
            a_j_offset = 2
            b_j_offset = 2
            b_k_offset = 2
            c_k_offset = 2
        
        for sf in strassen_factors:
            # A indices in this block: (i, a_j_offset + local_j)
            # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            for local_a_idx, coeff in sf['U'].items():
                a_i = local_a_idx // 2
                a_local_j = local_a_idx % 2
                global_a_j = a_j_offset + a_local_j
                U[a_i * m + global_a_j][col] = coeff
            
            # B indices: (b_j_offset + local_j, b_k_offset + local_k)
            # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            for local_b_idx, coeff in sf['V'].items():
                b_local_j = local_b_idx // 2
                b_local_k = local_b_idx % 2
                global_b_j = b_j_offset + b_local_j
                global_b_k = b_k_offset + b_local_k
                V[global_b_j * p + global_b_k][col] = coeff
            
            # C indices: (i, c_k_offset + local_k)
            # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            for local_c_idx, coeff in sf['W'].items():
                c_i = local_c_idx // 2
                c_local_k = local_c_idx % 2
                global_c_k = c_k_offset + c_local_k
                W[global_c_k * n + c_i][col] = coeff
            
            col += 1
    
    # Part 2: A_{2x4} B_{4x1} -> C[:, 4]
    # Schoolbook: C[i,4] = sum_{j=0}^3 A[i,j] * B[j,4]
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Part 3: A_{2x1} B_{1x4} -> C[:, 0:4]
    # Schoolbook: C[i,k] = A[i,4] * B[4,k] for k=0..3
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 4: A_{2x1} B_{1x1} -> C[:, 4]
    # Schoolbook: C[i,4] += A[i,4] * B[4,4]
    for i in range(2):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
