"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
      For 2x2 * 2x2 -> 2x2:
        A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        C = A*B = [[c11, c12], [c21, c22]]
        
    Strassen's 7 products:
      P1 = (a11+a22)*(b11+b22)
      P2 = (a21+a22)*b11
      P3 = a11*(b12-b22)
      P4 = a22*(b21-b11)
      P5 = (a11+a12)*b22
      P6 = (a21-a11)*(b11+b12)
      P7 = (a12-a22)*(b21+b22)
      
      c11 = P1 + P4 - P5 + P7
      c12 = P3 + P5
      c21 = P2 + P4
      c22 = P1 - P2 + P3 + P6
    """
    # U: shape (4, 7) - rows indexed by (i,j) where i in {0,1}, j in {0,1}
    # U[i*2+j, col] = coefficient of a_{i+1, j+1} in product col
    U = [[0]*7 for _ in range(4)]
    # V: shape (4, 7) - rows indexed by (j,k) where j in {0,1}, k in {0,1}
    # V[j*2+k, col] = coefficient of b_{j+1, k+1} in product col
    V = [[0]*7 for _ in range(4)]
    # W: shape (4, 7) - rows indexed by (k,i) where k in {0,1}, i in {0,1}
    # W[k*2+i, col] = coefficient of c_{i+1, k+1} in product col
    
    # P1 = (a11+a22)*(b11+b22)
    # a: a11=U[0], a22=U[3]; b: b11=V[0], b22=V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # c11 += P1, c22 += P1
    # c11 is i=0,k=0 -> W[0*2+0] = W[0]
    # c22 is i=1,k=1 -> W[1*2+1] = W[3]
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    # a: a21=U[2], a22=U[3]; b: b11=V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # c21 += P2, c22 -= P2
    # c21 is i=1,k=0 -> W[0*2+1] = W[1]
    # c22 is i=1,k=1 -> W[1*2+1] = W[3]
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    # a: a11=U[0]; b: b12=V[1], b22=V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # c12 += P3, c22 += P3
    # c12 is i=0,k=1 -> W[1*2+0] = W[2]
    # c22 is i=1,k=1 -> W[1*2+1] = W[3]
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    # a: a22=U[3]; b: b21=V[2], b11=V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # c11 += P4, c21 += P4
    # c11 is i=0,k=0 -> W[0]
    # c21 is i=1,k=0 -> W[1]
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    # a: a11=U[0], a12=U[1]; b: b22=V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # c11 -= P5, c12 += P5
    # c11 is i=0,k=0 -> W[0]
    # c12 is i=0,k=1 -> W[2]
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    # a: a21=U[2], a11=U[0]; b: b11=V[0], b12=V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # c22 += P6
    # c22 is i=1,k=1 -> W[3]
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    # a: a12=U[1], a22=U[3]; b: b21=V[2], b22=V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # c11 += P7
    # c11 is i=0,k=0 -> W[0]
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition by applying 2x2 Strassen at the block level.
    Split 4x4 into 2x2 blocks of 2x2.
    This gives 7 * (rank of 2x2) = 7*7 = 49 terms for 4x4*4x4->4x4.
    
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Block indices for A: (bi, bj) where bi in {0,1}, bj in {0,1}
    # Block (bi, bj) covers rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # Local row in block: li in {0,1}, local col: lj in {0,1}
    # Global row index: bi*2 + li, global col index: bj*2 + lj
    # U row index = (bi*2+li)*4 + (bj*2+lj)
    
    # Similarly for B: block (bj, bk), V row index = (bj*2+lj)*4 + (bk*2+lk)
    # For C: block (bi, bk), W row index = (bk*2+lk)*4 + (bi*2+li)
    
    # Strassen products at block level:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # Block product assignments:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Map: for each Strassen product s (0..6), the block-level coefficients determine
    # which blocks of A, B, C are involved.
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_defs = [
        # (a_blocks, b_blocks, c_blocks)
        # P1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # P2: (A21+A22)*B11 -> C21 += 1, C22 -= 1
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # P3: A11*(B12-B22) -> C12 += 1, C22 += 1
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # P4: A22*(B21-B11) -> C11 += 1, C21 += 1
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # P5: (A11+A12)*B22 -> C11 -= 1, C12 += 1
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # P6: (A21-A11)*(B11+B12) -> C22 += 1
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # P7: (A12-A22)*(B21+B22) -> C11 += 1
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        # For this Strassen product, we have a 2x2*2x2->2x2 subproblem
        # with 7 scalar terms (from U2, V2, W2).
        # The subproblem's "A" is a 2x2 matrix formed by combining blocks.
        
        # For each scalar term t in the 2x2 subproblem (t=0..6):
        # U2[local_a_row, t] gives coefficient for element in "A_sub"
        # V2[local_b_row, t] gives coefficient for element in "B_sub"
        # W2[local_c_row, t] gives coefficient for element in "C_sub"
        
        for t in range(7):
            col = s * 7 + t
            
            # Process A contributions
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        # This coefficient applies to the element at local position (la_i, la_j) in A_sub
                        # A_sub is formed from blocks of A.
                        # The element (la_i, la_j) in A_sub corresponds to:
                        # For each block (bi, bj) in a_blocks with sign sigma:
                        #   The global element is at row (bi*2 + la_i), col (bj*2 + la_j)
                        #   U row index = (bi*2 + la_i)*4 + (bj*2 + la_j)
                        #   Contribution = sigma * coeff_a
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            # Process B contributions
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            # Process C contributions
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    Each decomp is (U, V, W, i_range, j_range, k_range) where ranges are (start, end) exclusive.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        # Map rows
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using Strassen on 2x2 blocks.
    A(4x4) * B(4x2) -> C(4x2)
    Split into 2x2 blocks. 4 Strassen products of rank 7 each. Total rank 28.
    
    U shape: (16, 28), V shape: (8, 28), W shape: (8, 28)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)]  # 4*4
    V = [[0]*rank for _ in range(8)]   # 4*2
    W = [[0]*rank for _ in range(8)]   # 2*4
    
    # Four products:
    # Product 0: A11 * B11 -> C11
    # Product 1: A12 * B21 -> C11
    # Product 2: A21 * B11 -> C21
    # Product 3: A22 * B21 -> C21
    
    products = [
        # (a_row_start, a_col_start, b_row_start, c_row_start, c_sign)
        (0, 0, 0, 0, 1),   # A11 * B11 -> C11
        (0, 2, 2, 0, 1),   # A12 * B21 -> C11
        (2, 0, 0, 2, 1),   # A21 * B11 -> C21
        (2, 2, 2, 2, 1),   # A22 * B21 -> C21
    ]
    
    for s, (a_rs, a_cs, b_rs, c_rs, c_sign) in enumerate(products):
        for t in range(7):
            col = s * 7 + t
            
            # A contributions: 2x2 block at (a_rs, a_cs)
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_rs + la_i
                        global_j = a_cs + la_j
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
            
            # B contributions: 2x2 block at (b_rs, 0)
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_rs + lb_j
                        global_k = lb_k
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
            
            # C contributions: 2x2 block at (c_rs, 0)
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = lc_k
                        global_i = c_rs + lc_i
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c * c_sign
    
    return U, V, W

def _build_2x1x1():
    """
    Build 2x1x1 decomposition. Rank 2.
    A(2x1) * B(1x1) -> C(2x1)
    c0 = a0*b0, c1 = a1*b0
    """
    rank = 2
    U = [[1, 0], [0, 1]]  # 2*1 = 2 rows
    V = [[1, 1]]          # 1*1 = 1 row
    W = [[1, 0], [0, 1]]  # 1*2 = 2 rows
    return U, V, W

def _build_4x1x6_optimized():
    """
    Build 4x1x6 decomposition with rank 12 instead of 24.
    A(4x1) * B(1x6) -> C(4x6)
    c[i,k] = a[i] * b[k]
    
    We can group the 6 k-values into 3 pairs and use a 2x2 Strassen-like trick
    on the output dimension, but actually for rank-1 matrices (A is 4x1, B is 1x6),
    the tensor is just the outer product of A and B.
    
    Actually, the tensor for 4x1x6 has rank 6 (not 24).
    c[i,k] = a[i] * b[k]
    
    We can write this as:
    c[i,k] = sum_r U[i, r] * V[r, k]
    
    If we let U be 4x6, V be 6x6... no, V must be (1*6, rank) = (6, rank).
    W must be (6*4, rank) = (24, rank).
    
    The rank of the outer product tensor a*b is 1 if we consider it as a single tensor,
    but in the matrix multiplication tensor framework, the rank is the number of scalar
    multiplications needed.
    
    For A(4x1) * B(1x6):
    c[i,k] = a[i] * b[k]
    
    We need:
    U shape: (4*1, rank) = (4, rank)
    V shape: (1*6, rank) = (6, rank)
    W shape: (6*4, rank) = (24, rank)
    
    The standard decomposition uses rank 24:
    For each (i, k), one term: U[i, i*6+k] = 1, V[k, i*6+k] = 1, W[k*4+i, i*6+k] = 1
    
    But we can do better! The tensor T[i,k,j] = delta_{j,0} * a[i] * b[k]
    This is a product of three vectors: a[i] * b[k] * delta[j,0]
    So the rank is 1!
    
    Wait, let me reconsider. The matrix multiplication tensor for n x m * m x p -> n x p
    has entries T[i,j,k,l] = delta_{j,l} where i in [n], j in [m], k in [p], l in [m].
    For n=4, m=1, p=6: T[i,j,k,l] = delta_{j,l} where j,l in {0}, so T[i,0,k,0] = 1 for all i,k.
    
    The tensor is T[i,k] = 1 (for the relevant indices), which is the all-ones tensor of size 4x6.
    The rank of the all-ones tensor of size 4x6 is 1 (it's a rank-1 tensor).
    
    So we can decompose it with rank 1:
    U[i, 0] = 1 for all i
    V[0, 0] = 1
    W[k*4+i, 0] = 1 for all k,i
    
    Let's verify:
    C[i,k] = sum_r U[i,r] * V[r,k] * W[k*4+i, r]... 
    
    Actually, the decomposition is:
    T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For n=4, m=1, p=6:
    T[i,0,k,0] = sum_r U[i*1+0, r] * V[0*6+k, r] * W[k*4+i, r]
    = sum_r U[i, r] * V[k, r] * W[k*4+i, r]
    
    We want T[i,0,k,0] = 1 for all i,k.
    
    If we set rank=1:
    U[i, 0] = 1
    V[k, 0] = 1
    W[k*4+i, 0] = 1
    
    Then sum_r ... = U[i,0] * V[k,0] * W[k*4+i,0] = 1*1*1 = 1. Correct!
    
    So the rank is 1, not 24!
    """
    rank = 1
    U = [[1] for _ in range(4)]
    V = [[1] for _ in range(6)]
    W = [[1] for _ in range(24)]
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. 4x4x4 using Strassen (Rank 49)
    # 2. 4x1x6 using optimized decomposition (Rank 1)
    # 3. 4x4x2 using Strassen on 2x2 blocks (Rank 28)
    # Total Rank = 49 + 1 + 28 = 78
    
    try:
        # 1. Build 4x4x4 Strassen
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        # 2. Build 4x1x6 optimized
        U2, V2, W2 = _build_4x1x6_optimized()
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        # 3. Build 4x4x2 using Strassen on 2x2 blocks
        U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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

# EVOLVE-BLOCK-END
