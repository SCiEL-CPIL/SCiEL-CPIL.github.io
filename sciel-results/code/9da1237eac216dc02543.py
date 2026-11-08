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
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
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

def _build_4x4x2_schoolbook():
    """
    Build 4x4x2 decomposition using schoolbook.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4*4*2 = 32.
    But we can do better by splitting into 2x2x1 blocks.
    Actually, let's use schoolbook for simplicity first: rank 32.
    Then try to optimize.
    """
    n, m, p = 4, 4, 2
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

def _build_2x2x1():
    """
    Build 2x2x1 schoolbook decomposition.
    A is 2x2, B is 2x1, C is 2x1.
    Rank = 4.
    """
    n, m, p = 2, 2, 1
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

def _build_4x4x2_optimized():
    """
    Build 4x4x2 decomposition by splitting into four 2x2x1 blocks.
    Rank = 4 * 4 = 16.
    Each 2x2x1 block has rank 4 (schoolbook).
    """
    n, m, p = 4, 4, 2
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U2, V2, W2 = _build_2x2x1()
    r_sub = len(U2[0])  # rank of 2x2x1 = 4
    
    # Block definitions: (a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start, base_rank)
    # A is 4x4, B is 4x2, C is 4x2
    # U row index: i * m + j = i * 4 + j
    # V row index: j * p + k = j * 2 + k
    # W row index: k * n + i = k * 4 + i
    blocks = [
        # (a_i_s, a_j_s, b_j_s, b_k_s, c_i_s, c_k_s, base_rank)
        (0, 0, 0, 0, 0, 0, 0),   # A[0:2,0:2] * B[0:2,0:1] -> C[0:2,0:1]
        (0, 0, 0, 1, 0, 1, 4),   # A[0:2,0:2] * B[0:2,1:2] -> C[0:2,1:2]
        (2, 0, 0, 0, 2, 0, 8),   # A[2:4,0:2] * B[0:2,0:1] -> C[2:4,0:1]
        (2, 0, 0, 1, 2, 1, 12),  # A[2:4,0:2] * B[0:2,1:2] -> C[2:4,1:2]
    ]
    
    for a_i_s, a_j_s, b_j_s, b_k_s, c_i_s, c_k_s, base_rank in blocks:
        # Copy U2 (shape 4x4) into U at appropriate positions
        for u_row_sub in range(4):
            i_loc = u_row_sub // 2
            j_loc = u_row_sub % 2
            u_row_glob = (a_i_s + i_loc) * 4 + (a_j_s + j_loc)
            for c in range(r_sub):
                U[u_row_glob][base_rank + c] = U2[u_row_sub][c]
        
        # Copy V2 (shape 2x4) into V at appropriate positions
        for v_row_sub in range(2):
            j_loc = v_row_sub // 1
            k_loc = v_row_sub % 1
            v_row_glob = (b_j_s + j_loc) * 2 + (b_k_s + k_loc)
            for c in range(r_sub):
                V[v_row_glob][base_rank + c] = V2[v_row_sub][c]
        
        # Copy W2 (shape 2x4) into W at appropriate positions
        for w_row_sub in range(2):
            k_loc = w_row_sub // 2
            i_loc = w_row_sub % 2
            w_row_glob = (c_k_s + k_loc) * 4 + (c_i_s + i_loc)
            for c in range(r_sub):
                W[w_row_glob][base_rank + c] = W2[w_row_sub][c]
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    Special case: if m=1, use rank 1 (outer product).
    """
    if m == 1:
        # Outer product case: C[i,k] = A[i,0] * B[0,k]
        rank = 1
        U = [[1] for _ in range(n)]
        V = [[1] for _ in range(p)]
        W = [[1] for _ in range(p*n)]
        return U, V, W
    
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct by checking that
    the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    equals 1 if i,j,k form a valid multiplication index, else 0.
    Actually, the multiplication tensor is T[i,j,k] = delta_{j,k} in the sense that
    C[i,k] = sum_j A[i,j] * B[j,k].
    So the tensor should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    Wait, the convention is W[k*n+i, r]. So the tensor is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should equal 1 for all i,j,k? No, that's not right.
    
    The matrix multiplication tensor is defined such that:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor decomposition form:
    (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... no this is getting confusing.
    
    Let me think again. The standard tensor decomposition for matrix multiplication:
    T[i,j,k] = delta_{i,j} * delta_{j,k} ... no.
    
    Actually, the multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if the indices correspond to a valid term in the multiplication, i.e., always 1 for the "diagonal" in some sense.
    
    More precisely, the tensor is:
    M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
    
    For matrix multiplication, the tensor is:
    M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No wait, that's the schoolbook. The tensor multiplication map is:
    (A, B) -> A*B where (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    In terms of a trilinear form:
    <A, B, C> = sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] ... no.
    
    Let me use the convention from the problem: W[k*n+i, r]. So the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k]
    
    where M[i,j,k] is the multiplication tensor. For matrix multiplication, M[i,j,k] = 1 for all i,j,k? That can't be right because then the rank would be at least n*m*p.
    
    Actually, I think the tensor is:
    M[i,j,k] = 1 if j is the "middle" index connecting A and B, i.e., M[i,j,k] = 1 for all i,j,k.
    
    No, I think I'm overcomplicating this. Let me just verify with a simple case.
    
    For 1x1x1: U[0,0]=1, V[0,0]=1, W[0,0]=1. sum_r U[0,0]*V[0,0]*W[0,0] = 1. Correct.
    
    For 2x2x2 Strassen: we can verify numerically.
    
    Let me just do a numerical check with random matrices.
    """
    import numpy as np
    
    rank = len(U[0])
    
    # Create random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for r in range(rank):
                    C_computed[i, k] += A[i, j] * B[j, k] * U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Actually, the decomposition should give:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # This should equal sum_j A[i,j] * B[j,k]
    
    # Let me restructure: for each r, define:
    # A_r[i,j] = U[i*m+j, r] * A[i,j]
    # B_r[j,k] = V[j*p+k, r] * B[j,k]
    # Then C_r[i,k] = sum_j A_r[i,j] * B_r[j,k]
    # And C = sum_r C_r * W[k*n+i, r] ... no, W is indexed by [k*n+i, r]
    
    # I think the correct interpretation is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k])
    #         = sum_r W[k*n+i, r] * (sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k]))
    
    # Let me just check if the sum equals the expected value.
    max_diff = 0
    for i in range(n):
        for k in range(p):
            val = 0
            for j in range(m):
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] * A[i,j] * B[j,k]
            max_diff = max(max_diff, abs(val - C_expected[i,k]))
    
    return max_diff < 1e-10

def _build_2x3x3_decomposition():
    """
    Build a 2x3x3 matrix multiplication decomposition.
    A is 2x3, B is 3x3, C is 2x3.
    Rank = 6 (schoolbook) or try to find better.
    """
    n, m, p = 2, 3, 3
    rank = n * m * p  # Start with schoolbook
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

def _kronecker_combine(decomp1, decomp2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions using Kronecker product structure.
    decomp1: (U1, V1, W1) for n1 x m1 * m1 x p1
    decomp2: (U2, V2, W2) for n2 x m2 * m2 x p2
    Result: decomposition for (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2)
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                # U1 index
                u1_row = i1 * m1 + j1
                for i2 in range(n2):
                    for j2 in range(m2):
                        # U2 index
                        u2_row = i2 * m2 + j2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                r_idx = r1_idx * r2 + r2_idx
                                u_row = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[u_row][r_idx] += U1[u1_row][r1_idx] * U2[u2_row][r2_idx]
    
    for j1 in range(m1):
        for k1 in range(p1):
            # V1 index
            v1_row = j1 * p1 + k1
            for j2 in range(m2):
                for k2 in range(p2):
                    # V2 index
                    v2_row = j2 * p2 + k2
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            r_idx = r1_idx * r2 + r2_idx
                            v_row = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V[v_row][r_idx] += V1[v1_row][r1_idx] * V2[v2_row][r2_idx]
    
    for k1 in range(p1):
        for i1 in range(n1):
            # W1 index
            w1_row = k1 * n1 + i1
            for k2 in range(p2):
                for i2 in range(n2):
                    # W2 index
                    w2_row = k2 * n2 + i2
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            r_idx = r1_idx * r2 + r2_idx
                            w_row = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W[w_row][r_idx] += W1[w1_row][r1_idx] * W2[w2_row][r2_idx]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Try Kronecker product approach:
        # 4x5x6 = (2x2x2) x (2x3x3) in some sense
        # Actually, let's think of it as:
        # A is 4x5, B is 5x6
        # We can split A into blocks and use tensor product structure
        
        # Approach 1: Current additive block (rank 101)
        # 1. Build 4x4x4 Strassen (Rank 49)
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        # 2. Build 4x1x6 Schoolbook (Rank 24)
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        # 3. Build 4x4x2 optimized (Rank 16)
        U3, V3, W3 = _build_4x4x2_optimized()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U_add, V_add, W_add = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        
        # Approach 2: Kronecker product
        # Decompose 4x5x6 as (2x2x2) ⊗ (2x3x3) doesn't quite work directly
        # Let's try: split dimensions as 4=2*2, 5=2*3 (not exact), 6=2*3
        # Better: use 4x5x6 = (2x3x3) combined with (2x2x2) in a block manner
        
        # Actually, let's try a different split:
        # A (4x5) can be viewed as [[A11, A12], [A21, A22]] where A11, A21 are 2x3 and A12, A22 are 2x2
        # B (5x6) can be viewed as [[B11], [B21]] where B11 is 3x6 and B21 is 2x6
        # This doesn't give a clean tensor product
        
        # Let's stick with the additive approach but verify it
        U, V, W = U_add, V_add, W_add
        
        # Verify if time permits
        if time.time() < deadline - 5.0:
            is_valid = _verify_decomposition(U, V, W, n, m, p)
            if not is_valid:
                # Fallback to schoolbook if verification fails
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
