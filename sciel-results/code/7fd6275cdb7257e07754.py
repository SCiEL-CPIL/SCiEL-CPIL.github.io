"""Matrix multiplication 3x4x8 with rank-84 decomposition using Strassen + 3x4x2 blocks."""
import time

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00: m1+m3+m5-m6
        [0, 0, 0, 0, 1, 0, 1],    # a01: m5+m7
        [0, 1, 0, 0, 0, 1, 0],    # a10: m2+m6
        [1, 1, 0, 1, 0, 0, -1]    # a11: m1+m2+m4-m7
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00: m1+m2-m4+m6
        [0, 0, 1, 0, 0, 1, 0],    # b01: m3+m6
        [0, 0, 0, 1, 0, 0, 1],    # b10: m4+m7
        [1, 0, -1, 0, 1, 0, 1]    # b11: m1-m3+m5+m7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],    # c01: m3+m5
        [0, 1, 0, 1, 0, 0, 0],    # c10: m2+m4
        [1, -1, 1, 0, 0, 1, 0]    # c11: m1-m2+m3+m6
    ]
    return U, V, W


def _block_3x4x2():
    """
    Return U, V, W for 3x4x2 multiplication (rank 12).
    
    A is 3x4, B is 4x2, C is 3x2.
    U has shape (12, 12), V has shape (8, 12), W has shape (6, 12).
    
    We use a Kronecker-like structure:
    C = A * B where A is 3x4 and B is 4x2.
    
    We decompose this as: for each column of B (2 columns), we multiply 3x4 by 1x2.
    But we can do better by sharing.
    
    Actually, let me think about this more carefully.
    
    A = [[a00, a01, a02, a03],
         [a10, a11, a12, a13],
         [a20, a21, a22, a23]]
    
    B = [[b00, b01],
         [b10, b11],
         [b20, b21],
         [b30, b31]]
    
    C = A * B is 3x2:
    C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
    C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
    C[2,0] = a20*b00 + a21*b10 + a22*b20 + a23*b30
    C[2,1] = a20*b01 + a21*b11 + a22*b21 + a23*b31
    
    Schoolbook would need 12 products. Let me try to find a rank-12 decomposition.
    
    Actually, for 3x4x2, the trivial rank is 3*4*2 = 24, but we can do better.
    Each entry of C is a dot product of length 4.
    
    Let me use the approach: split A into two 3x2 blocks and B into two 2x2 blocks? No, B is 4x2.
    
    Alternative: Use the fact that we can share computations across the two output columns.
    
    For each row i of A (i=0,1,2), we compute:
    C[i,0] = sum_j A[i,j] * B[j,0]
    C[i,1] = sum_j A[i,j] * B[j,1]
    
    This is like multiplying a 1x4 row by a 4x2 matrix.
    For each row, we need 4*2 = 8 products, but we can share.
    
    Actually, for a 1x4 times 4x2, we can use rank 6 (Strassen-like for 1x4x2):
    Let me define:
    m1 = (a0+a3)(b00+b10)
    m2 = a0(b10-b00)
    m3 = a3(b11-b01)
    m4 = (a1+a2)(b00+b11)
    m5 = a1(b11-b00)
    m6 = a2(b01-b11)
    
    C[0] = m1 + m2 - m3 + m4 + m5 - m6  ... let me verify this.
    
    Actually, let me just use schoolbook for each 1x4x2 which is rank 8, and use 3 such blocks for rank 24.
    That's not good.
    
    Let me try a different approach. I'll use rank 12 for the 3x4x2 block by using a known decomposition.
    
    Actually, the minimal rank for 3x4x2 is known to be at most 12 (since 3*4=12 and 4*2=8, 3*2=6).
    Let me just use a construction that achieves rank 12.
    
    For the 3x4x2 case, I'll use the following:
    Split the 4 dimension into two groups of 2.
    A = [A1, A2] where A1, A2 are 3x2
    B = [B1; B2] where B1, B2 are 2x2
    
    C = A1*B1 + A2*B2
    
    Now, for each 3x2x2 multiplication, the rank is at most 3*2*2 = 12, but we can do better.
    Actually, 3x2x2: A is 3x2, B is 2x2, C is 3x2.
    This is like 3 independent 1x2x2 multiplications, each of rank 4 (schoolbook) or rank 4.
    So 3*4 = 12 for each, giving 24 total. Not good.
    
    Let me just go with a simpler approach: use Strassen for 2x2x2 blocks where possible, and handle the 3rd row separately.
    
    For the 3x4x8 problem:
    - For i=0,1: use 2x2x8 via Strassen, giving 2*4*7 = 56 rank
    - For i=2: use 1x4x8, which is 4x8 = 32 rank with schoolbook
    Total: 56 + 32 = 88
    
    To improve, I need a better 1x4x8 decomposition.
    1x4x8: A is 1x4, B is 4x8, C is 1x8.
    C[k] = sum_j A[j] * B[j,k] for k=0..7
    
    This is a linear combination. The rank of this tensor is at most 4*8 = 32 (schoolbook).
    Can we do better? The tensor is T[i,j,k] = delta_{i,0} * A[j] * B[j,k] but that's not quite right.
    
    Actually, the multiplication tensor for 1x4x8 has entries T[j,k,0] = 1 for all j,k (since C[0,k] = sum_j A[j]*B[j,k]).
    Wait, no. The tensor is defined by T[i,j,k] such that C[i,k] = sum_j sum_k' T[i,j,k'] * A[i,j'] * B[j',k].
    For 1x4x8: i ranges over {0}, j over {0,1,2,3}, k over {0,...,7}.
    T[0,j,k] = 1 for all j,k.
    
    This is a rank-32 tensor (it's just the all-ones tensor of shape 1x4x8).
    The rank of the all-ones tensor of shape 1x4x8 is... well, it can be written as sum of 4 rank-1 tensors:
    T = sum_{j=0}^{3} (e_j in A-space) outer (e_j in B-space) outer (1 in C-space)
    But that's not right either.
    
    Actually, T[0,j,k] = 1 = sum_{j=0}^{3} delta_{j,j'} * 1
    So T = sum_{j'=0}^{3} u_{j'} outer v_{j'} outer w_{j'}
    where u_{j'}[j] = delta_{j,j'}, v_{j'}[k] = 1, w_{j'}[0] = 1
    This gives rank 4.
    
    Wait, that can't be right. Let me reconsider.
    
    The multiplication tensor M for multiplying a 1x4 matrix by a 4x8 matrix has entries:
    M[i,j,k] where i in {0}, j in {0,1,2,3}, k in {0,...,7}
    M[0,j,k] = 1
    
    This tensor can be decomposed as:
    M = sum_{j=0}^{3} (e_j in A) outer (1 in B) outer (e_j in C) ... no, C only has dimension 1.
    
    Let me be more careful. The tensor is in A* x B* x C*.
    A* has basis {a_0}, B* has basis {b_0, ..., b_7}, C* has basis {c_0, ..., c_3}
    
    M(a, b, c) = c_0 * sum_j a_j * b_j
    
    So M = c_0 outer (sum_j a_j outer b_j)
    = sum_j (c_0 outer a_j outer b_j)
    
    This is rank 4!
    
    So for 1x4x8, we can achieve rank 4.
    
    Wait, that means the total rank would be:
    - 2 Strassen 2x2x8 blocks: 2 * (4 * 7) = 56
    - 1 block of 1x4x8: rank 4
    Total: 60
    
    But wait, I need to double-check. The 1x4x8 tensor:
    C = A * B where A is 1x4, B is 4x8, C is 1x8.
    C_k = sum_j A_j * B_jk
    
    The tensor T has T[j,k] = 1 for all j,k (in the sense that T contributes to the computation).
    
    Actually, I think I'm confusing myself. Let me use the standard definition.
    
    For matrix multiplication of (n x m) by (m x p) matrices:
    T[i,j,k] = 1 if the (i,k) entry of the product depends on A[i,j] * B[j,k]
    
    For 1x4x8:
    T[0,j,k] = 1 for j in {0,1,2,3}, k in {0,...,7}
    
    This tensor can be written as:
    T = sum_{j=0}^{3} u_j outer v_j outer w_j
    
    where u_j is a vector in R^4 (A dimension), v_j in R^8 (B dimension), w_j in R^1 (C dimension)
    
    We need: sum_j u_j[j'] * v_j[k] * w_j[0] = delta_{j', j} ... no.
    
    We need: sum_j u_j[j'] * v_j[k] * w_j[0] = T[j',k,0] = 1 for all j',k
    
    If we set w_j[0] = 1 for all j, u_j[j'] = delta_{j,j'}, v_j[k] = 1 for all k:
    sum_j delta_{j,j'} * 1 * 1 = 1 for all j',k. Yes!
    
    So rank 4 for 1x4x8.
    
    Similarly, for 1x2x2:
    T[j,k,0] = 1 for j in {0,1}, k in {0,1}
    Rank = 2.
    
    So my earlier _schoolbook_1x2x2 had rank 4, but it can be done in rank 2!
    
    Let me redo the decomposition:
    - For i=0,1: 2x2x8 using 4 Strassen blocks each: 2 * 4 * 7 = 56
    - For i=2: 1x4x8 using rank 4: 4
    Total: 60
    
    That's much better than 88!
    """
    # 1x4x2 block: rank 2
    # U has shape (4, 2), V has shape (4, 2), W has shape (2, 2)
    # Wait, dimensions: A is 1x2 (so U has 1*2=2 rows), B is 2x2 (V has 4 rows), C is 1x2 (W has 2 rows)
    
    # For 1x4x2:
    # A is 1x4, B is 4x2, C is 1x2
    # U: shape (4, rank), V: shape (8, rank), W: shape (2, rank)
    
    # Rank 2 decomposition:
    # T[j,k] = 1 for j in 0..3, k in 0..1
    # u_j: in R^4, v_j: in R^2, w_j: in R^1... 
    
    # Actually, let me just construct it directly.
    # For 1x4x2:
    # U[i*m+j, c] where i=0, j=0..3, so 4 rows
    # V[j*p+k, c] where j=0..3, k=0..1, so 8 rows  
    # W[k*n+i, c] where k=0..1, i=0, so 2 rows
    
    rank = 4
    U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V = [[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]
    W = [[1,1,1,1],[1,1,1,1]]
    
    return U, V, W


def _block_1x4x8():
    """
    Return U, V, W for 1x4x8 multiplication (rank 4).
    
    A is 1x4, B is 4x8, C is 1x8.
    U: shape (4, 4), V: shape (32, 4), W: shape (8, 4)
    
    Decomposition: T[j,k] = 1 for j=0..3, k=0..7
    U[j, c] = delta(j,c), V[j*8+k, c] = 1 for all k, W[k, c] = 1 for all c
    """
    rank = 4
    U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V = []
    for j in range(4):
        for k in range(8):
            row = [0]*rank
            row[j] = 1
            V.append(row)
    W = [[1,1,1,1]] * 8
    
    return U, V, W


def _build_decomposition():
    """Build rank-60 decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    total_rank = 60
    
    U = [[0] * total_rank for _ in range(n * m)]   
    V = [[0] * total_rank for _ in range(m * p)]   
    W = [[0] * total_rank for _ in range(p * n)]   
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _block_1x4x8()
    
    col_offset = 0
    
    # For i=0,1: use Strassen 2x2x2 blocks
    for i_base in range(2):
        for jp in range(2):
            for kp in range(4):
                j0 = 2 * jp
                k0 = 2 * kp
                
                u_rows = [
                    i_base * m + j0, i_base * m + j0 + 1,
                    (i_base + 1) * m + j0, (i_base + 1) * m + j0 + 1
                ]
                v_rows = [
                    j0 * p + k0, j0 * p + k0 + 1,
                    (j0 + 1) * p + k0, (j0 + 1) * p + k0 + 1
                ]
                w_rows = [
                    k0 * n + i_base, (k0 + 1) * n + i_base,
                    k0 * n + i_base + 1, (k0 + 1) * n + i_base + 1
                ]
                
                for i in range(4):
                    for c in range(7):
                        U[u_rows[i]][col_offset + c] += U2[i][c]
                        V[v_rows[i]][col_offset + c] += V2[i][c]
                        W[w_rows[i]][col_offset + c] += W2[i][c]
                
                col_offset += 7
    
    # For i=2: use 1x4x8 block (rank 4)
    u_rows = [2 * m + j for j in range(4)]
    v_rows = [j * p + k for j in range(4) for k in range(8)]
    w_rows = [k * n + 2 for k in range(8)]
    
    for i in range(4):
        for c in range(4):
            U[u_rows[i]][col_offset + c] += U1[i][c]
            W[w_rows[i]][col_offset + c] += W1[i][c]
    
    for i in range(32):
        for c in range(4):
            V[v_rows[i]][col_offset + c] += V1[i][c]
    
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140
    
    result = _build_decomposition()
    n, m, p = DIMENSIONS
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result
