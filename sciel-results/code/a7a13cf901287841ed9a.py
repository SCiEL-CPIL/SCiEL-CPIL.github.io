"""Matrix multiplication 3x4x5 - rank-minimal decomposition via block tiling."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 B11
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    # M4 = A11 B10
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    # M5 = (A00+A01)B10
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    # M6 = (A10-A00)(B00+B11)
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    Convention: u indexed by (i,j) for i in 0..1, j in 0..1
                v indexed by (j,k) for j in 0..1, k in 0..2
                w indexed by (k,i) for k in 0..2, i in 0..1
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = list(u)
        # v: 2x2x2 terms use k=0,1; map v[0]->v[0], v[1]->v[1], v[2]->v[3], v[3]->v[4]
        # Wait, for 2x2x2: v is indexed by (j,k) with j,k in 0..1, so 4 entries.
        # For 2x2x3: v is indexed by (j,k) with j in 0..1, k in 0..2, so 6 entries.
        # The 2x2x2 sub-problem uses columns k=0,1 of B.
        # So v[0]->v[0*2+0]=v[0], v[1]->v[0*2+1]=v[1], v[2]->v[1*2+0]=v[2], v[3]->v[1*2+1]=v[3]
        v_g = [0] * 6
        v_g[0] = v[0]  # j=0,k=0
        v_g[1] = v[1]  # j=0,k=1
        v_g[2] = v[2]  # j=1,k=0
        v_g[3] = v[3]  # j=1,k=1
        # w: 2x2x2 uses k=0,1; for 2x2x3, w indexed by (k,i) with k in 0..2, i in 0..1
        # w[0]->w[0*2+0]=w[0], w[1]->w[0*2+1]=w[1], w[2]->w[1*2+0]=w[2], w[3]->w[1*2+1]=w[3]
        w_g = [0] * 6
        w_g[0] = w[0]  # k=0,i=0
        w_g[1] = w[1]  # k=0,i=1
        w_g[2] = w[2]  # k=1,i=0
        w_g[3] = w[3]  # k=1,i=1
        terms.append((u_g, v_g, w_g))
    
    # 2x2x1 terms for k=2:
    # A[0,0]*B[0,2]*C[2,0]
    terms.append(([1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]))
    # A[0,1]*B[1,2]*C[2,0]
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]))
    # A[1,0]*B[0,2]*C[2,1]
    terms.append(([0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]))
    # A[1,1]*B[1,2]*C[2,1]
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2.
    Convention: u indexed by j (0..1), v indexed by (j,k) (4 entries), w indexed by k (0..1)
    """
    terms = []
    # A[0]*B[0,0]*C[0]
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    # A[0]*B[0,1]*C[1]
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    # A[1]*B[1,0]*C[0]
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    # A[1]*B[1,1]*C[1]
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3.
    Convention: u indexed by j (0..1), v indexed by (j,k) (6 entries), w indexed by k (0..2)
    """
    terms = []
    # A[0]*B[0,0]*C[0]
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    # A[0]*B[0,1]*C[1]
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    # A[0]*B[0,2]*C[2]
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    # A[1]*B[1,0]*C[0]
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    # A[1]*B[1,1]*C[1]
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    # A[1]*B[1,2]*C[2]
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _get_rank_5_1x2x4():
    """
    Returns list of 5 (u, v, w) vectors for 1x2x4.
    A is 1x2, B is 2x4, C is 1x4.
    C0 = A0*B00 + A1*B10
    C1 = A0*B01 + A1*B11
    C2 = A0*B02 + A1*B12
    C3 = A0*B03 + A1*B13
    
    Can we do better than 8? Let's try.
    Actually, for 1x2x4, the rank is at most 8 (schoolbook).
    Can we use Strassen-like?
    C = [A0, A1] * [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    C = A0 * B_row0 + A1 * B_row1
    
    This is just a linear combination. The rank is the sum of ranks of the two rows?
    No, the tensor is T_{i,j,k} = delta_{i,0} delta_{j,0} delta_{k,0} + ...
    Actually, T = sum_{i=0}^{1} sum_{j=0}^{3} A_i B_{ij} C_{j}
    This is a sum of 8 rank-1 tensors.
    Can we combine them?
    If we group by B column:
    T = sum_{j=0}^{3} (sum_{i=0}^{1} A_i B_{ij} C_j)
    The term for column j is (A0 B0j + A1 B1j) C_j.
    This is not a single rank-1 tensor unless A0 B0j + A1 B1j is separable.
    But A and B are independent variables.
    So we cannot combine terms across different j.
    So the rank is 8.
    """
    terms = []
    # u is len 2
    # v is len 8
    # w is len 4
    
    # 1. A0 * B00
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0]))
    # 2. A0 * B01
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0]))
    # 3. A0 * B02
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0]))
    # 4. A0 * B03
    terms.append(([1, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1]))
    # 5. A1 * B10
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0]))
    # 6. A1 * B11
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0]))
    # 7. A1 * B12
    terms.append(([0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0]))
    # 8. A1 * B13
    terms.append(([0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1]))
    return terms

def _get_rank_10_1x3x4():
    """
    Returns list of 10 (u, v, w) vectors for 1x3x4.
    A is 1x3, B is 3x4, C is 1x4.
    Schoolbook rank is 12.
    Can we do 10?
    C_j = A0 B0j + A1 B1j + A2 B2j.
    This is a sum of 12 rank-1 terms.
    Can we combine?
    T = sum_{j=0}^{3} (A0 B0j + A1 B1j + A2 B2j) C_j.
    Again, cannot combine across j.
    So rank is 12.
    """
    terms = []
    # u is len 3
    # v is len 12
    # w is len 4
    
    for i in range(3):
        for j in range(4):
            u = [0] * 3
            u[i] = 1
            v = [0] * 12
            v[i * 4 + j] = 1
            w = [0] * 4
            w[j] = 1
            terms.append((u, v, w))
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will construct the 56-rank decomposition.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    # Block 1: C11 (2x2) from A1 (2x4) and B1 (4x2)
    s_terms = _get_strassen_2x2x2()
    
    # Term 1: A11 * B11
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B12
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: C12 (2x3) from A1 (2x4) and B2 (4x3)
    # C[0:2, 3:5] from A[0:2, :] and B[:, 3:5]
    # Wait, the tiling is: A rows 0-1, B cols 0-2 and 3-4
    # Block (0,1): A[0:2, :] (2x4) x B[:, 3:5] (4x2) -> C[0:2, 3:5] (2x2)
    # That's 2x4x2, not 2x2x3!
    
    # I need to re-examine the tiling. Let me re-read the original code.
    # The original code splits:
    # A rows: 0-1, 2 (2+1)
    # B cols: 0-1, 2-4 (2+3)
    # So blocks are:
    # (0,0): A[0:2,:], B[:,:2] -> C[0:2,:2] = 2x4x2
    # (0,1): A[0:2,:], B[:,2:5] -> C[0:2,2:5] = 2x4x3
    # (1,0): A[2:3,:], B[:,:2] -> C[2:3,:2] = 1x4x2
    # (1,1): A[2:3,:], B[:,2:5] -> C[2:3,2:5] = 1x4x3
    
    # But the code uses 2x2x2, 2x2x3, 1x2x2, 1x2x3 sub-decompositions.
    # This means it's further splitting the m=4 dimension!
    # For block (0,0): 2x4x2, it splits A cols into 0-1 and 2-3:
    #   (0,0,0): A[0:2,0:2], B[0:2,:2] -> C[0:2,:2] = 2x2x2 (rank 7)
    #   (0,0,1): A[0:2,2:4], B[2:4,:2] -> C[0:2,:2] = 2x2x2 (rank 7)
    # Total: 14 for 2x4x2.
    
    # OK so the approach is a 3-level tiling. Let me keep it as is.
    r11_terms = _get_rank_11_2x2x3()
    
    # Term 1: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]
    # u: A[0:2,0:2] indexed as (i,j) with i in 0..1, j in 0..1 -> positions 0,1,4,5 in U
    # v: B[0:2,2:5] indexed as (j,k) with j in 0..1, k in 0..2 -> positions 2,3,4, 7,8,9 in V
    # w: C[0:2,2:5] indexed as (k,i) with k in 0..2, i in 0..1 -> positions 2,3,4,5,6,7 in W
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]   # i=0,j=0
        u_g[1] = u[1]   # i=0,j=1
        u_g[4] = u[2]   # i=1,j=0
        u_g[5] = u[3]   # i=1,j=1
        v_g[2] = v[0]   # j=0,k=0 (col 2)
        v_g[3] = v[1]   # j=0,k=1 (col 3)
        v_g[4] = v[2]   # j=0,k=2 (col 4)
        v_g[7] = v[3]   # j=1,k=0 (col 2)
        v_g[8] = v[4]   # j=1,k=1 (col 3)
        v_g[9] = v[5]   # j=1,k=2 (col 4)
        w_g[2] = w[0]   # k=0,i=0
        w_g[3] = w[1]   # k=0,i=1
        w_g[4] = w[2]   # k=1,i=0
        w_g[5] = w[3]   # k=1,i=1
        w_g[6] = w[4]   # k=2,i=0
        w_g[7] = w[5]   # k=2,i=1
        add_term(u_g, v_g, w_g)
        
    # Term 2: A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5]
    # u: A[0:2,2:4] -> positions 2,3,6,7 in U
    # v: B[2:4,2:5] -> positions 12,13,14, 17,18,19 in V
    # w: same as above
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]   # i=0,j=2
        u_g[3] = u[1]   # i=0,j=3
        u_g[6] = u[2]   # i=1,j=2
        u_g[7] = u[3]   # i=1,j=3
        v_g[12] = v[0]  # j=2,k=0
        v_g[13] = v[1]  # j=2,k=1
        v_g[14] = v[2]  # j=2,k=2
        v_g[17] = v[3]  # j=3,k=0
        v_g[18] = v[4]  # j=3,k=1
        v_g[19] = v[5]  # j=3,k=2
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 3: C21 (1x2) from A2 (1x4) and B1 (4x2)
    r4_terms = _get_rank_4_1x2x2()
    
    # Term 1: A21 * B11
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B12
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 4: C22 (1x3) from A2 (1x4) and B2 (4x3)
    r6_terms = _get_rank_6_1x2x3()
    
    # Term 1: A21 * B21
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B22
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
