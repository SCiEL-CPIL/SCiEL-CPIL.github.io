"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for 2x2x2 matrix multiplication."""
    # Strassen's 7 products
    # U: (i*2+k) x 7
    # V: (k*2+j) x 7
    # W: (j*2+i) x 7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 -> U[0], A22 -> U[3], B11 -> V[0], B22 -> V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1  # W[j*2+i]: j=0,i=0 -> 0; j=1,i=1 -> 3. C11 += M1, C22 -= M1
    
    # M2 = (A21+A22)B11
    # A21 -> U[2], A22 -> U[3], B11 -> V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # j=0,i=1 -> 2. C21 += M2
    
    # M3 = A11(B21-B22)
    # A11 -> U[0], B21 -> V[2], B22 -> V[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # j=1,i=0 -> 1. C12 += M3
    
    # M4 = (A11-A22)(B12+B22)
    # A11 -> U[0], A22 -> U[3], B12 -> V[1], B22 -> V[3]
    U[0][3] = 1; U[3][3] = -1
    V[1][3] = 1; V[3][3] = 1
    W[1][3] = 1; W[2][3] = -1  # j=1,i=0 -> 1; j=0,i=1 -> 2. C12 += M4, C21 -= M4
    
    # M5 = (A11+A12)B22
    # A11 -> U[0], A12 -> U[1], B22 -> V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[3][4] = 1  # j=1,i=1 -> 3. C22 += M5
    
    # M6 = (A21-A11)(B11+B12)
    # A21 -> U[2], A11 -> U[0], B11 -> V[0], B12 -> V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = -1  # j=0,i=0 -> 0; j=1,i=0 -> 1. C11 += M6, C12 -= M6
    
    # M7 = (A12-A22)(B21+B22)
    # A12 -> U[1], A22 -> U[3], B21 -> V[2], B22 -> V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[3][6] = -1  # j=0,i=1 -> 2; j=1,i=1 -> 3. C21 += M7, C22 -= M7
    
    return U, V, W

def _combine_decompositions(decomp_list):
    """Combine multiple decompositions into one."""
    total_rank = sum(len(d['U'][0]) for d in decomp_list)
    n = len(decomp_list[0]['U']) // 4  # U has n*m rows
    m = 4  # from 2x2x2
    p = 2
    
    # We need to determine dimensions from the decompositions
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For 2x2x2: n=2, m=2, p=2
    # U: 4x7, V: 4x7, W: 4x7
    
    U = [[0] * total_rank for _ in range(decomp_list[0]['U'][0] and len(decomp_list[0]['U']) or 4)]
    V = [[0] * total_rank for _ in range(decomp_list[0]['V'][0] and len(decomp_list[0]['V']) or 4)]
    W = [[0] * total_rank for _ in range(decomp_list[0]['W'][0] and len(decomp_list[0]['W']) or 4)]
    
    offset = 0
    for d in decomp_list:
        rank_d = len(d['U'][0])
        for i in range(len(d['U'])):
            for c in range(rank_d):
                U[i][offset + c] = d['U'][i][c]
        for i in range(len(d['V'])):
            for c in range(rank_d):
                V[i][offset + c] = d['V'][i][c]
        for i in range(len(d['W'])):
            for c in range(rank_d):
                W[i][offset + c] = d['W'][i][c]
        offset += rank_d
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """Return a rank-49 decomposition for 4x4x4 matrix multiplication.
    
    Uses Strassen recursively: 4x4x4 = 7 * (2x2x2) where each 2x2x2 has rank 7.
    Total rank = 7 * 7 = 49.
    """
    # The 4x4x4 tensor can be decomposed as 7 blocks, each being a 2x2x2 tensor.
    # We need to map the 4x4 indices to 2x2 blocks.
    # A is 4x4, B is 4x4, C is 4x4.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Strassen on the 2x2 block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = (A11-A22)(B12+B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M6 - M4 + M7
    # C12 = M3 + M4
    # C21 = M2 - M6 + M7
    # C22 = M1 - M3 + M5 - M7
    
    # Each M is a 2x2x2 tensor, which has rank 7.
    # So we have 7 * 7 = 49 terms total.
    
    # We need to construct U, V, W of shape (16, 49), (16, 49), (16, 49).
    # U index: i*4+k where i,k in 0..3
    # V index: k*4+j where k,j in 0..3
    # W index: j*4+i where j,i in 0..3
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Get the base 2x2x2 Strassen decomposition
    base_U, base_V, base_W = _strassen_2x2x2()
    
    # Block mapping:
    # Block (bi, bk) where bi is row block (0 or 1), bk is col block (0 or 1)
    # A block (bi, bk) has indices i = bi*2 + i2, k = bk*2 + k2
    # B block (bk, bj) has indices k = bk*2 + k2, j = bj*2 + j2
    # C block (bi, bj) has indices i = bi*2 + i2, j = bj*2 + j2
    
    # Strassen block-level terms:
    # M1: A11+A22 times B11+B22 -> contributes to C11 (+1) and C22 (-1)
    # M2: A21+A22 times B11 -> contributes to C21 (+1)
    # M3: A11 times B21-B22 -> contributes to C12 (+1) and C22 (-1)
    # M4: A11-A22 times B12+B22 -> contributes to C12 (+1) and C21 (-1)
    # M5: A11+A12 times B22 -> contributes to C22 (+1)
    # M6: A21-A11 times B11+B12 -> contributes to C11 (+1) and C12 (-1)
    # M7: A12-A22 times B21+B22 -> contributes to C21 (+1) and C22 (-1)
    
    # For each Strassen term s (0..6) and each inner Strassen term t (0..6):
    # We have a contribution to U, V, W.
    
    # Let's define the block-level Strassen coefficients:
    # For term s, we have coefficients for A-blocks and B-blocks.
    # A-blocks: (11, 12, 21, 22) -> indices (0,0), (0,1), (1,0), (1,1) in bi,bk
    # B-blocks: (11, 12, 21, 22) -> indices (0,0), (0,1), (1,0), (1,1) in bk,bj
    
    # A-block coefficients for each Strassen term s:
    # M1: A11(+1), A22(+1)
    # M2: A21(+1), A22(+1)
    # M3: A11(+1)
    # M4: A11(+1), A22(-1)
    # M5: A11(+1), A12(+1)
    # M6: A21(+1), A11(-1)
    # M7: A12(+1), A22(-1)
    
    # B-block coefficients for each Strassen term s:
    # M1: B11(+1), B22(+1)
    # M2: B11(+1)
    # M3: B21(+1), B22(-1)
    # M4: B12(+1), B22(+1)
    # M5: B22(+1)
    # M6: B11(+1), B12(+1)
    # M7: B21(+1), B22(+1)
    
    # C-block contributions for each Strassen term s:
    # M1: C11(+1), C22(-1)
    # M2: C21(+1)
    # M3: C12(+1), C22(-1)
    # M4: C12(+1), C21(-1)
    # M5: C22(+1)
    # M6: C11(+1), C12(-1)
    # M7: C21(+1), C22(-1)
    
    # A-block mapping: (bi, bk) -> bi in {0,1}, bk in {0,1}
    # A11 -> (0,0), A12 -> (0,1), A21 -> (1,0), A22 -> (1,1)
    
    # B-block mapping: (bk, bj) -> bk in {0,1}, bj in {0,1}
    # B11 -> (0,0), B12 -> (0,1), B21 -> (1,0), B22 -> (1,1)
    
    # C-block mapping: (bi, bj) -> bi in {0,1}, bj in {0,1}
    # C11 -> (0,0), C12 -> (0,1), C21 -> (1,0), C22 -> (1,1)
    
    # Define the coefficients
    A_coeff = [
        {0: 1, 3: 1},  # M1: A11(0,0)->idx 0, A22(1,1)->idx 3
        {2: 1, 3: 1},  # M2: A21(1,0)->idx 2, A22(1,1)->idx 3
        {0: 1},        # M3: A11(0,0)->idx 0
        {0: 1, 3: -1}, # M4: A11(0,0)->idx 0, A22(1,1)->idx 3
        {0: 1, 1: 1},  # M5: A11(0,0)->idx 0, A12(0,1)->idx 1
        {2: 1, 0: -1}, # M6: A21(1,0)->idx 2, A11(0,0)->idx 0
        {1: 1, 3: -1}, # M7: A12(0,1)->idx 1, A22(1,1)->idx 3
    ]
    
    B_coeff = [
        {0: 1, 3: 1},  # M1: B11(0,0)->idx 0, B22(1,1)->idx 3
        {0: 1},        # M2: B11(0,0)->idx 0
        {2: 1, 3: -1}, # M3: B21(1,0)->idx 2, B22(1,1)->idx 3
        {1: 1, 3: 1},  # M4: B12(0,1)->idx 1, B22(1,1)->idx 3
        {3: 1},        # M5: B22(1,1)->idx 3
        {0: 1, 1: 1},  # M6: B11(0,0)->idx 0, B12(0,1)->idx 1
        {2: 1, 3: 1},  # M7: B21(1,0)->idx 2, B22(1,1)->idx 3
    ]
    
    C_coeff = [
        {0: 1, 3: -1},  # M1: C11(0,0)->idx 0, C22(1,1)->idx 3
        {2: 1},         # M2: C21(1,0)->idx 2
        {1: 1, 3: -1},  # M3: C12(0,1)->idx 1, C22(1,1)->idx 3
        {1: 1, 2: -1},  # M4: C12(0,1)->idx 1, C21(1,0)->idx 2
        {3: 1},         # M5: C22(1,1)->idx 3
        {0: 1, 1: -1},  # M6: C11(0,0)->idx 0, C12(0,1)->idx 1
        {2: 1, 3: -1},  # M7: C21(1,0)->idx 2, C22(1,1)->idx 3
    ]
    
    # For each s in 0..6 and t in 0..6:
    # The inner Strassen term t has base_U[i2*2+k2][t], base_V[k2*2+j2][t], base_W[j2*2+i2][t]
    # where i2, k2, j2 are in 0..1.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U contributions:
            for a_block, a_val in A_coeff[s].items():
                bi = a_block // 2
                bk = a_block % 2
                for i2 in range(2):
                    for k2 in range(2):
                        u_idx_2x2 = i2 * 2 + k2
                        val = base_U[u_idx_2x2][t]
                        if val != 0:
                            i = bi * 2 + i2
                            k = bk * 2 + k2
                            u_idx = i * 4 + k
                            U[u_idx][col] += a_val * val
            
            # V contributions:
            for b_block, b_val in B_coeff[s].items():
                bk = b_block // 2
                bj = b_block % 2
                for k2 in range(2):
                    for j2 in range(2):
                        v_idx_2x2 = k2 * 2 + j2
                        val = base_V[v_idx_2x2][t]
                        if val != 0:
                            k = bk * 2 + k2
                            j = bj * 2 + j2
                            v_idx = k * 4 + j
                            V[v_idx][col] += b_val * val
            
            # W contributions:
            for c_block, c_val in C_coeff[s].items():
                bi = c_block // 2
                bj = c_block % 2
                for j2 in range(2):
                    for i2 in range(2):
                        w_idx_2x2 = j2 * 2 + i2
                        val = base_W[w_idx_2x2][t]
                        if val != 0:
                            j = bj * 2 + j2
                            i = bi * 2 + i2
                            w_idx = j * 4 + i
                            W[w_idx][col] += c_val * val
    
    return {"U": U, "V": V, "W": W}

def _try_rank114():
    """Try to construct a rank-114 decomposition.
    
    Uses Strassen for the first 8 columns (4x4x8) and schoolbook for the last column (4x4x1).
    Rank = 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    
    # Strassen decomposition for 4x4x4 (first 4 columns)
    s1 = _strassen_4x4x4()
    # Strassen decomposition for 4x4x4 (next 4 columns)
    s2 = _strassen_4x4x4()
    # Schoolbook for 4x4x1 (last 1 column)
    school = _schoolbook(4, 4, 1)
    
    # Combine
    total_rank = 49 + 49 + 16
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    offset = 0
    for s in range(2):
        rank_s = 49
        # U: same for both (A is 4x4)
        for i in range(16):
            for c in range(rank_s):
                U[i][offset + c] = s1['U'][i][c] if s == 0 else s2['U'][i][c]
        
        # V: for s=0, columns 0-3; for s=1, columns 4-7
        for k in range(4):
            for j in range(4):
                v_idx_4x4 = k * 4 + j
                v_idx_4x8 = k * 8 + (j + s * 4)
                for c in range(rank_s):
                    V[v_idx_4x8][offset + c] = s1['V'][v_idx_4x4][c] if s == 0 else s2['V'][v_idx_4x4][c]
        
        # W: for s=0, columns 0-3; for s=1, columns 4-7
        for j in range(4):
            for i in range(4):
                w_idx_4x4 = j * 4 + i
                w_idx_4x8 = j * 8 + (i + s * 4)
                for c in range(rank_s):
                    W[w_idx_4x8][offset + c] = s1['W'][w_idx_4x4][c] if s == 0 else s2['W'][w_idx_4x4][c]
        
        offset += rank_s
    
    # Schoolbook for last column (j=8)
    rank_school = 16
    for i in range(4):
        for k in range(4):
            u_idx = i * 4 + k
            for c in range(rank_school):
                U[u_idx][offset + c] = school['U'][u_idx][c]
            
            v_idx = k * 9 + 8
            for c in range(rank_school):
                V[v_idx][offset + c] = school['V'][k * 1 + 0][c]
            
            w_idx = 8 * 4 + i
            for c in range(rank_school):
                W[w_idx][offset + c] = school['W'][0 * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try rank-114 decomposition
    try:
        result = _try_rank114()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    assert _verify(best['U'], best['V'], best['W'], n, m, p), "Schoolbook decomposition failed verification"
    return best
# EVOLVE-BLOCK-END
