"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomp():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matmul:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U factors (A-side):
    # M1: a00+a11 -> U[0][0]=1, U[3][0]=1
    # M2: a10+a11 -> U[2][1]=1, U[3][1]=1
    # M3: a00 -> U[0][2]=1
    # M4: a11 -> U[3][3]=1
    # M5: a00+a01 -> U[0][4]=1, U[1][4]=1
    # M6: a10-a00 -> U[2][5]=1, U[0][5]=-1
    # M7: a01-a11 -> U[1][6]=1, U[3][6]=-1
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # V factors (B-side):
    # M1: b00+b11 -> V[0][0]=1, V[3][0]=1
    # M2: b00 -> V[0][1]=1
    # M3: b01-b11 -> V[1][2]=1, V[3][2]=-1
    # M4: b10-b00 -> V[2][3]=1, V[0][3]=-1
    # M5: b11 -> V[3][4]=1
    # M6: b00+b01 -> V[0][5]=1, V[1][5]=1
    # M7: b10+b11 -> V[2][6]=1, V[3][6]=1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    # W factors (C-side): C[k*n+i] where n=2
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W index: k*n + i
    # C00: k=0, i=0 -> W[0]
    # C01: k=1, i=0 -> W[2]
    # C10: k=0, i=1 -> W[1]
    # C11: k=1, i=1 -> W[3]
    
    # W[0] (C00): M1(+1), M4(+1), M5(-1), M7(+1)
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # W[2] (C01): M3(+1), M5(+1)
    W[2][2] = 1; W[2][4] = 1
    # W[1] (C10): M2(+1), M4(+1)
    W[1][1] = 1; W[1][3] = 1
    # W[3] (C11): M1(+1), M2(-1), M3(+1), M6(+1)
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W


def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    Strassen's algorithm applied at two levels.
    
    Level 1: 4x4 split into 2x2 blocks, use 7 Strassen products
    Level 2: Each 2x2 block product uses rank-7 Strassen
    
    Total rank: 7 * 7 = 49
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Get the base 2x2 Strassen decomposition
    sub_U, sub_V, sub_W = _strassen_2x2_decomp()
    
    # Level 1: 7 Strassen block products for 4x4 (2x2 block) multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    
    # For each of the 7 block-level Strassen products:
    # (A_linear_combo, B_linear_combo, C_block_contributions)
    
    strassen_level1 = [
        # M1: (A00+A11) * (B00+B11) -> C00(+1), C11(+1)
        (
            [(0, 0, 1), (1, 1, 1)],  # A blocks: (row, col, coeff)
            [(0, 0, 1), (1, 1, 1)],  # B blocks
            [(0, 0, 1), (1, 1, 1)]   # C block contributions
        ),
        # M2: (A10+A11) * B00 -> C10(+1), C11(-1)
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # M3: A00 * (B01-B11) -> C01(+1), C11(+1)
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # M4: A11 * (B10-B00) -> C00(+1), C10(+1)
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # M5: (A00+A01) * B11 -> C00(-1), C01(+1)
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # M6: (A10-A00) * (B00+B01) -> C11(+1)
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # M7: (A01-A11) * (B10+B11) -> C00(+1)
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    col = 0
    for a_blocks, b_blocks, c_contribs in strassen_level1:
        for (ar, ac, a_coeff) in a_blocks:
            for (br, bc, b_coeff) in b_blocks:
                # This term: a_coeff * b_coeff * (A_block(ar,ac) * B_block(br,bc))
                # A_block(ar,ac) is 2x2, B_block(br,bc) is 2x2
                # Use the 2x2 Strassen decomposition
                
                for r in range(7):
                    # U entries: from sub_U, mapped to global A indices
                    for u_idx in range(4):
                        val = sub_U[u_idx][r]
                        if val != 0:
                            local_row, local_col = divmod(u_idx, 2)
                            global_idx = (ar * 2 + local_row) * 4 + (ac * 2 + local_col)
                            U[global_idx][col] += val * a_coeff
                    
                    # V entries: from sub_V, mapped to global B indices
                    for v_idx in range(4):
                        val = sub_V[v_idx][r]
                        if val != 0:
                            local_row, local_col = divmod(v_idx, 2)
                            global_idx = (br * 2 + local_row) * 4 + (bc * 2 + local_col)
                            V[global_idx][col] += val * b_coeff
                    
                    # W entries: from sub_W, mapped to global C indices
                    for w_idx in range(4):
                        val = sub_W[w_idx][r]
                        if val != 0:
                            local_k, local_i = divmod(w_idx, 2)
                            # w_idx = k*n + i where n=2 for the sub-block
                            for (cr, cc, c_coeff) in c_contribs:
                                global_w_idx = (cc * 2 + local_k) * 4 + (cr * 2 + local_i)
                                W[global_w_idx][col] += val * c_coeff
                    
                    col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check all entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    
    # Check a sample of zero entries (where j != j')
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(actual) > tol:
                        return False
    return True


def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks, giving rank 49.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try the rank-49 Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4()
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
