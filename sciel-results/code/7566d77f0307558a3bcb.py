"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 coefficients (verified correct):
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
#
# Each M_r is a product of a linear form in A and a linear form in B.
# U entries: coefficient of A_{ij} in the A-linear form of M_r
# V entries: coefficient of B_{jk} in the B-linear form of M_r
# W entries: coefficient of C_{ik} in the output (each W_r is a single C entry with coeff 1)

STRASSEN_U = [
    [1, 0, 0, 0],   # M1: A00
    [0, 1, 0, 0],   # M1: A11
    [0, 0, 1, 0],   # M2: A10
    [0, 0, 0, 1],   # M2: A11
    [1, 0, 0, 0],   # M3: A00
    [0, 0, 0, 1],   # M4: A11
    [1, 0, 0, 0],   # M5: A00
    [0, 1, 0, 0],   # M5: A01
    [0, 0, 1, 0],   # M6: A10
    [1, 0, 0, 0],   # M6: A00 (negative, handled below)
    [0, 1, 0, 0],   # M7: A01
    [0, 0, 0, 1],   # M7: A11 (negative, handled below)
]

# Actually let me build this more carefully.
# 7 Strassen products, each M_r = (sum_j u_{r,j} A_j) * (sum_k v_{r,k} B_k)
# where A_j are the 4 entries A00,A01,A10,A11 (row-major)
# and B_k are the 4 entries B00,B01,B10,B11 (row-major)

# M1 = (A00 + A11)(B00 + B11)
# M2 = (A10 + A11) B00
# M3 = A00 (B01 - B11)
# M4 = A11 (B10 - B00)
# M5 = (A00 + A01) B11
# M6 = (A10 - A00)(B00 + B01)
# M7 = (A01 - A11)(B10 + B11)

# U: 4 rows (A00,A01,A10,A11) x 7 cols (M1..M7)
STRASSEN_U = [
    [1, 0, 1, 0, 1, -1, 0],   # A00
    [0, 0, 0, 0, 1,  0, 1],   # A01
    [0, 1, 0, 0, 0,  1, 0],   # A10
    [1, 1, 0, 1, 0,  0,-1],   # A11
]

# V: 4 rows (B00,B01,B10,B11) x 7 cols (M1..M7)
STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],   # B00
    [0, 0, 1,  0, 0, 1, 0],   # B01
    [0, 0, 0,  1, 0, 0, 1],   # B10
    [1, 0,-1,  0, 1, 0, 1],   # B11
]

# W: 4 rows (C00,C01,C10,C11) x 7 cols (M1..M7)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],   # C00
    [0, 0, 1, 0,  1, 0, 0],   # C01
    [0, 1, 0, 1,  0, 0, 0],   # C10
    [1,-1, 1, 0,  0, 1, 0],   # C11
]


def _verify_strassen(n_sub=2, m_sub=2, p_sub=2):
    """Verify Strassen decomposition for 2x2x2 multiplication.
    Check: delta(i,i')delta(j,j')delta(k,k') = sum_r U[i*m+j,r] V[j'*p+k',r] W[k'*n+i',r]
    for all i,j,k,i',j',k' in [0,2).
    """
    U = STRASSEN_U  # shape (2*2, 7)
    V = STRASSEN_V  # shape (2*2, 7)
    W = STRASSEN_W  # shape (2*2, 7)
    rank = 7
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    for kp in range(p_sub):
                        for ip in range(n_sub):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i * m_sub + j][r] * V[jp * p_sub + kp][r] * W[kp * n_sub + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 28

    # Try Strassen-based decomposition: rank 7 (for 2x2x2 part) + 5 (for remaining columns) = 12
    # Actually: for the 2x2x2 subproblem (first 2 columns of B), Strassen gives rank 7.
    # For the remaining 5 columns, schoolbook gives 2*2*5 = 20... that's too much.
    # 
    # Better approach: The full tensor T[i,j,k, i',j',k'] = delta(i,i')delta(j,j')delta(k,k')
    # We can use Strassen for the (i,j) x (i',j') part when k=k' is fixed, but that doesn't directly work.
    #
    # Alternative: Use Strassen structure for the first 2 columns (k=0,1), schoolbook for rest.
    # For k=0,1: the sub-tensor is 2x2x2, decomposable at rank 7.
    # For k=2..6: schoolbook, each (i,j,k) triple needs 1 term, so 2*2*5 = 20 terms.
    # Total: 7 + 20 = 27. That's only 1 better than 28.
    #
    # Better: Apply Strassen across the full k dimension.
    # Think of it as: C = A * B where A is 2x2, B is 2x7.
    # Partition B = [B_left | B_right] where B_left is 2x2, B_right is 2x5.
    # C = A*B_left + A*B_right
    # A*B_left: 2x2x2 multiplication, rank 7 via Strassen.
    # A*B_right: 2x2x5 multiplication, rank 20 via schoolbook.
    # Total: 27.
    #
    # Can we do better? Let's think of B as 2x7 = 2x(2+5).
    # Or partition differently: use Strassen for the 2x2 A part with a 2x2 sub-block of B,
    # and handle the rest.
    #
    # Actually, a cleaner approach: view the 2x2x7 multiplication as having structure.
    # The tensor is T[i,j,k, i',j',k'] = delta(i,i')delta(j,j')delta(k,k').
    # We can factor this as: for each k, the 2x2x1 slice is rank 4 (schoolbook) or rank 7 (Strassen).
    # But we need to share structure across k values.
    #
    # Key insight: The multiplication C_ik = sum_j A_ij B_jk can be seen as:
    # For fixed k, C_.k = A * B_.k (vector-matrix product), rank 4.
    # For all k, we need rank 28 total (schoolbook).
    #
    # Strassen works because it exploits the 2x2x2 structure. For 2x2x7, we can:
    # 1. Use Strassen (rank 7) for the first 2 columns: handles C_.0 and C_.1
    # 2. Use schoolbook (rank 2*2*5=20) for columns 3-7: handles C_.2 through C_.6
    # Total: 27
    #
    # Or: partition B into 2x2 + 2x2 + 2x3
    # Strassen + Strassen + schoolbook(2*2*3=12) = 7+7+12 = 26
    #
    # Or: 2x2 + 2x2 + 2x2 + 2x1
    # 7 + 7 + 7 + 4 = 25
    #
    # Let me verify: 7 = 2+2+2+1, so we use Strassen 3 times and schoolbook 1 time.
    # Total rank: 7*3 + 4 = 25.
    #
    # Let me build this.

    # First, verify Strassen works
    strassen_valid = _verify_strassen()
    
    if not strassen_valid:
        # Fall back to schoolbook
        rank = schoolbook_rank
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

    # Build decomposition: 3 Strassen blocks (for k=0,1; k=2,3; k=4,5) + 1 schoolbook (for k=6)
    # Each Strassen block: rank 7, handles 2 output columns
    # Schoolbook block: rank 4, handles 1 output column
    # Total rank: 7*3 + 4 = 25

    num_strassen = p // 2  # 3
    remaining = p % 2      # 1
    rank = num_strassen * 7 + remaining * n * m  # 21 + 4 = 25

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block in range(num_strassen):
        k_base = block * 2  # starting k index for this block
        # Strassen block handles columns k_base and k_base+1
        # U: 4x7, V: 4x7, W: 4x7
        for r in range(7):
            # U part: A indices (i,j) in 2x2
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = STRASSEN_U[i * m + j][r]
            # V part: B indices (j,k) where k is k_base or k_base+1
            for j in range(m):
                for kk in range(2):
                    k_local = kk  # 0 or 1 within the block
                    V[j * p + (k_base + kk)][col] = STRASSEN_V[j * 2 + kk][r]
            # W part: C indices (i,k) where k is k_base or k_base+1
            for i in range(n):
                for kk in range(2):
                    W[(k_base + kk) * n + i][col] = STRASSEN_W[i * 2 + kk][r]
            col += 1

    # Schoolbook for remaining column(s)
    for k in range(num_strassen * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify the full decomposition
    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                rhs = 0.0
                                for r in range(rank):
                                    rhs += U[i * m + j][r] * V[jp * p + kp][r] * W[kp * n + ip][r]
                                if abs(lhs - rhs) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    rank = schoolbook_rank
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

# EVOLVE-BLOCK-END
