import numpy as np
import random
import time

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()

def _time_left(deadline):
    return deadline - time.time()

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                
                if j1 != j2 or k1 != k2 or i1 != i2:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err = abs(val)
                    max_err = max(max_err, err)
    
    return max_err <= tol

def _strassen_decompose_2x2x2():
    """Return U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrix mult:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U rows indexed by (i,j) -> i*2+j
    # V rows indexed by (j,k) -> j*2+k  
    # W rows indexed by (k,i) -> k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (A11+A22)(B11+B22)
    # A11 is at (i=0,j=0) -> U[0], A22 at (i=1,j=1) -> U[3]
    # B11 at (j=0,k=0) -> V[0], B22 at (j=1,k=1) -> V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 = M1 + ..., C22 = M1 - ...
    # C11 is at (i=0,k=0) -> W[0*2+0] = W[0]
    # C22 is at (i=1,k=1) -> W[1*2+1] = W[3]
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: M2 = (A21+A22)B11
    # A21 at (i=1,j=0) -> U[2], A22 at (i=1,j=1) -> U[3]
    # B11 at (j=0,k=0) -> V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 = M2 + ..., C22 = -M2
    # C21 at (i=1,k=0) -> W[0*2+1] = W[1]
    # C22 at (i=1,k=1) -> W[3]
    W[1, 1] = 1; W[3, 1] = -1
    
    # Product 3: M3 = A11(B21-B22)
    # A11 at (i=0,j=0) -> U[0]
    # B21 at (j=1,k=0) -> V[2], B22 at (j=1,k=1) -> V[3]
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C12 = M3 + ..., C22 = M3
    # C12 at (i=0,k=1) -> W[1*2+0] = W[2]
    # C22 at (i=1,k=1) -> W[3]
    W[2, 2] = 1; W[3, 2] = 1
    
    # Product 4: M4 = A22(B12-B11)
    # A22 at (i=1,j=1) -> U[3]
    # B12 at (j=0,k=1) -> V[1], B11 at (j=0,k=0) -> V[0]
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C11 = M4 + ..., C21 = M4
    # C11 at (i=0,k=0) -> W[0]
    # C21 at (i=1,k=0) -> W[1]
    W[0, 3] = 1; W[1, 3] = 1
    
    # Product 5: M5 = (A11+A12)B22
    # A11 at (i=0,j=0) -> U[0], A12 at (i=0,j=1) -> U[1]
    # B22 at (j=1,k=1) -> V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 = -M5 + ..., C12 = M5
    # C11 at (i=0,k=0) -> W[0]
    # C12 at (i=0,k=1) -> W[2]
    W[0, 4] = -1; W[2, 4] = 1
    
    # Product 6: M6 = (A21-A11)(B11+B12)
    # A21 at (i=1,j=0) -> U[2], A11 at (i=0,j=0) -> U[0]
    # B11 at (j=0,k=0) -> V[0], B12 at (j=0,k=1) -> V[1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 = M6
    # C22 at (i=1,k=1) -> W[3]
    W[3, 5] = 1
    
    # Product 7: M7 = (A12-A22)(B21+B22)
    # A12 at (i=0,j=1) -> U[1], A22 at (i=1,j=1) -> U[3]
    # B21 at (j=1,k=0) -> V[2], B22 at (j=1,k=1) -> V[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 = M7
    # C11 at (i=0,k=0) -> W[0]
    W[0, 6] = 1
    
    return U, V, W

def _construct_structured_decomposition():
    """Construct a structured decomposition for 3x4x6 using block decomposition.
    
    Split: A is 3x4, B is 4x6, C is 3x6.
    Use Strassen for 2x2 blocks where possible.
    
    Strategy: 
    - Split A into rows: A = [A_top(2x4); A_bottom(1x4)]
    - Split B into cols: B = [B_left(4x3) B_right(4x3)]
    - C = [A_top * B_left  A_top * B_right; A_bottom * B_left  A_bottom * B_right]
    
    For A_top * B_left (2x4 * 4x3):
      Split further: A_top = [A_tl(2x2) A_tr(2x2)], B_left = [B_tl(2x3); B_bl(2x3)]
      This is 2x4 * 4x3.
      
    Let me use a different approach: pad to powers of 2 and use Strassen recursively.
    3x4x6: pad to 4x4x8 or 4x8x8? That's too much padding.
    
    Alternative: Direct block decomposition with Strassen on 2x2 sub-blocks.
    
    Let's try: 
    A (3x4) = [A1(2x4); A2(1x4)]
    B (4x6) = [B1(4x3); B2(4x3)]
    
    C (3x6) = [A1*B1  A1*B2; A2*B1  A2*B2]
    
    A1*B1: 2x4 * 4x3
    A1*B2: 2x4 * 4x3
    A2*B1: 1x4 * 4x3
    A2*B2: 1x4 * 4x3
    
    For 2x4*4x3:
    A1 = [A11(2x2) A12(2x2)]
    B1 = [B11(2x3); B12(2x3)]
    A1*B1 = A11*B11 + A12*B12
    
    A11*B11: 2x2 * 2x3 -> can use Strassen-like
    A12*B12: 2x2 * 2x3
    
    For 2x2*2x3:
    A = [a b; c d], B = [e f g; h i j]
    C = [ae+ch, af+ci, ag+cj; be+dh, bf+di, bg+dj]
    
    This is 2x3 = 6 outputs. Schoolbook rank = 2*2*3 = 12.
    Can we do better? 
    
    Actually, let me just carefully construct the full decomposition.
    """
    n, m, p = 3, 4, 6
    
    # I'll build the decomposition by explicitly computing the tensor structure
    # and using Strassen where 2x2x2 sub-blocks exist.
    
    # Let me use a cleaner approach: construct the full 72-rank schoolbook,
    # then try to merge/combine terms using Strassen identities.
    
    # Actually, let me directly construct a rank-68 decomposition using the following:
    # 
    # The 3x4x6 multiplication can be decomposed as:
    # 
    # A (3x4), B (4x6)
    # 
    # Split columns of B: B = [B_L (4x2) | B_M (4x2) | B_R (4x2)]
    # Split rows of A: A = [A_T (2x4); A_B (1x4)]
    # 
    # C = [A_T * B_L  A_T * B_M  A_T * B_R;
    #      A_B * B_L  A_B * B_M  A_B * B_R]
    #
    # A_T * B_*: 2x4 * 4x2
    # A_B * B_*: 1x4 * 4x2
    #
    # For 2x4*4x2:
    #   A_T = [A_TL (2x2) A_TR (2x2)]
    #   B_* = [B_T (2x2); B_B (2x2)]
    #   A_T * B_* = A_TL * B_T + A_TR * B_B
    #   Each 2x2*2x2 uses Strassen (rank 7)
    #   So 2x4*4x2 has rank 7+7 = 14
    #
    # For 1x4*4x2:
    #   A_B = [a1 a2 a3 a4]
    #   B_* = [b11 b12; b21 b22; b31 b32; b41 b42]
    #   C = [a1*b11+a2*b21+a3*b31+a4*b41, a1*b12+a2*b22+a3*b32+a4*b42]
    #   This is 1x2, schoolbook rank = 4*4*1... no, 1*4*2 = 8
    #   Can we do better? The tensor is T[a,b,c] where a in [0,3], b in [0,7], c in [0,1]
    #   T[a, 2b+c, c] = 1 if... actually this is just a sum, rank = 8 (no structure to exploit)
    #   Actually rank of 1x4x2 tensor: it's a bilinear form, rank = 8 (generic)
    #
    # Total: 3 blocks of (2x4*4x2) with rank 14 each = 42
    #        3 blocks of (1x4*4x2) with rank 8 each = 24
    #        Total = 66
    #
    # That gives rank 66! Let me construct this.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, sign_u=None, sign_v=None, sign_w=None):
        """Add a Strassen 2x2x2 decomposition.
        u_offset: (i_row, j_col) base for U (within the 2x2 block)
        v_offset: (j_row, k_col) base for V
        w_offset: (k_row, i_col) base for W
        """
        sU, sV, sW = _strassen_decompose_2x2x2()
        if sign_u:
            sU = sU * sign_u
        if sign_v:
            sV = sV * sign_v
        if sign_w:
            sW = sW * sign_w
        return sU, sV, sW
    
    # We need to build U of shape (12, rank), V of shape (24, rank), W of shape (18, rank)
    # U index: i*4+j for i in [0,2], j in [0,3]
    # V index: j*6+k for j in [0,3], k in [0,5]
    # W index: k*3+i for k in [0,5], i in [0,2]
    
    # Let me build this more carefully by tracking column indices.
    
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        """Add a single term to the decomposition."""
        U.append((u_row, u_val))
        V.append((v_row, v_val))
        W.append((w_row, w_val))
    
    def add_strassen_block(i0, j0, k0, u_scale=1, v_scale=1, w_scale=1):
        """Add a Strassen 2x2x2 block at position (i0,j0) for A, (j0,k0) for B, (k0,i0) for C.
        i0: starting row in A (0-indexed), must have i0+1 <= n
        j0: starting col in A / row in B
        k0: starting col in B / row in C
        """
        sU, sV, sW = _strassen_decompose_2x2x2()
        for r in range(7):
            for (ui, val) in enumerate(sU[:, r]):
                if val != 0:
                    u_row = (i0 + ui // 2) * 4 + (j0 + ui % 2)
                    add_term(u_row, (j0 + r // 2) * 6 + (k0 + r % 2), (k0 + r // 2) * 3 + (i0 + r % 2), 
                           u_val=val*u_scale, v_val=v_scale, w_val=w_scale)
                    break  # Each r has at most one non-zero per matrix in Strassen
            # Actually let me redo this - each Strassen product r has specific non-zero entries
            for ui in range(4):
                if sU[ui, r] != 0:
                    i_idx = ui // 2
                    j_idx = ui % 2
                    u_row = (i0 + i_idx) * 4 + (j0 + j_idx)
                    add_term(u_row, -1, -1, u_val=sU[ui, r] * u_scale, v_val=1, w_val=1)
            for vi in range(4):
                if sV[vi, r] != 0:
                    j_idx = vi // 2
                    k_idx = vi % 2
                    v_row = (j0 + j_idx) * 6 + (k0 + k_idx)
                    add_term(-1, v_row, -1, u_val=1, v_val=sV[vi, r] * v_scale, w_val=1)
            for wi in range(4):
                if sW[wi, r] != 0:
                    k_idx = wi // 2
                    i_idx = wi % 2
                    w_row = (k0 + k_idx) * 3 + (i0 + i_idx)
                    add_term(-1, -1, w_row, u_val=1, v_val=1, w_val=sW[wi, r] * w_scale)
    
    # This is getting complicated. Let me just directly build the matrices.
    
    # Clear and restart with a cleaner approach
    # Pre-allocate with a safe upper bound on rank
    MAX_RANK = 100
    U = [[0] * MAX_RANK for _ in range(12)]
    V = [[0] * MAX_RANK for _ in range(24)]
    W = [[0] * MAX_RANK for _ in range(18)]
    
    col = 0
    
    def add_strassen_2x2x2_block(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        sU, sV, sW = _strassen_decompose_2x2x2()
        for r in range(7):
            # U contributions
            for ui in range(4):
                if sU[ui, r] != 0:
                    i_idx = ui // 2
                    j_idx = ui % 2
                    u_row = (i0 + i_idx) * 4 + (j0 + j_idx)
                    U[u_row][col] = sU[ui, r]
            # V contributions
            for vi in range(4):
                if sV[vi, r] != 0:
                    j_idx = vi // 2
                    k_idx = vi % 2
                    v_row = (j0 + j_idx) * 6 + (k0 + k_idx)
                    V[v_row][col] = sV[vi, r]
            # W contributions
            for wi in range(4):
                if sW[wi, r] != 0:
                    k_idx = wi // 2
                    i_idx = wi % 2
                    w_row = (k0 + k_idx) * 3 + (i0 + i_idx)
                    W[w_row][col] = sW[wi, r]
            col += 1
    
    def add_simple_1x2x2_block(i0, j0, k0):
        """Add schoolbook decomposition for A[i0, j0:j0+4] * B[j0:j0+4, k0:k0+2] -> C[i0, k0:k0+2]
        Actually this is 1x4 * 4x2, rank 8.
        A[i0, j] * B[j, k] for j in [0,3], k in [0,1]
        """
        nonlocal col
        for j in range(4):
            for k in range(2):
                u_row = i0 * 4 + j0 + j
                v_row = (j0 + j) * 6 + k0 + k
                w_row = (k0 + k) * 3 + i0
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    # Now let's build the 3x4x6 decomposition:
    # Split: A = [A_T(2x4); A_B(1x4)], B = [B_L(4x2) B_M(4x2) B_R(4x2)]
    # 
    # C_TL = A_T * B_L: 2x4 * 4x2
    #   A_T = [A_TL(2x2) A_TR(2x2)], B_L = [B_TL(2x2); B_BL(2x2)]
    #   C_TL = A_TL * B_TL + A_TR * B_BL
    #   A_TL * B_TL: Strassen at (i0=0, j0=0, k0=0)
    #   A_TR * B_BL: Strassen at (i0=0, j0=2, k0=0)
    
    # C_TM = A_T * B_M: 2x4 * 4x2
    #   A_T = [A_TL(2x2) A_TR(2x2)], B_M = [B_TM(2x2); B_BM(2x2)]
    #   C_TM = A_TL * B_TM + A_TR * B_BM
    #   A_TL * B_TM: Strassen at (i0=0, j0=0, k0=2)
    #   A_TR * B_BM: Strassen at (i0=0, j0=2, k0=2)
    
    # C_TR = A_T * B_R: 2x4 * 4x2
    #   A_T = [A_TL(2x2) A_TR(2x2)], B_R = [B_TR(2x2); B_BR(2x2)]
    #   C_TR = A_TL * B_TR + A_TR * B_BR
    #   A_TL * B_TR: Strassen at (i0=0, j0=0, k0=4)
    #   A_TR * B_BR: Strassen at (i0=0, j0=2, k0=4)
    
    # C_BL = A_B * B_L: 1x4 * 4x2, rank 8
    # C_BM = A_B * B_M: 1x4 * 4x2, rank 8
    # C_BR = A_B * B_R: 1x4 * 4x2, rank 8
    
    # Total rank: 3 * 2 * 7 + 3 * 8 = 42 + 24 = 66
    
    # Strassen blocks for top part (i0=0):
    # A_TL * B_TL: i0=0, j0=0, k0=0
    add_strassen_2x2x2_block(0, 0, 0)
    # A_TR * B_BL: i0=0, j0=2, k0=0
    add_strassen_2x2x2_block(0, 2, 0)
    # A_TL * B_TM: i0=0, j0=0, k0=2
    add_strassen_2x2x2_block(0, 0, 2)
    # A_TR * B_BM: i0=0, j0=2, k0=2
    add_strassen_2x2x2_block(0, 2, 2)
    # A_TL * B_TR: i0=0, j0=0, k0=4
    add_strassen_2x2x2_block(0, 0, 4)
    # A_TR * B_BR: i0=0, j0=2, k0=4
    add_strassen_2x2x2_block(0, 2, 4)
    
    # Simple blocks for bottom part (i0=2):
    # A_B * B_L: i0=2, j0=0, k0=0
    add_simple_1x2x2_block(2, 0, 0)
    # A_B * B_M: i0=2, j0=0, k0=2
    add_simple_1x2x2_block(2, 0, 2)
    # A_B * B_R: i0=2, j0=0, k0=4
    add_simple_1x2x2_block(2, 0, 4)
    
    rank = col
    # Trim U, V, W to rank columns
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                
                if j1 != j2 or k1 != k2 or i1 != i2:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err = abs(val)
                    max_err = max(max_err, err)
    
    return max_err <= tol

def _find_low_rank_decomposition(max_rank=72, deadline=None):
    n, m, p = DIMENSIONS
    best_result = _construct_schoolbook()
    best_rank = n * m * p
    
    # Try structured decomposition first
    structured = _construct_structured_decomposition()
    if _verify_decomposition(structured["U"], structured["V"], structured["W"]):
        best_result = structured
        best_rank = len(structured["U"][0])
    
    return best_result

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Try structured decomposition
    if _time_left(deadline) > 5:
        structured = _construct_structured_decomposition()
        if _verify_decomposition(structured["U"], structured["V"], structured["W"]):
            return structured
    
    # Fallback to schoolbook
    return _construct_schoolbook()
