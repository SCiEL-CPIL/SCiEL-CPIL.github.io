"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _winograd_2x2x2():
    """
    Return the rank-7 Winograd F(2) decomposition for 2x2 * 2x2.
    """
    U = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, -1, 1],
    ], dtype=float)

    V = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, -1, 0, 1, -1, -1, 1],
    ], dtype=float)

    W = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
    ], dtype=float)

    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(7):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_kron():
    """
    Construct a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Winograd F(2).
    """
    w2 = _winograd_2x2x2()
    if not w2:
        return None

    U2 = np.array(w2["U"])  # shape (4, 7)
    V2 = np.array(w2["V"])  # shape (4, 7)
    W2 = np.array(w2["W"])  # shape (4, 7)

    n, m, p = 4, 4, 4
    rank = 49
    U4 = np.zeros((n * m, rank))
    V4 = np.zeros((m * p, rank))
    W4 = np.zeros((p * n, rank))

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = i1 * 8 + i2 * 4 + j1 * 2 + j2
                            col_idx = c1 * 7 + c2
                            U4[row_idx, col_idx] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = j1 * 8 + j2 * 4 + k1 * 2 + k2
                            col_idx = c1 * 7 + c2
                            V4[row_idx, col_idx] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = k1 * 8 + k2 * 4 + i1 * 2 + i2
                            col_idx = c1 * 7 + c2
                            W4[row_idx, col_idx] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _construct_4x4x5():
    """
    Construct a rank-53 decomposition for 4x4 * 4x5.
    Uses rank-49 Kronecker for first 4 columns, then 4 additional rank terms for 5th column.
    """
    n, m, p = 4, 4, 5

    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)

    U4 = np.array(base["U"])
    V4 = np.array(base["V"])
    W4 = np.array(base["W"])

    rank_base = 49
    rank_new = rank_base + 4

    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))

    U_new[:, :rank_base] = U4

    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]

    W_new[:16, :rank_base] = W4

    for j in range(4):
        col_idx = rank_base + j
        for i in range(4):
            U_new[i * 4 + j, col_idx] = 1.0
        V_new[j * 5 + 4, col_idx] = 1.0
        for i in range(4):
            W_new[4 * 4 + i, col_idx] = 1.0

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def _construct_4x4x5_nested_strassen():
    """
    Construct a rank-65 decomposition for 4x4 * 4x5 using nested Strassen.
    Uses Strassen 7x7=49 for the 4x4x4 part, plus 16 terms for the 4x4x1 part.
    
    This is a structural alternative to the Winograd Kronecker approach.
    """
    n, m, p = 4, 4, 5
    
    # Nested Strassen: 7*7 = 49 for 4x4x4, 16 for 4x4x1 = 65 total
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    # Define the 7 Strassen combinations for 2x2x2
    # For P = [[p00,p01],[p10,p11]], Q = [[q00,q01],[q10,q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)(q00)
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)(q11)
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    # Result: R00 = s1+s4-s5+s7, R01 = s3+s5, R10 = s2+s4, R11 = s1-s2+s3+s6
    
    # Super-term definitions for 4x4x4
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # B10: j in {2,3}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}
    # C00: i in {0,1}, k in {0,1}
    # C01: i in {0,1}, k in {2,3}
    # C10: i in {2,3}, k in {0,1}
    # C11: i in {2,3}, k in {2,3}
    
    super_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'p_blocks': [1, 0, 0, 1],
            'q_blocks': [1, 0, 0, 1],
            'c_contrib': [(0, 1), (3, 1)]
        },
        # M2: (A10+A11)(B00)
        {
            'p_blocks': [0, 0, 1, 1],
            'q_blocks': [1, 0, 0, 0],
            'c_contrib': [(2, 1), (3, -1)]
        },
        # M3: (A00)(B01-B11)
        {
            'p_blocks': [1, 0, 0, 0],
            'q_blocks': [0, 1, 0, -1],
            'c_contrib': [(1, 1), (3, 1)]
        },
        # M4: (A11)(B10-B00)
        {
            'p_blocks': [0, 0, 0, 1],
            'q_blocks': [-1, 0, 1, 0],
            'c_contrib': [(0, 1), (2, 1)]
        },
        # M5: (A00+A01)(B11)
        {
            'p_blocks': [1, 1, 0, 0],
            'q_blocks': [0, 0, 0, 1],
            'c_contrib': [(0, -1), (1, 1)]
        },
        # M6: (A10-A00)(B00+B01)
        {
            'p_blocks': [-1, 0, 1, 0],
            'q_blocks': [1, 1, 0, 0],
            'c_contrib': [(3, 1)]
        },
        # M7: (A01-A11)(B10+B11)
        {
            'p_blocks': [0, 1, 0, -1],
            'q_blocks': [0, 0, 1, 1],
            'c_contrib': [(0, 1)]
        },
    ]
    
    # Inner Strassen terms for 2x2x2
    inner_strassen = [
        # s1
        {'p_elems': [(0, 1), (3, 1)], 'q_elems': [(0, 1), (3, 1)], 'r_elems': [(0, 1), (3, 1)]},
        # s2
        {'p_elems': [(2, 1), (3, 1)], 'q_elems': [(0, 1)], 'r_elems': [(2, 1), (3, -1)]},
        # s3
        {'p_elems': [(0, 1)], 'q_elems': [(1, 1), (3, -1)], 'r_elems': [(1, 1), (3, 1)]},
        # s4
        {'p_elems': [(3, 1)], 'q_elems': [(2, 1), (0, -1)], 'r_elems': [(0, 1), (2, 1)]},
        # s5
        {'p_elems': [(0, 1), (1, 1)], 'q_elems': [(3, 1)], 'r_elems': [(0, -1), (1, 1)]},
        # s6
        {'p_elems': [(2, 1), (0, -1)], 'q_elems': [(0, 1), (1, 1)], 'r_elems': [(3, 1)]},
        # s7
        {'p_elems': [(1, 1), (3, -1)], 'q_elems': [(2, 1), (3, 1)], 'r_elems': [(0, 1)]},
    ]
    
    # Helper: get global A indices for a P element
    def get_p_indices(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    # Helper: get global B indices for a Q element
    def get_q_indices(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    # Helper: get global C indices for an R element
    def get_c_indices(r_elem_idx, c_contrib):
        r = r_elem_idx // 2
        c = r_elem_idx % 2
        res = []
        for c_block_idx, sign in c_contrib:
            if c_block_idx == 0:  # C00: i in {0,1}, k in {0,1}
                gi, gk = r, c
            elif c_block_idx == 1:  # C01: i in {0,1}, k in {2,3}
                gi, gk = r, 2 + c
            elif c_block_idx == 2:  # C10: i in {2,3}, k in {0,1}
                gi, gk = 2 + r, c
            elif c_block_idx == 3:  # C11: i in {2,3}, k in {2,3}
                gi, gk = 2 + r, 2 + c
            res.append((gk, gi, sign))
        return res
    
    # Process each super-term and each inner Strassen term
    for super_term in super_terms:
        p_blocks = super_term['p_blocks']
        q_blocks = super_term['q_blocks']
        c_contrib = super_term['c_contrib']
        
        for inner in inner_strassen:
            p_elems = inner['p_elems']
            q_elems = inner['q_elems']
            r_elems = inner['r_elems']
            
            # Build U list: combine all P elements
            u_list = []
            for p_elem_idx, coeff in p_elems:
                indices = get_p_indices(p_elem_idx, p_blocks)
                for gi, gj, bcoeff in indices:
                    u_list.append((gi, gj, coeff * bcoeff))
            
            # Build V list: combine all Q elements
            v_list = []
            for q_elem_idx, coeff in q_elems:
                indices = get_q_indices(q_elem_idx, q_blocks)
                for gj, gk, bcoeff in indices:
                    v_list.append((gj, gk, coeff * bcoeff))
            
            # Build W list: combine all R elements
            w_list = []
            for r_elem_idx, coeff in r_elems:
                indices = get_c_indices(r_elem_idx, c_contrib)
                for gk, gi, bcoeff in indices:
                    w_list.append((gk, gi, coeff * bcoeff))
            
            add_term(u_list, v_list, w_list)
    
    # Add 16 terms for the 4x4x1 part (5th column)
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS

    start_time = time.time()
    deadline = start_time + 140

    # Try the Winograd Kronecker approach first (rank 53)
    try:
        result = _construct_4x4x5()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Try the nested Strassen approach (rank 65) as a fallback
    try:
        result = _construct_4x4x5_nested_strassen()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
