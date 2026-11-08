import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _verify_2x2x2_terms(terms):
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    total = 0.0
                    for U, V, W in terms:
                        total += U.get((a,b),0) * V.get((c,d),0) * W.get((a,d),0)
                    if abs(total - 1.0) > 1e-9:
                        return False
    return True

def _strassen_2x2x2():
    # 7 rank-1 terms for 2x2 * 2x2 -> 2x2 using Strassen
    # Each term: (U_dict, V_dict, W_dict)
    # U: (a,b) -> coeff for A[a,b]
    # V: (c,d) -> coeff for B[c,d]
    # W: (a,d) -> coeff for C[a,d]
    #
    # p1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    # p2 = (A10+A11)*B00       -> C10 += 1
    # p3 = A00*(B01-B11)       -> C01 += 1
    # p4 = A11*(B10-B00)       -> C11 += 1
    # p5 = (A00+A01)*B11       -> C00 += 1, C01 += 1
    # p6 = (A10-A00)*(B00+B01) -> C11 += 1
    # p7 = (A01-A11)*(B10+B11) -> C00 -= 1
    terms = [
        # p1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # p2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # p3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # p4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1}),
        # p5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1}),
        # p6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # p7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1}),
    ]
    # Verify
    if not _verify_2x2x2_terms(terms):
        raise ValueError("2x2x2 Strassen decomposition is incorrect")
    return terms

def _build_4x4x4_strassen(terms_2x2):
    """
    Build rank-49 terms for 4x4 * 4x4 -> 4x4 using Strassen.
    Returns list of (U_dict, V_dict, W_dict) with full global indices.
    U: (i,j) for A, V: (j,k) for B, W: (i,k) for C
    """
    all_terms = []
    
    # Strassen M definitions for 4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C outputs:
    # C11 = M1 - M3 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M4 + M6
    
    # Block definitions: (row_range, col_range)
    A_blocks = {
        'A11': (0,1,0,1),  # i in [0,1], j in [0,1]
        'A12': (0,1,2,3),  # i in [0,1], j in [2,3]
        'A21': (2,3,0,1),  # i in [2,3], j in [0,1]
        'A22': (2,3,2,3),  # i in [2,3], j in [2,3]
    }
    B_blocks = {
        'B11': (0,1,0,1),  # j in [0,1], k in [0,1]
        'B12': (0,1,2,3),  # j in [0,1], k in [2,3]
        'B21': (2,3,0,1),  # j in [2,3], k in [0,1]
        'B22': (2,3,2,3),  # j in [2,3], k in [2,3]
    }
    C_blocks = {
        'C11': (0,1,0,1),  # i in [0,1], k in [0,1]
        'C12': (0,1,2,3),  # i in [0,1], k in [2,3]
        'C21': (2,3,0,1),  # i in [2,3], k in [0,1]
        'C22': (2,3,2,3),  # i in [2,3], k in [2,3]
    }
    
    # M definitions: list of (A_coeffs, B_coeffs)
    M_defs = [
        ({'A11':1, 'A22':1}, {'B11':1, 'B22':1}),  # M1
        ({'A21':1, 'A22':1}, {'B11':1}),           # M2
        ({'A11':1}, {'B12':1, 'B22':-1}),           # M3
        ({'A22':1}, {'B21':1, 'B11':-1}),           # M4
        ({'A11':1, 'A12':1}, {'B22':1}),            # M5
        ({'A21':1, 'A11':-1}, {'B11':1, 'B12':1}),  # M6
        ({'A12':1, 'A22':-1}, {'B21':1, 'B22':1}),  # M7
    ]
    
    # C coefficients for each M: dict from C block name to sign
    C_signs = [
        {'C11':1, 'C22':1},      # M1
        {'C21':1},               # M2
        {'C11':-1, 'C12':1},     # M3
        {'C21':1, 'C22':1},      # M4
        {'C11':1, 'C12':1},      # M5
        {'C22':1},               # M6
        {'C11':-1},              # M7
    ]
    
    for m_idx in range(7):
        A_coeffs, B_coeffs = M_defs[m_idx]
        c_signs = C_signs[m_idx]
        
        for U2, V2, W2 in terms_2x2:
            U_full = {}
            V_full = {}
            W_full = {}
            
            # Build U_full
            for blk_name, coeff in A_coeffs.items():
                i0, i1, j0, j1 = A_blocks[blk_name]
                for a in range(2):
                    for b in range(2):
                        ig = i0 + a
                        jg = j0 + b
                        val = U2.get((a,b), 0)
                        if val != 0:
                            key = (ig, jg)
                            U_full[key] = U_full.get(key, 0) + coeff * val
            
            # Build V_full
            for blk_name, coeff in B_coeffs.items():
                j0, j1, k0, k1 = B_blocks[blk_name]
                for c in range(2):
                    for d in range(2):
                        jg = j0 + c
                        kg = k0 + d
                        val = V2.get((c,d), 0)
                        if val != 0:
                            key = (jg, kg)
                            V_full[key] = V_full.get(key, 0) + coeff * val
            
            # Build W_full
            for blk_name, coeff in c_signs.items():
                i0, i1, k0, k1 = C_blocks[blk_name]
                for a in range(2):
                    for d in range(2):
                        ig = i0 + a
                        kg = k0 + d
                        val = W2.get((a,d), 0)
                        if val != 0:
                            key = (ig, kg)
                            W_full[key] = W_full.get(key, 0) + coeff * val
            
            all_terms.append((U_full, V_full, W_full))
    
    return all_terms

def _build_4x4x2_schoolbook_strassen(terms_2x2):
    """
    Build terms for 4x4 * 4x2 -> 4x2.
    Split into 4 sub-products of 2x2 * 2x2, each using Strassen (rank 7).
    Total rank: 28.
    
    B_right has columns 4,5 of the full B (4x6).
    """
    all_terms = []
    
    # A blocks (same as before)
    A_blocks = {
        'A11': (0, 0, 1),  # i in [0,1], j in [0,1] -> (i0, j0, size)
        'A12': (0, 2, 2),  # i in [0,1], j in [2,3]
        'A21': (2, 0, 2),  # i in [2,3], j in [0,1]
        'A22': (2, 2, 2),  # i in [2,3], j in [2,3]
    }
    
    # B_right blocks: j in [0,3], k_local in [0,1] -> k_global = 4 + k_local
    # Br1: j in [0,1], k_local in [0,1]
    # Br2: j in [2,3], k_local in [0,1]
    B_blocks = {
        'Br1': (0, 0),  # j0, k0_local
        'Br2': (2, 0),
    }
    
    # C_right blocks: i in [0,3], k_local in [0,1] -> k_global = 4 + k_local
    # Cr1: i in [0,1], k_local in [0,1]
    # Cr2: i in [2,3], k_local in [0,1]
    C_blocks = {
        'Cr1': (0, 0),  # i0, k0_local
        'Cr2': (2, 0),
    }
    
    # 4 sub-products: (A_block, B_block, C_block)
    sub_prods = [
        ('A11', 'Br1', 'Cr1'),
        ('A12', 'Br2', 'Cr1'),
        ('A21', 'Br1', 'Cr2'),
        ('A22', 'Br2', 'Cr2'),
    ]
    
    for A_blk, B_blk, C_blk in sub_prods:
        a_i0, a_j0 = A_blocks[A_blk]
        b_j0, b_k0 = B_blocks[B_blk]
        c_i0, c_k0 = C_blocks[C_blk]
        
        for U2, V2, W2 in terms_2x2:
            U_full = {}
            V_full = {}
            W_full = {}
            
            # U: A elements
            for a in range(2):
                for b in range(2):
                    ig = a_i0 + a
                    jg = a_j0 + b
                    val = U2.get((a,b), 0)
                    if val != 0:
                        key = (ig, jg)
                        U_full[key] = U_full.get(key, 0) + val
            
            # V: B elements (k_global = 4 + k_local)
            for c in range(2):
                for d in range(2):
                    jg = b_j0 + c
                    kg = 4 + b_k0 + d
                    val = V2.get((c,d), 0)
                    if val != 0:
                        key = (jg, kg)
                        V_full[key] = V_full.get(key, 0) + val
            
            # W: C elements (k_global = 4 + k_local)
            for a in range(2):
                for d in range(2):
                    ig = c_i0 + a
                    kg = 4 + c_k0 + d
                    val = W2.get((a,d), 0)
                    if val != 0:
                        key = (ig, kg)
                        W_full[key] = W_full.get(key, 0) + val
            
            all_terms.append((U_full, V_full, W_full))
    
    return all_terms

def _full_schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _terms_to_matrices(all_terms, n, m, p):
    """Convert list of (U_dict, V_dict, W_dict) to U, V, W matrices."""
    rank = len(all_terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        Ud, Vd, Wd = all_terms[r]
        for (i, j), val in Ud.items():
            U[i * m + j][r] = val
        for (j, k), val in Vd.items():
            V[j * p + k][r] = val
        for (i, k), val in Wd.items():
            W[k * n + i][r]  # trace convention: W[k*n + i, r]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                ur = U[i * m + j]
                vr = V[j * p + k]
                wr = W[k * n + i]
                for r in range(len(ur)):
                    total += ur[r] * vr[r] * wr[r]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = U_arr.shape[1]
    # tensor[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Reshape and use einsum
    U_t = U_arr.reshape(n, m, rank)  # U[i*m+j, r]
    V_t = V_arr.reshape(m, p, rank)  # V[j*p+k, r]
    W_t = W_arr.reshape(p, n, rank)  # W[k*n+i, r]
    
    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    expected = np.ones((n, m, p))
    return np.allclose(result, expected, atol=tol)

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    
    # Get the verified 2x2 Strassen decomposition
    terms_2x2 = _strassen_2x2x2()
    
    # Build 4x4 * 4x4 part (k in 0..3)
    terms_444 = _build_4x4x4_strassen(terms_2x2)
    
    # Build 4x4 * 4x2 part (k in 4..5)
    terms_442 = _build_4x4x2_schoolbook_strassen(terms_2x2)
    
    all_terms = terms_444 + terms_442
    rank = len(all_terms)
    
    # Convert to matrices
    U, V, W = _terms_to_matrices(all_terms, n, m, p)
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    if not _verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p):
        # Fall back to schoolbook
        result = _full_schoolbook(n, m, p)
        return result
    
    return {"U": U, "V": V, "W": W}
