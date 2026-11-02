import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_full(U, V, W, n, m, p):
    """Full verification: T[i,j,k]=1 iff indices match, else 0."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U_arr, V_arr, W_arr)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
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
    return U, V, W, rank

def _strassen_2x2x2(i1, i2, j1, j2, k1, k2):
    """
    Return the 7 rank-1 terms for a 2x2x2 Strassen decomposition
    mapped to global indices.
    
    A is 2x2: rows (i1,i2), cols (j1,j2)
    B is 2x2: rows (j1,j2), cols (k1,k2)
    C is 2x2: rows (i1,i2), cols (k1,k2)
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    u1 = [(i1,j1), (i2,j2)]
    v1 = [(j1,k1), (j2,k2)]
    w1 = [(k1,i1), (k2,i2)]
    terms.append((u1, v1, w1, 1.0))
    
    # M2 = (A21+A22)B11
    u2 = [(i2,j1), (i2,j2)]
    v2 = [(j1,k1)]
    w2 = [(k1,i2)]
    terms.append((u2, v2, w2, 1.0))
    
    # M3 = A11(B12-B22)
    u3 = [(i1,j1)]
    v3 = [(j1,k2), (j2,k2)]
    w3 = [(k2,i1)]
    terms.append((u3, v3, w3, 1.0))
    terms.append((u3, v3, w3, -1.0))  # We'll handle signs by adding separate terms
    
    # Actually, let me redo this more carefully. Each term is a product of
    # a linear combination of A entries times a linear combination of B entries
    # times a linear combination of C entries (which should be 1 for the target).
    
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication means
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the sub-problem.
    
    # For Strassen, we express the 2x2x2 multiplication tensor as sum of 7
    # rank-1 tensors. Each rank-1 term is u ⊗ v ⊗ w where u, v, w are vectors
    # over the A, B, C indices respectively.
    
    # Let me define this properly.
    # A indices: (0,0), (0,1), (1,0), (1,1) corresponding to (i1,j1), (i1,j2), (i2,j1), (i2,j2)
    # B indices: (0,0), (0,1), (1,0), (1,1) corresponding to (j1,k1), (j1,k2), (j2,k1), (j2,k2)
    # C indices: (0,0), (0,1), (1,0), (1,1) corresponding to (i1,k1), (i1,k2), (i2,k1), (i2,k2)
    
    # The multiplication tensor for 2x2x2 has T[a,b,c] = 1 if the indices are
    # "compatible" in the matrix multiplication sense. Actually, for the full
    # tensor, T[i,j,k] = 1 for all (i,j,k) because we want C[i,k] = sum_j A[i,j]*B[j,k].
    
    # Wait, I need to be more careful. The tensor decomposition is for the
    # multiplication map: T(A,B)[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] where T[i,j,k] = 1 if we're computing C[i,k] using
    # A[i,j] and B[j,k]. So T[i,j,k] = 1 for all (i,j,k) in the 2x2x2 case.
    
    # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is over (i,j), v_r is over (j,k), w_r is over (k,i).
    
    # Strassen's 7 terms:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # The factorization is:
    # u1 = (1,0,0,1) over A indices (i0j0, i0j1, i1j0, i1j1)
    # v1 = (1,0,0,1) over B indices (j0k0, j0k1, j1k0, j1k1)
    # w1 = (1,0,0,1) over C indices (i0k0, i0k1, i1k0, i1k1)
    # But this doesn't work directly because of the C mapping.
    
    # Let me use the standard Strassen formulas and build the terms.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mr is a scalar. The tensor term for Mr is:
    # (coefficient on A) ⊗ (coefficient on B) ⊗ (coefficient on C)
    
    # M1: A part: (1,0,0,1), B part: (1,0,0,1), C part: (1,0,0,1) [contributes to C00 and C11]
    # But wait, M1 contributes to C00 with +1 and to C11 with +1.
    # So the C part should be (1, 0, 0, 1) in (C00, C01, C10, C11) order.
    
    # M2: A part: (0,0,1,1), B part: (1,0,0,0), C part: (0,0,1,0) [contributes to C10]
    # M3: A part: (1,0,0,0), B part: (0,1,0,-1), C part: (0,1,0,0) [contributes to C01]
    # M4: A part: (0,0,0,1), B part: (0,0,1,-1), C part: (1,0,1,0) [contributes to C00 and C10]
    # M5: A part: (1,1,0,0), B part: (0,0,0,1), C part: (-1,1,0,0) [contributes to C00 and C01]
    # M6: A part: (-1,0,1,0), B part: (1,1,0,0), C part: (0,0,0,1) [contributes to C11]
    # M7: A part: (0,1,0,-1), B part: (0,0,1,1), C part: (1,0,0,0) [contributes to C00]
    
    # Now I need to map local indices to global indices.
    # Local A index (li, lj) maps to global (I[li], J[lj])
    # Local B index (lj, lk) maps to global (J[lj], K[lk])
    # Local C index (li, lk) maps to global (I[li], K[lk])
    
    I = [i1, i2]
    J = [j1, j2]
    K = [k1, k2]
    
    def map_a(li, lj):
        return (I[li], J[lj])
    
    def map_b(lj, lk):
        return (J[lj], K[lk])
    
    def map_c(li, lk):
        return (I[li], K[lk])
    
    # Build the 7 terms
    # A indices in order: (0,0), (0,1), (1,0), (1,1)
    a_idx = [(0,0), (0,1), (1,0), (1,1)]
    # B indices in order: (0,0), (0,1), (1,0), (1,1)
    b_idx = [(0,0), (0,1), (1,0), (1,1)]
    # C indices in order: (0,0), (0,1), (1,0), (1,1)
    c_idx = [(0,0), (0,1), (1,0), (1,1)]
    
    terms_data = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    result = []
    for u_c, v_c, w_c in terms_data:
        u_list = []
        for coeff, (li, lj) in zip(u_c, a_idx):
            if coeff != 0:
                gi, gj = map_a(li, lj)
                u_list.append((gi, gj, coeff))
        
        v_list = []
        for coeff, (lj, lk) in zip(v_c, b_idx):
            if coeff != 0:
                gj, gk = map_b(lj, lk)
                v_list.append((gj, gk, coeff))
        
        w_list = []
        for coeff, (li, lk) in zip(w_c, c_idx):
            if coeff != 0:
                gi, gk = map_c(li, lk)
                w_list.append((gk, gi, coeff))  # W is indexed by (k, i)
        
        result.append((u_list, v_list, w_list))
    
    return result

def _schoolbook_1x2x2(i, j1, j2, k1, k2):
    """
    Return the 4 rank-1 terms for a 1x2x2 schoolbook decomposition
    mapped to global indices.
    
    A is 1x2: row i, cols (j1, j2)
    B is 2x2: rows (j1, j2), cols (k1, k2)
    C is 1x2: row i, cols (k1, k2)
    
    C[i,k1] = A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
    C[i,k2] = A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    
    4 terms:
    T1: A[i,j1] * B[j1,k1] * 1
    T2: A[i,j1] * B[j1,k2] * 1
    T3: A[i,j2] * B[j2,k1] * 1
    T4: A[i,j2] * B[j2,k2] * 1
    """
    terms = []
    terms.append(
        [((i, j1), 1)],
        [((j1, k1), 1)],
        [((k1, i), 1)]
    )
    terms.append(
        [((i, j1), 1)],
        [((j1, k2), 1)],
        [((k2, i), 1)]
    )
    terms.append(
        [((i, j2), 1)],
        [((j2, k1), 1)],
        [((k1, i), 1)]
    )
    terms.append(
        [((i, j2), 1)],
        [((j2, k2), 1)],
        [((k2, i), 1)]
    )
    return terms

def _build_block_decomposition(n, m, p):
    """
    Build the block decomposition for 3x4x4.
    
    Split J into {j1,j2} and {j3,j4}, K into {k1,k2} and {k3,k4}.
    For each of the 4 (J,K) blocks:
      - If I = {i1,i2}: use Strassen 2x2x2 (rank 7)
      - If I = {i3}: use schoolbook 1x2x2 (rank 4)
    
    Total rank: 4 * (7 + 4) = 44
    """
    all_terms = []
    
    J_blocks = [(0, 1), (2, 3)]  # (j1, j2) and (j3, j4) using 0-based
    K_blocks = [(0, 1), (2, 3)]  # (k1, k2) and (k3, k4) using 0-based
    
    I_two = (0, 1)  # i1, i2
    I_one = 2        # i3
    
    for j1, j2 in J_blocks:
        for k1, k2 in K_blocks:
            # Strassen for I = {i1, i2}
            strassen_terms = _strassen_2x2x2(I_two[0], I_two[1], j1, j2, k1, k2)
            all_terms.extend(strassen_terms)
            
            # Schoolbook for I = {i3}
            school_terms = _schoolbook_1x2x2(I_one, j1, j2, k1, k2)
            all_terms.extend(school_terms)
    
    # Now build U, V, W matrices
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    rank = len(all_terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_list, v_list, w_list) in enumerate(all_terms):
        for (gi, gj, coeff) in u_list:
            U[gi * m + gj][r] = coeff
        for (gj, gk, coeff) in v_list:
            V[gj * p + gk][r] = coeff
        for (gk, gi, coeff) in w_list:
            W[gk * n + gi][r] = coeff
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try the block decomposition (rank 44)
    U, V, W, rank = _build_block_decomposition(n, m, p)
    
    if _verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
