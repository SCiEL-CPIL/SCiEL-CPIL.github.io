import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Use Strassen's algorithm on the top-left 2x2 block of the output (C[0:2, 0:2]).
       This reduces the cost for these 4 outputs from 12 scalar multiplications to 7.
       The Strassen terms involve linear combinations of A[0:2, 0:2] and B[0:2, 0:2].
       Specifically, we use the standard 7 multiplications.
       
    2. For the remaining outputs (C[0:2, 2:4] and C[2, 0:4]), use the standard
       schoolbook algorithm. 
       - C[0, 2], C[0, 3] depend on A[0, :] and B[:, 2], B[:, 3].
       - C[1, 2], C[1, 3] depend on A[1, :] and B[:, 2], B[:, 3].
       - C[2, 0], C[2, 1], C[2, 2], C[2, 3] depend on A[2, :] and B[:, :].
       
       Total schoolbook terms:
       - C[0,2]: 3 terms
       - C[0,3]: 3 terms
       - C[1,2]: 3 terms
       - C[1,3]: 3 terms
       - C[2,0]: 3 terms
       - C[2,1]: 3 terms
       - C[2,2]: 3 terms
       - C[2,3]: 3 terms
       Total = 24 terms.
       
    Total rank = 7 (Strassen) + 24 (Schoolbook) = 31.
    
    Interface:
    U: shape (n*m, rank) = (9, 31). Index i*m+j corresponds to A[i,j].
    V: shape (m*p, rank) = (12, 31). Index j*p+k corresponds to B[j,k].
    W: shape (p*n, rank) = (12, 31). Index k*n+i corresponds to C[i,k].
    """
    n, m, p = 3, 3, 4
    rank = 31
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][val] = 1.0
        V[v_idx][val] = 1.0
        W[w_idx][val] = 1.0

    def add_term_combo(u_coeffs, v_coeffs, w_coeffs, val):
        """
        u_coeffs: list of (idx, coeff)
        v_coeffs: list of (idx, coeff)
        w_coeffs: list of (idx, coeff)
        """
        for idx, coeff in u_coeffs:
            U[idx][val] += coeff
        for idx, coeff in v_coeffs:
            V[idx][val] += coeff
        for idx, coeff in w_coeffs:
            W[idx][val] += coeff

    # --- Strassen Part (Terms 0-6) ---
    # Standard Strassen for C[0:2, 0:2]
    # Inputs: A[0:2, 0:2], B[0:2, 0:2]
    # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C indices (k*n+i):
    # C[0,0] -> 0*3+0 = 0
    # C[0,1] -> 1*3+0 = 3
    # C[1,0] -> 0*3+1 = 1
    # C[1,1] -> 1*3+1 = 4
    
    # M1 = (A00 + A11) * (B00 + B11)
    # C00 += M1, C11 += M1
    add_term_combo(
        [(0, 1), (4, 1)],  # A00, A11
        [(0, 1), (4, 1)],  # B00, B11
        [(0, 1), (4, 1)],  # C00, C11
        0
    )

    # M2 = A10 * (B00 + B01)
    # C10 += M2
    add_term_combo(
        [(3, 1)],  # A10
        [(0, 1), (1, 1)], # B00, B01
        [(1, 1)],  # C10
        1
    )

    # M3 = (A00 - A10) * (B10 + B11)
    # C00 += M3, C01 -= M3
    add_term_combo(
        [(0, 1), (3, -1)], # A00 - A10
        [(3, 1), (4, 1)],  # B10 + B11
        [(0, 1), (3, -1)], # C00 + M3, C01 - M3
        2
    )

    # M4 = (A10 + A11) * (B11 - B01)
    # C10 += M4, C11 -= M4
    add_term_combo(
        [(3, 1), (4, 1)],  # A10 + A11
        [(4, 1), (1, -1)], # B11 - B01
        [(1, 1), (4, -1)], # C10 + M4, C11 - M4
        3
    )

    # M5 = (A00 + A01) * (B00 - B10)
    # C01 += M5
    add_term_combo(
        [(0, 1), (1, 1)],  # A00 + A01
        [(0, 1), (3, -1)], # B00 - B10
        [(3, 1)],  # C01
        4
    )

    # M6 = (A01 - A11) * (B10 + B11)
    # C00 -= M6, C11 += M6
    add_term_combo(
        [(1, 1), (4, -1)], # A01 - A11
        [(3, 1), (4, 1)],  # B10 + B11
        [(0, -1), (4, 1)], # C00 - M6, C11 + M6
        5
    )

    # M7 = (A00 + A11) * (B01 - B11)
    # C01 += M7
    add_term_combo(
        [(0, 1), (4, 1)],  # A00 + A11
        [(1, 1), (4, -1)], # B01 - B11
        [(3, 1)],  # C01
        6
    )

    # --- Schoolbook Part (Terms 7-30) ---
    # We need to compute:
    # C[0, 2], C[0, 3]
    # C[1, 2], C[1, 3]
    # C[2, 0], C[2, 1], C[2, 2], C[2, 3]
    
    term_idx = 7
    
    # Helper to add schoolbook terms for C[i, k]
    def add_schoolbook(i, k, term_idx):
        # C[i, k] = sum_j A[i, j] B[j, k]
        # C index: k*n + i
        c_idx = k * n + i
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + k
            U[a_idx][term_idx] = 1.0
            V[b_idx][term_idx] = 1.0
            W[c_idx][term_idx] = 1.0
            term_idx += 1
        return term_idx

    # C[0, 2]
    term_idx = add_schoolbook(0, 2, term_idx)
    # C[0, 3]
    term_idx = add_schoolbook(0, 3, term_idx)
    
    # C[1, 2]
    term_idx = add_schoolbook(1, 2, term_idx)
    # C[1, 3]
    term_idx = add_schoolbook(1, 3, term_idx)
    
    # C[2, 0]
    term_idx = add_schoolbook(2, 0, term_idx)
    # C[2, 1]
    term_idx = add_schoolbook(2, 1, term_idx)
    # C[2, 2]
    term_idx = add_schoolbook(2, 2, term_idx)
    # C[2, 3]
    term_idx = add_schoolbook(2, 3, term_idx)
    
    # term_idx should be 31
    if term_idx != rank:
        raise ValueError(f"Expected rank 31, got {term_idx}")

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
