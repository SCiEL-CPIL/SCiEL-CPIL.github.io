"""Optimized Rank-22 decomposition for 2x3x4 matrix multiplication using Block Strassen."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_component(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
    
    # Indices for A (2x3): row i, col j -> i*m + j
    # Indices for B (3x4): row j, col k -> j*p + k
    # Indices for C (2x4): row i, col k -> k*n + i (Trace convention W[k*n+i])
    
    # --- Part 1: A_L (2x2) * B_T1 (2x2) ---
    # A_L = A[0:2, 0:2], B_T1 = B[0:2, 0:2]
    # Strassen for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # M1
    add_component(0, 0, 0, 1, 1, 1) # A11, B11, C11
    add_component(1, 0, 0, 0, 0, 0) # Placeholder, will overwrite below? No, better to sum into existing or use separate cols?
    # Wait, Strassen components are sums of products. Each term in the sum is a product of a linear combination of A entries and a linear combination of B entries.
    # So we can just add the contributions to the U, V, W matrices for that column.
    
    # Let's restart the logic for adding Strassen terms.
    # Each Strassen term M_i corresponds to one column in U, V, W.
    # U entries are coefficients of A entries.
    # V entries are coefficients of B entries.
    # W entries are coefficients of C entries.
    
    # Reset col
    col = 0
    
    # Helper to add a full Strassen term (which is a sum of products)
    # But wait, the tensor decomposition is a sum of rank-1 tensors.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # This is 4 rank-1 terms.
    # So for each Strassen multiplication, we need to expand it into rank-1 terms.
    
    # Let's define a function to add a rank-1 term: u_vec * v_vec * w_vec
    def add_rank1(u_dict, v_dict, w_dict):
        """
        u_dict: {A_index: coeff}
        v_dict: {B_index: coeff}
        w_dict: {C_index: coeff}
        """
        nonlocal col
        for idx, coeff in u_dict.items():
            U[idx, col] += coeff
        for idx, coeff in v_dict.items():
            V[idx, col] += coeff
        for idx, coeff in w_dict.items():
            W[idx, col] += coeff
        col += 1

    # A indices:
    # A11: 0, A12: 1, A21: 2, A22: 3
    # A31: 4, A32: 5, A33: 6 (for A_R part)
    
    # B indices:
    # B11: 0, B12: 1, B13: 2, B14: 3
    # B21: 4, B22: 5, B23: 6, B24: 7
    # B31: 8, B32: 9, B33: 10, B34: 11
    
    # C indices (k*n + i):
    # C11: 0 (k=0,i=0), C12: 1 (k=1,i=0), C13: 2 (k=2,i=0), C14: 3 (k=3,i=0)
    # C21: 2 (k=0,i=1), C22: 3 (k=1,i=1), C23: 4 (k=2,i=1), C24: 5 (k=3,i=1)
    
    # --- Part 1: A_L (2x2) * B_T1 (2x2) -> C_T1 (2x2) ---
    # C_T1 is C[0:2, 0:2]
    # C11 (0), C12 (1), C21 (2), C22 (3)
    
    # M1 = (A11+A22)(B11+B22)
    # Terms:
    add_rank1({0:1, 3:1}, {0:1, 5:1}, {0:1, 2:1}) # Wait, C11 and C21?
    # No, M1 contributes to C11 and C22?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's expand each M_i into rank-1 terms and add them to the respective C indices.
    
    # M1 = (A11+A22)(B11+B22)
    # A part: {0:1, 3:1}
    # B part: {0:1, 5:1}
    # C parts: C11 (0) and C22 (3)
    # M1 contributes +1 to C11 and +1 to C22.
    # But M1 is a single product of sums. It is not a single rank-1 tensor.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # So we need 4 rank-1 terms for M1 alone?
    # No, the rank of the 2x2x2 tensor is 7. This means we can represent the 8 products with 7 rank-1 terms.
    # The 7 terms are the M1..M7.
    # But each M_i is a product of a linear form in A and a linear form in B.
    # So each M_i IS a rank-1 tensor in the space of A and B, but its output is a scalar.
    # To form the output matrix C, we sum these scalars with coefficients.
    # So, for each M_i, we have one column in U, V, W.
    # U column has the coefficients of A for that linear form.
    # V column has the coefficients of B for that linear form.
    # W column has the coefficients of C for that scalar's contribution.
    
    # Let's re-implement add_rank1 to take a single linear form for A, B, and a set of C contributions.
    
    col = 0
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {A_idx: coeff}
        v_coeffs: dict {B_idx: coeff}
        w_coeffs: dict {C_idx: coeff}
        """
        nonlocal col
        for idx, coeff in u_coeffs.items():
            U[idx, col] = coeff
        for idx, coeff in v_coeffs.items():
            V[idx, col] = coeff
        for idx, coeff in w_coeffs.items():
            W[idx, col] = coeff
        col += 1

    # M1 = (A11+A22)(B11+B22)
    # Contributes to C11 (+1) and C22 (+1)
    add_strassen_term({0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    
    # M2 = (A21+A22)B11
    # Contributes to C21 (+1) and C22 (-1)
    add_strassen_term({2:1, 3:1}, {0:1}, {2:1, 3:-1})
    
    # M3 = A11(B12-B22)
    # Contributes to C12 (+1) and C22 (+1)
    add_strassen_term({0:1}, {1:1, 5:-1}, {1:1, 3:1})
    
    # M4 = A22(B21-B11)
    # Contributes to C11 (+1) and C21 (+1)
    add_strassen_term({3:1}, {4:1, 0:-1}, {0:1, 2:1})
    
    # M5 = (A11+A12)B22
    # Contributes to C11 (-1) and C12 (+1)
    add_strassen_term({0:1, 1:1}, {5:1}, {0:-1, 1:1})
    
    # M6 = (A21-A11)(B11+B12)
    # Contributes to C22 (+1)
    add_strassen_term({2:1, 0:-1}, {0:1, 1:1}, {3:1})
    
    # M7 = (A12-A22)(B21+B22)
    # Contributes to C11 (+1)
    add_strassen_term({1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # --- Part 2: A_L (2x2) * B_T2 (2x2) -> C_T2 (2x2) ---
    # B_T2 = B[0:2, 2:4]
    # B indices: B13(2), B14(3), B23(6), B24(7)
    # C_T2 = C[0:2, 2:4]
    # C indices: C13(2), C14(3), C23(4), C24(5)
    
    # M1 = (A11+A22)(B13+B23)
    add_strassen_term({0:1, 3:1}, {2:1, 6:1}, {2:1, 5:1})
    
    # M2 = (A21+A22)B13
    add_strassen_term({2:1, 3:1}, {2:1}, {4:1, 5:-1})
    
    # M3 = A11(B14-B23)
    add_strassen_term({0:1}, {3:1, 6:-1}, {3:1, 5:1})
    
    # M4 = A22(B23-B13)
    add_strassen_term({3:1}, {6:1, 2:-1}, {2:1, 4:1})
    
    # M5 = (A11+A12)B23
    add_strassen_term({0:1, 1:1}, {6:1}, {2:-1, 3:1})
    
    # M6 = (A21-A11)(B13+B14)
    add_strassen_term({2:1, 0:-1}, {2:1, 3:1}, {5:1})
    
    # M7 = (A12-A22)(B23+B24)
    add_strassen_term({1:1, 3:-1}, {6:1, 7:1}, {2:1})
    
    # --- Part 3: A_R (2x1) * B_B (1x4) -> C_RB (2x4) ---
    # A_R = A[0:2, 2:3] -> A31(4), A32(5)
    # B_B = B[2:3, 0:4] -> B31(8), B32(9), B33(10), B34(11)
    # C_RB = C[0:2, 0:4]
    # C indices: C11(0), C12(1), C13(2), C14(3), C21(2), C22(3), C23(4), C24(5)
    
    # This is a simple outer product sum.
    # C_i,k = A_i,2 * B_2,k
    # 8 rank-1 terms.
    
    # C11 (0) = A31 * B31
    add_strassen_term({4:1}, {8:1}, {0:1})
    # C12 (1) = A31 * B32
    add_strassen_term({4:1}, {9:1}, {1:1})
    # C13 (2) = A31 * B33
    add_strassen_term({4:1}, {10:1}, {2:1})
    # C14 (3) = A31 * B34
    add_strassen_term({4:1}, {11:1}, {3:1})
    
    # C21 (2) = A32 * B31
    add_strassen_term({5:1}, {8:1}, {2:1})
    # C22 (3) = A32 * B32
    add_strassen_term({5:1}, {9:1}, {3:1})
    # C23 (4) = A32 * B33
    add_strassen_term({5:1}, {10:1}, {4:1})
    # C24 (5) = A32 * B34
    add_strassen_term({5:1}, {11:1}, {5:1})
    
    # Total rank: 7 + 7 + 8 = 22
    # Check col == 22
    assert col == 22, f"Expected col 22, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
