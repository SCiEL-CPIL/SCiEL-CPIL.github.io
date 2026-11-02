import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # Split m=5 into 2, 2, 1.
    # Block 1: M_{2,2,6} with rank 15 (7 Strassen + 8 Schoolbook)
    # Block 2: M_{2,2,6} with rank 15 (7 Strassen + 8 Schoolbook)
    # Block 3: M_{2,1,6} with rank 12 (Schoolbook)
    # Total rank = 15 + 15 + 12 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a product term
    def add_term(a_idx, b_idx, c_coeffs):
        nonlocal col
        # a_idx is (i, j) for A
        # b_idx is (j, k) for B
        # c_coeffs is dict of (i, k) to coefficient
        U[a_idx[0] * m + a_idx[1], col] = 1.0
        V[b_idx[0] * p + b_idx[1], col] = 1.0
        for (ci, ck), coeff in c_coeffs.items():
            W[ck * n + ci, col] = coeff
        col += 1

    # Block 1: j=0,1
    # Strassen for k=0,1
    # A indices: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
    # B indices: (0,0)=B00, (0,1)=B01, (1,0)=B10, (1,1)=B11
    
    # M1 = (A00+A11)(B00+B11)
    # This is not a single product term in U,V,W form directly, 
    # but we can express it as sum of products? No, U,V,W are linear.
    # We need to decompose the product of sums into linear terms?
    # No, the tensor decomposition allows U,V,W entries to be any numbers.
    # But the product is U[:,c] * V[:,c].
    # So we cannot have (A00+A11) in U. U must pick a single A element.
    # Wait, the decomposition is T = sum U_c V_c W_c.
    # Each term is a rank-1 tensor.
    # Rank-1 tensor for matrix mult is A_{i,j} B_{j,k} C_{i,k}.
    # So each term corresponds to a single (i,j,k) triple?
    # No, that's schoolbook.
    # Strassen works by combining terms.
    # But U, V, W are matrices.
    # U[:, c] is a vector.
    # If U[:, c] has multiple non-zeros, it's not a rank-1 term in the standard sense?
    # Yes, it is. The term is (sum_i U_{i,c} A_i) * (sum_j V_{j,c} B_j) * (sum_k W_{k,c} C_k).
    # This allows linear combinations.
    
    # So for M1 = (A00+A11)(B00+B11):
    # We need U to have 1 at A00 and 1 at A11.
    # V to have 1 at B00 and 1 at B11.
    # W to have coefficients for C.
    
    def add_combined_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        for (ai, aj), coeff in a_coeffs.items():
            U[ai * m + aj, col] += coeff
        for (bj, bk), coeff in b_coeffs.items():
            V[bj * p + bk, col] += coeff
        for (ci, ck), coeff in c_coeffs.items():
            W[ck * n + ci, col] += coeff
        col += 1

    # Block 1: j=0,1, k=0,1 (Strassen)
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1 - M4 - M6 + M7 ? 
    # Standard Strassen reconstruction:
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M5 + M3
    
    # We will add the 7 terms and accumulate into W.
    
    # Term 1
    add_combined_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}) 
    # Wait, the coefficients in W depend on the reconstruction.
    # It is better to add the terms to a temporary accumulator for W?
    # No, W is fixed per column.
    # We must determine the W coefficients for each of the 7 products.
    
    # Let's define the 7 products and their contributions to C.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = P1 - P4 - P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 - P5 + P3
    
    # We will create 7 columns for these products.
    
    # Col 0: P1
    add_combined_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}) # Partial W, will adjust
    # This approach of adding to W is tricky because W is per column.
    # We need to set W for each column such that sum W_c * (U_c V_c) = C.
    
    # Let's just fill U, V, W for each of the 7 products with the correct W coefficients.
    
    # P1
    U[0*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+0, col] = 1; V[1*p+1, col] = 1
    W[0*n+0, col] = 1 # C00
    W[1*n+0, col] = 1 # C10? No, C11?
    # Let's map indices carefully.
    # C[i,k]
    # P1 contributes to C00, C01, C10, C11?
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00B00 -> C00
    # A00B11 -> C01 (B11 is j=1, k=1? No B_{j,k}. B11 is j=1, k=1? No B_{1,1} is j=1, k=1. A00 is i=0, j=0.
    # A_{0,0} B_{1,1} -> C_{0,1}? No, sum over j.
    # A_{0,0} B_{0,0} -> C_{0,0}
    # A_{0,1} B_{1,0} -> C_{0,0}
    # A_{1,0} B_{0,0} -> C_{1,0}
    # A_{1,1} B_{1,0} -> C_{1,0}
    # A_{0,0} B_{0,1} -> C_{0,1}
    # A_{0,1} B_{1,1} -> C_{0,1}
    # A_{1,0} B_{0,1} -> C_{1,1}
    # A_{1,1} B_{1,1} -> C_{1,1}
    
    # So P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00B00 -> C00
    # A00B11 -> C01 (j=1, k=1? No B_{1,1} is j=1, k=1. A_{0,1} is j=1.
    # Wait, A00 is A[0,0]. A11 is A[1,1].
    # B00 is B[0,0]. B11 is B[1,1].
    # A[0,0] B[0,0] -> C[0,0]
    # A[0,0] B[1,1] -> C[0,1]? No, B[1,1] has j=1. A[0,0] has j=0. They don't match in j.
    # The product is sum_j A[i,j] B[j,k].
    # (A[0,0]+A[1,1]) (B[0,0]+B[1,1]) is NOT a valid term in the sum because j must match.
    # Strassen works on blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # P1 = (A11+A22)(B11+B22)
    # Here A11 is block A[0,0], A22 is block A[1,1].
    # B11 is block B[0,0], B22 is block B[1,1].
    # This is valid.
    
    # So for Block 1 (j=0,1):
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    
    # P1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # This term contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    # Specifically:
    # A[0,0]B[0,0] -> C[0,0]
    # A[0,0]B[1,1] -> C[0,1] (j=1 in B, k=1)
    # A[1,1]B[0,0] -> C[1,0] (j=0 in B, k=0)
    # A[1,1]B[1,1] -> C[1,1]
    
    # So W for P1 should have 1 at (0,0), (0,1), (1,0), (1,1).
    
    add_combined_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1})
    col += 1
    
    # P2 = (A[1,0]+A[1,1])B[0,0]
    add_combined_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1})
    col += 1
    
    # P3 = A[0,0](B[1,0]-B[1,1])
    add_combined_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):-1})
    col += 1
    
    # P4 = A[1,1](B[0,1]-B[0,0])
    add_combined_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):-1, (1,1):1})
    col += 1
    
    # P5 = (A[0,0]+A[0,1])B[1,1]
    add_combined_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1})
    col += 1
    
    # P6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    add_combined_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1})
    col += 1
    
    # P7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    add_combined_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1})
    col += 1
    
    # Now we have 7 columns. We need to combine them to get C.
    # But the columns are separate. The sum of U_c V_c W_c over c must equal C.
    # The W coefficients above are the contributions of each product to C.
    # We must ensure that the linear combination of these 7 terms gives the correct C.
    # The reconstruction formulas give the coefficients.
    # C00 = P1 - P4 - P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 - P5 + P3
    
    # We have stored the "raw" contribution of each P to each C element in W.
    # We need to scale the columns (or W) to match the reconstruction.
    # Actually, we can just set W to the reconstruction coefficients.
    # But U and V are fixed.
    # We can multiply W by the coefficient.
    
    # Let's re-define the W for these 7 columns to be the reconstruction coefficients.
    # But the reconstruction is a linear combination of the P's.
    # This means we need to scale the entire term (U, V, W).
    # We can scale W.
    
    # Col 0 (P1): W should be (1, 0, 0, 1) for C00, C01, C10, C11?
    # No, P1 contributes to all 4.
    # In the reconstruction, P1 has coeff 1 for C00, 0 for C01, 0 for C10, 1 for C11.
    # So we set W for Col 0 to (1, 0, 0, 1).
    
    # We will overwrite W for the first 7 columns.
    
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 in W row (k*n+i)
    # W row index = k*n + i.
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 2
    # (1,1) -> 3
    
    # P1 (Col 0)
    W[:, 0] = 0
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # P2 (Col 1)
    W[:, 1] = 0
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # P3 (Col 2)
    W[:, 2] = 0
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # P4 (Col 3)
    W[:, 3] = 0
    W[0, 3] = -1 # C00
    W[2, 3] = 1 # C10
    
    # P5 (Col 4)
    W[:, 4] = 0
    W[1, 4] = 1 # C01
    W[3, 4] = -1 # C11
    
    # P6 (Col 5)
    W[:, 5] = 0
    W[0, 5] = -1 # C00
    
    # P7 (Col 6)
    W[:, 6] = 0
    W[0, 6] = 1 # C00
    
    # Now handle remaining k=2..5 for Block 1 (j=0,1)
    # Schoolbook: 2 products per k.
    # A[0,0]B[0,k] -> C[0,k]
    # A[0,1]B[1,k] -> C[0,k]
    # A[1,0]B[0,k] -> C[1,k]
    # A[1,1]B[1,k] -> C[1,k]
    
    for k in range(2, 6):
        # Term 1: A[0,0]B[0,k]
        U[0*m+0, col] = 1
        V[0*p+k, col] = 1
        W[k*n+0, col] = 1
        col += 1
        
        # Term 2: A[0,1]B[1,k]
        U[0*m+1, col] = 1
        V[1*p+k, col] = 1
        W[k*n+0, col] = 1
        col += 1
        
        # Term 3: A[1,0]B[0,k]
        U[1*m+0, col] = 1
        V[0*p+k, col] = 1
        W[k*n+1, col] = 1
        col += 1
        
        # Term 4: A[1,1]B[1,k]
        U[1*m+1, col] = 1
        V[1*p+k, col] = 1
        W[k*n+1, col] = 1
        col += 1

    # Block 2: j=2,3
    # Same structure.
    # A indices: (0,2), (0,3), (1,2), (1,3)
    # B indices: (2,0)..(3,5)
    
    # Strassen for k=0,1
    # P1
    U[0*m+2, col] = 1; U[1*m+3, col] = 1
    V[2*p+0, col] = 1; V[3*p+1, col] = 1
    W[:, col] = 0
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # P2
    U[1*m+2, col] = 1; U[1*m+3, col] = 1
    V[2*p+0, col] = 1
    W[:, col] = 0
    W[2, col] = 1; W[3, col] = -1
    col += 1
    
    # P3
    U[0*m+2, col] = 1
    V[3*p+0, col] = 1; V[3*p+1, col] = -1
    W[:, col] = 0
    W[1, col] = 1; W[3, col] = 1
    col += 1
    
    # P4
    U[1*m+3, col] = 1
    V[2*p+1, col] = 1; V[2*p+0, col] = -1
    W[:, col] = 0
    W[0, col] = -1; W[2, col] = 1
    col += 1
    
    # P5
    U[0*m+2, col] = 1; U[0*m+3, col] = 1
    V[3*p+1, col] = 1
    W[:, col] = 0
    W[1, col] = 1; W[3, col] = -1
    col += 1
    
    # P6
    U[1*m+2, col] = 1; U[0*m+2, col] = -1
    V[2*p+0, col] = 1; V[2*p+1, col] = 1
    W[:, col] = 0
    W[0, col] = -1
    col += 1
    
    # P7
    U[0*m+3, col] = 1; U[1*m+3, col] = -1
    V[3*p+0, col] = 1; V[3*p+1, col] = 1
    W[:, col] = 0
    W[0, col] = 1
    col += 1
    
    # Schoolbook for k=2..5
    for k in range(2, 6):
        U[0*m+2, col] = 1; V[2*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[0*m+3, col] = 1; V[3*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1*m+2, col] = 1; V[2*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        U[1*m+3, col] = 1; V[3*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

    # Block 3: j=4
    # Schoolbook for all k
    for k in range(6):
        U[0*m+4, col] = 1; V[4*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1*m+4, col] = 1; V[4*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

    return {"U": U, "V": V, "W": W}
