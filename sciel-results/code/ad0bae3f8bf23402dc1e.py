import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4 matrix multiplication
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 4x4 block decomposition
    # M_k = L_k(A) * R_k(B)
    # L_k and R_k are 2x2 matrices of linear forms in A and B blocks.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # Define L_k and R_k as 2x2 arrays of indices into A and B
    # A index: I*2 + i, J*2 + j -> (I*4 + J*2 + i)*4 + (J*2 + j)? No.
    # A is 4x4. Block I,J is rows 2I..2I+1, cols 2J..2J+1.
    # A[I*2+i, J*2+j]
    
    # We need to express L_k[i,j] as a sum of A[...].
    # And R_k[i,j] as a sum of B[...].
    
    # Helper to get A index
    def a_idx(I, i, J, j):
        return (I * 2 + i) * 4 + (J * 2 + j)
        
    # Helper to get B index
    def b_idx(I, i, J, j):
        return (I * 2 + i) * 4 + (J * 2 + j)
        
    # Helper to get W index (Output C)
    # C[row, col] is stored at W[col*4 + row]
    def w_idx(row, col):
        return col * 4 + row

    # Strassen 4x4 definitions
    # k=0..6
    # L_k is 2x2, R_k is 2x2
    
    # k=0: M1 = (A00+A11)(B00+B11)
    # L0 = [[A00, 0], [0, A11]]? No, L0 is a matrix.
    # L0[0,0] = A00, L0[0,1] = 0, L0[1,0] = 0, L0[1,1] = A11?
    # No, (A00+A11) is a single 2x2 matrix.
    # So L0 = A00 + A11.
    # L0[i,j] = A[2*0+i, 2*0+j] + A[2*1+i, 2*1+j]
    
    L_defs = [
        lambda i, j: [(a_idx(0, i, 0, j), 1), (a_idx(1, i, 1, j), 1)], # k=0
        lambda i, j: [(a_idx(1, i, 0, j), 1), (a_idx(1, i, 1, j), 1)], # k=1 (A10+A11)
        lambda i, j: [(a_idx(0, i, 0, j), 1)],                         # k=2 (A00)
        lambda i, j: [(a_idx(1, i, 1, j), 1)],                         # k=3 (A11)
        lambda i, j: [(a_idx(0, i, 0, j), 1), (a_idx(0, i, 1, j), 1)], # k=4 (A00+A01)
        lambda i, j: [(a_idx(1, i, 0, j), 1), (a_idx(0, i, 0, j), -1)],# k=5 (A10-A00)
        lambda i, j: [(a_idx(0, i, 1, j), 1), (a_idx(1, i, 1, j), -1)] # k=6 (A01-A11)
    ]
    
    R_defs = [
        lambda i, j: [(b_idx(0, i, 0, j), 1), (b_idx(1, i, 1, j), 1)], # k=0
        lambda i, j: [(b_idx(0, i, 0, j), 1)],                         # k=1
        lambda i, j: [(b_idx(0, i, 1, j), 1), (b_idx(1, i, 1, j), -1)],# k=2
        lambda i, j: [(b_idx(1, i, 0, j), 1), (b_idx(0, i, 0, j), -1)],# k=3
        lambda i, j: [(b_idx(1, i, 1, j), 1)],                         # k=4
        lambda i, j: [(b_idx(0, i, 0, j), 1), (b_idx(0, i, 1, j), 1)], # k=5
        lambda i, j: [(b_idx(1, i, 0, j), 1), (b_idx(1, i, 1, j), 1)]  # k=6
    ]
    
    # Strassen 2x2 decomposition coefficients for M_k
    # P1..P7
    # M[i,j] = sum_l T[i,j,l] P_l
    
    T = np.array([
        [1, 0, 0, 1, -1, 0, 1], # M[0,0] = P1 + P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0],  # M[0,1] = P3 + P5
        [0, 1, 0, 1, 0, 0, 0],  # M[1,0] = P2 + P4
        [1, -1, 1, 0, 0, 1, 0]  # M[1,1] = P1 - P2 + P3 + P6
    ])
    
    # Coefficients for P_l in terms of L and R
    # P1 = (L00+L11)(R00+R11)
    # P2 = (L10+L11)R00
    # P3 = L00(R01-R11)
    # P4 = L11(R10-R00)
    # P5 = (L00+L01)R11
    # P6 = (L10-L00)(R00+R01)
    # P7 = (L01-L11)(R10+R11)
    
    # We need to generate U and V for each (k, l)
    
    for k in range(7):
        for l in range(7):
            c = k * 7 + l
            
            # Determine U (coeffs for A)
            # U is 16x1. We need to fill U[idx, c]
            # P_l is a product of two linear forms: L_form * R_form
            # L_form is a sum of A elements.
            # R_form is a sum of B elements.
            
            # Get L elements for the specific P_l
            # We need to identify which L[i,j] and R[i,j] are involved and their coefficients.
            
            # This is easier to do by expanding the formulas.
            # Let's create a dictionary of coefficients for A and B for this specific product term.
            
            coeff_A = {}
            coeff_B = {}
            
            # Define the linear forms for L and R for the specific P_l
            # P1: (L00+L11) * (R00+R11)
            # P2: (L10+L11) * R00
            # P3: L00 * (R01-R11)
            # P4: L11 * (R10-R00)
            # P5: (L00+L01) * R11
            # P6: (L10-L00) * (R00+R01)
            # P7: (L01-L11) * (R10+R11)
            
            if l == 0: # P1
                l_terms = [(0,0,1), (1,1,1)]
                r_terms = [(0,0,1), (1,1,1)]
            elif l == 1: # P2
                l_terms = [(1,0,1), (1,1,1)]
                r_terms = [(0,0,1)]
            elif l == 2: # P3
                l_terms = [(0,0,1)]
                r_terms = [(0,1,1), (1,1,-1)]
            elif l == 3: # P4
                l_terms = [(1,1,1)]
                r_terms = [(1,0,1), (0,0,-1)]
            elif l == 4: # P5
                l_terms = [(0,0,1), (0,1,1)]
                r_terms = [(1,1,1)]
            elif l == 5: # P6
                l_terms = [(1,0,1), (0,0,-1)]
                r_terms = [(0,0,1), (0,1,1)]
            else: # P7
                l_terms = [(0,1,1), (1,1,-1)]
                r_terms = [(1,0,1), (1,1,1)]
                
            # Expand L terms
            # Each l_term is (i, j, coeff)
            # L[i,j] is defined by L_defs[k](i, j)
            for i, j, coeff_l in l_terms:
                for a_idx_val, coeff_a in L_defs[k](i, j):
                    coeff_A[a_idx_val] = coeff_A.get(a_idx_val, 0) + coeff_l * coeff_a
                    
            # Expand R terms
            for i, j, coeff_r in r_terms:
                for b_idx_val, coeff_b in R_defs[k](i, j):
                    coeff_B[b_idx_val] = coeff_B.get(b_idx_val, 0) + coeff_r * coeff_b
            
            # Fill U and V
            for idx, val in coeff_A.items():
                U[idx][c] = val
            for idx, val in coeff_B.items():
                V[idx][c] = val
            
            # Fill W
            # Contribution to C comes from M_k[i,j]
            # M_k[i,j] = T[i,j,l] * P_l
            # C[I*2+i, J*2+j] += S[I,J,k] * M_k[i,j]
            # S coefficients for 4x4 Strassen:
            # C00 = M1 + M4 - M5 + M7  -> S[0,0] = [1,0,0,1,-1,0,1]
            # C01 = M3 + M5            -> S[0,1] = [0,0,1,0,1,0,0]
            # C10 = M2 + M4            -> S[1,0] = [0,1,0,1,0,0,0]
            # C11 = M1 - M2 + M3 + M6  -> S[1,1] = [1,-1,1,0,0,1,0]
            
            S = np.array([
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ])
            
            for i in range(2):
                for j in range(2):
                    # T[i,j,l] is coefficient of P_l in M_k[i,j]
                    t_val = T[i, j, l]
                    if t_val == 0: continue
                    
                    # S[I,J,k] is coefficient of M_k in C[I,J]
                    # Note: k is 0-indexed in loop, S uses 0-indexed cols
                    for I in range(2):
                        for J in range(2):
                            s_val = S[I, J, k]
                            if s_val == 0: continue
                            
                            total_coeff = t_val * s_val
                            if total_coeff == 0: continue
                            
                            # Output C index: row = I*2+i, col = J*2+j
                            row = I * 2 + i
                            col = J * 2 + j
                            w_idx_val = w_idx(row, col)
                            
                            W[w_idx_val][c] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
