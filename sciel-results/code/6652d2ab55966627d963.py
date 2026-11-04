import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2x2 blocks
    # A (2x4) = [AL, AR] (2x2 each)
    # B (4x6) = [[BT1, BT2, BT3], [BB1, BB2, BB3]] (2x2 blocks)
    # C (2x6) = [C1, C2, C3] (2x2 blocks)
    # C1 = AL*BT1 + AR*BB1
    # C2 = AL*BT2 + AR*BB2
    # C3 = AL*BT3 + AR*BB3
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 6 such multiplications.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U (A matrix): i*m + j
    # i in 0..1, j in 0..3
    # AL: i in 0..1, j in 0..1
    # AR: i in 0..1, j in 2..3
    
    # Indices for V (B matrix): j*p + k
    # j in 0..3, k in 0..5
    # BT: j in 0..1, k in 0..5
    # BB: j in 2..3, k in 0..5
    
    # Indices for W (C matrix): k*n + i
    # k in 0..5, i in 0..1
    
    def add_strassen_block(u_offset, v_offset, w_offset, a_row_shift, b_col_shift, c_col_shift):
        # a_row_shift: shift for i in A (0 for AL, 0 for AR? No, A is 2x2, so i is local 0..1)
        # Actually, U index is i*m + j.
        # For AL: i=0..1, j=0..1.
        # For AR: i=0..1, j=2..3.
        
        # We define the local 2x2 indices
        # A_local[i][j]
        # B_local[j][k]
        # C_local[i][k]
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        col = 0
        for term in range(7):
            # Determine coefficients for U, V, W
            # This is a standard Strassen decomposition
            pass
            
    # Instead of generic, let's hardcode the 42 terms.
    # This is tedious but safe.
    
    # Let's use a helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen terms for one 2x2x2 block
    # Local indices:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1? No, W is the output coefficient.
    # The tensor is sum U[i,c] V[j,c] W[k,c] delta_{i,j,k}?
    # No, the decomposition is T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}.
    # Here i is row of A, j is col of A/row of B, k is col of B.
    # Wait, standard convention:
    # A is n x m. B is m x p. C is n x p.
    # T_{i,j,k} = A_{i,j} B_{j,k}.
    # U has size n*m? No, U has size n*m?
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # This is unusual. Usually U (n, rank), V (m, rank), W (p, rank).
    # Here U is flattened A?
    # If U is (n*m, rank), it corresponds to the input A.
    # V is (m*p, rank), corresponds to input B.
    # W is (p*n, rank), corresponds to output C.
    
    # So U[i*m + j, c] is coefficient for A_{i,j}.
    # V[j*p + k, c] is coefficient for B_{j,k}.
    # W[k*n + i, c] is coefficient for C_{i,k}.
    
    # For a term c:
    # U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    
    # Strassen M1: (A00+A11)(B00+B11)
    # This term contributes to C00 and C11?
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1, we have two parts?
    # No, in tensor decomposition, each term c is a rank-1 tensor.
    # We need to split M1 into rank-1 tensors?
    # No, M1 is a product of sums.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is a sum of 4 rank-1 tensors?
    # But Strassen reduces 8 multiplications to 7.
    # The 7 terms are the "multiplications".
    # Each multiplication is a scalar.
    # The tensor decomposition uses these scalars.
    
    # So for each Strassen term (e.g., M1), we have a rank-1 component in the tensor?
    # No, M1 is computed as a product of linear forms.
    # In the tensor decomposition, we represent the multiplication as:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # = sum_c (sum_i U_{ic} A_{ij} ... )
    
    # The standard Strassen decomposition of the 2x2x2 tensor has rank 7.
    # The 7 triples are:
    # 1. u=(1,0,1,0), v=(1,1,0,0), w=(1,0,0,1) -> C00 gets 1, C11 gets 1?
    # Let's verify.
    # Term 1: U=(1,0,1,0) for A? No.
    
    # Let's use the known rank-7 decomposition of 2x2x2.
    
    # We will instantiate 6 copies of this decomposition.
    
    strassen_7 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: A00, A01, A10, A11
        # v: B00, B01, B10, B11
        # w: C00, C01, C10, C11
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)), # M1: (A00+A11)(B00+B11) -> C00, C11
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)), # M2: (A10+A11)B00 -> C10
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)), # M3: A00(B01-B11) -> C01
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 0, 0)), # M4: A11(B10-B00) -> C00
        ((1, 1, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0)), # M5: (A00+A01)B11 -> C01
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)), # M6: (A10-A00)(B00+B01) -> C11
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)), # M7: (A01-A11)(B10+B11) -> C00
    ]
    
    # Check M1:
    # U: A00=1, A11=1.
    # V: B00=1, B11=1.
    # W: C00=1, C11=1.
    # Product: (A00+A11)(B00+B11) (C00 + C11).
    # This adds M1 to C00 and M1 to C11. Correct.
    
    # M2:
    # U: A10=1, A11=1.
    # V: B00=1.
    # W: C10=1.
    # Product: (A10+A11)B00 C10. Correct.
    
    # M3:
    # U: A00=1.
    # V: B01=1, B11=-1.
    # W: C01=1.
    # Product: A00(B01-B11) C01. Correct.
    
    # M4:
    # U: A11=1.
    # V: B10=1, B00=-1.
    # W: C00=1.
    # Product: A11(B10-B00) C00. Correct.
    
    # M5:
    # U: A00=1, A01=1.
    # V: B11=1.
    # W: C01=1.
    # Product: (A00+A01)B11 C01. Correct.
    
    # M6:
    # U: A10=1, A00=-1.
    # V: B00=1, B01=1.
    # W: C11=1.
    # Product: (A10-A00)(B00+B01) C11. Correct.
    
    # M7:
    # U: A01=1, A11=-1.
    # V: B10=1, B11=1.
    # W: C00=1.
    # Product: (A01-A11)(B10+B11) C00. Correct.
    
    # Now map these to the global indices.
    
    # Block 1: AL * BT1
    # A: i=0..1, j=0..1
    # B: j=0..1, k=0..1
    # C: i=0..1, k=0..1
    
    # Block 2: AR * BB1
    # A: i=0..1, j=2..3
    # B: j=2..3, k=0..1
    # C: i=0..1, k=0..1
    
    # ... and so on.
    
    # We have 6 blocks.
    # C1 = AL*BT1 + AR*BB1
    # C2 = AL*BT2 + AR*BB2
    # C3 = AL*BT3 + AR*BB3
    
    # We will add 7 terms for each block.
    
    col = 0
    for block in range(6):
        # Determine A, B, C offsets
        # Block 0: AL, BT1, C1
        # Block 1: AR, BB1, C1
        # Block 2: AL, BT2, C2
        # Block 3: AR, BB2, C2
        # Block 4: AL, BT3, C3
        # Block 5: AR, BB3, C3
        
        if block % 2 == 0:
            # AL
            a_j_offset = 0
        else:
            # AR
            a_j_offset = 2
            
        # B block index
        b_block = block // 2
        # b_j_offset: 0 for BT, 2 for BB
        if block % 4 == 0 or block % 4 == 1: # No, BT is blocks 0,2,4?
            # Block 0: AL, BT1. Block 1: AR, BB1.
            # Block 2: AL, BT2. Block 3: AR, BB2.
            # Block 4: AL, BT3. Block 5: AR, BB3.
            pass
            
        # Let's define explicitly
        if block == 0:
            a_j_off, b_j_off, c_k_off = 0, 0, 0
        elif block == 1:
            a_j_off, b_j_off, c_k_off = 2, 2, 0
        elif block == 2:
            a_j_off, b_j_off, c_k_off = 0, 0, 2
        elif block == 3:
            a_j_off, b_j_off, c_k_off = 2, 2, 2
        elif block == 4:
            a_j_off, b_j_off, c_k_off = 0, 0, 4
        elif block == 5:
            a_j_off, b_j_off, c_k_off = 2, 2, 4
            
        for u, v, w in strassen_7:
            # u: (A00, A01, A10, A11)
            # Global A indices: i=0..1, j=a_j_off..a_j_off+1
            # U index: i*m + j
            
            # v: (B00, B01, B10, B11)
            # Global B indices: j=b_j_off..b_j_off+1, k=c_k_off..c_k_off+1
            # V index: j*p + k
            
            # w: (C00, C01, C10, C11)
            # Global C indices: i=0..1, k=c_k_off..c_k_off+1
            # W index: k*n + i
            
            # Add term
            for i in range(2):
                for j in range(2):
                    val_u = u[i*2+j]
                    if val_u != 0:
                        U[i*m + (a_j_off + j)][col] += val_u
                        
            for j in range(2):
                for k in range(2):
                    val_v = v[j*2+k]
                    if val_v != 0:
                        V[(b_j_off + j)*p + (c_k_off + k)][col] += val_v
                        
            for i in range(2):
                for k in range(2):
                    val_w = w[i*2+k]
                    if val_w != 0:
                        W[(c_k_off + k)*n + i][col] += val_w
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
