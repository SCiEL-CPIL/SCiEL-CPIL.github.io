"""Matrix multiplication 4x4x5 - rank minimization via Strassen decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

def verify_decomposition(U, V, W):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 exactly."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[i*m+j])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1.0:
                    return False
    return True

def construct_2x2x2_strassen():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3]
    # C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11

    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11

    # Term 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11

    # Term 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10

    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01

    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11

    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 65 decomposition: 49 (for 4x4x4) + 16 (for 4x4x1)
    
    U_2, V_2, W_2 = construct_2x2x2_strassen()
    
    # Build 4x4x4 part
    U_4 = []
    V_4 = []
    W_4 = []
    
    # Outer Strassen terms definitions
    # Each term is (A_blocks, B_blocks)
    # A_blocks: dict {block_idx: coeff}
    # B_blocks: dict {block_idx: coeff}
    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),      # M1: (A00+A11)(B00+B11)
        ({2: 1, 3: 1}, {0: 1}),             # M2: (A10+A11)B00
        ({0: 1}, {1: 1, 3: -1}),            # M3: A00(B01-B11)
        ({3: 1}, {2: 1, 0: -1}),            # M4: A11(B10-B00)
        ({0: 1, 1: 1}, {3: 1}),             # M5: (A00+A01)B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6: (A10-A00)(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1})       # M7: (A01-A11)(B10+B11)
    ]
    
    # Block indices for 4x4:
    # 0: 0,0 (rows 0-1, cols 0-1)
    # 1: 0,1 (rows 0-1, cols 2-3)
    # 2: 1,0 (rows 2-3, cols 0-1)
    # 3: 1,1 (rows 2-3, cols 2-3)
    
    def get_block_indices(block_idx):
        """Returns list of (i, j) for the 2x2 block."""
        r0 = block_idx // 2
        c0 = block_idx % 2
        return [(2*r0 + i, 2*c0 + j) for i in range(2) for j in range(2)]
    
    for r_o, (a_coeffs, b_coeffs) in enumerate(outer_terms):
        # Construct A_co and B_co as 2x2 matrices of coefficients
        # A_co[i0, j0] = sum_{block} coeff * A[block_row + i0, block_col + j0]
        # We need to express A_co entries in terms of global A indices.
        # A_co is 2x2.
        # Let's build the contribution of this outer term to the global U, V, W.
        
        # For each inner Strassen term r_i (0..6):
        for r_i in range(7):
            # Get inner coefficients
            # U_2[i0*2+j0, r_i]
            # V_2[j0*2+k0, r_i]
            # W_2[k0*2+i0, r_i]
            
            # The term is:
            # sum_{i0, j0, k0} W_2[k0*2+i0, r_i] * (A_co[i0, j0] * B_co[j0, k0]) * U_2[i0*2+j0, r_i] * V_2[j0*2+k0, r_i]
            
            # We need to distribute this into global U, V, W vectors.
            # Global U index: i*4+j
            # Global V index: j*5+k (Note: V is for 4x5, so j in 0..3, k in 0..4)
            # Global W index: k*4+i
            
            # Wait, the 4x4x4 part uses k in 0..3.
            # The 4x4x1 part uses k=4.
            
            # For the 4x4x4 part:
            # We add a new column to U_4, V_4, W_4.
            
            new_U = [0.0] * 16
            new_V = [0.0] * 20  # 4x5, but we only fill 4x4 part for now? No, V is 4x5.
            new_W = [0.0] * 20  # 5x4
            
            # Iterate over i0, j0, k0
            for i0 in range(2):
                for j0 in range(2):
                    for k0 in range(2):
                        w_coeff = W_2[k0*2+i0][r_i]
                        u_coeff = U_2[i0*2+j0][r_i]
                        v_coeff = V_2[j0*2+k0][r_i]
                        
                        if w_coeff == 0 or u_coeff == 0 or v_coeff == 0:
                            continue
                        
                        # A_co[i0, j0] is a linear combo of A entries.
                        # A_co[i0, j0] = sum_{ba} a_coeffs[ba] * A[2*(ba//2) + i0, 2*(ba%2) + j0]
                        # B_co[j0, k0] = sum_{bb} b_coeffs[bb] * B[2*(bb//2) + j0, 2*(bb%2) + k0]
                        
                        # The product A_co[i0, j0] * B_co[j0, k0] is a sum of products of A and B entries.
                        # We need to add a rank-1 term for EACH pair (ba, bb).
                        # This means we might need to split this inner term into multiple global rank-1 terms if the coefficients don't factorize?
                        # No, the decomposition is linear.
                        # Term: w_coeff * u_coeff * v_coeff * (sum_{ba} a[ba] A[...]) * (sum_{bb} b[bb] B[...])
                        # This is a sum of terms: w*u*v*a[ba]*b[bb] * A[...] * B[...]
                        # Each such term is a rank-1 tensor.
                        # So we must add a new column for EACH (ba, bb) pair.
                        
                        for ba, a_val in a_coeffs.items():
                            for bb, b_val in b_coeffs.items():
                                if a_val == 0 or b_val == 0:
                                    continue
                                
                                # Global indices
                                i_global = 2*(ba//2) + i0
                                j_global = 2*(ba%2) + j0
                                # Note: j_global must match for V.
                                # In B, j index is 2*(bb//2) + j0.
                                # For the term to be valid, the j index in A and B must be the same?
                                # Yes, matrix multiplication sums over j.
                                # Here we are decomposing the tensor.
                                # The tensor entry is T[i,j,k].
                                # In the product A[i,j] B[j,k], j is fixed for the term.
                                # In our expansion, j_global from A must equal j_global from B.
                                # j_A = 2*(ba%2) + j0
                                # j_B = 2*(bb%2) + j0
                                # If ba%2 != bb%2, then j_A != j_B.
                                # But in the tensor decomposition, we have separate U and V indices.
                                # U[i*m+j, r] and V[j*p+k, r].
                                # They share the index j.
                                # So we can only add a term if j_A == j_B.
                                # If j_A != j_B, this combination does not contribute to a single rank-1 term in the standard decomposition?
                                # Actually, the Strassen algorithm works because the intermediate products are valid matrix multiplications.
                                # The "j" in the outer block product is the block column of A and block row of B.
                                # Block ba has column index ba%2. Block bb has row index bb//2.
                                # For the product A_block * B_block to be non-zero, the column of A block must match the row of B block.
                                # i.e., ba%2 == bb//2.
                                # If ba%2 != bb//2, the term is 0.
                                
                                if (ba % 2) != (bb // 2):
                                    continue
                                    
                                # Check j consistency
                                # j_A = 2*(ba%2) + j0
                                # j_B = 2*(bb%2) + j0  <- Wait, B index is j*p+k.
                                # B row is j. B block bb is at row 2*(bb//2).
                                # So B row index is 2*(bb//2) + j0.
                                # A col index is 2*(ba%2) + j0.
                                # They match if ba%2 == bb//2.
                                
                                k_global = 2*(bb%2) + k0
                                
                                # Coefficients
                                coeff = w_coeff * u_coeff * v_coeff * a_val * b_val
                                
                                if coeff != 0:
                                    # Add to new_U, new_V, new_W
                                    # But we are adding multiple (ba,bb) pairs to the SAME inner term r_i?
                                    # No, each (ba,bb) pair generates a rank-1 term.
                                    # So we should create a new column for each (ba,bb) pair.
                                    
                                    # We cannot just accumulate into new_U/V/W for this r_i if we want minimal rank.
                                    # Each independent product A[i,j]B[j,k] is a rank-1 term.
                                    # The Strassen decomposition groups them.
                                    # If we split them, we lose the Strassen benefit.
                                    # The Strassen benefit comes from sharing the multiplications.
                                    # In the tensor decomposition, a "term" is a rank-1 tensor U_r \otimes V_r \otimes W_r.
                                    # If U_r has non-zero entries in multiple (i,j) pairs, it's still one term.
                                    
                                    # So we CAN accumulate into new_U, new_V, new_W for this r_o, r_i.
                                    # The resulting vector will have non-zero entries at multiple (i,j) and (j,k) positions.
                                    # This is valid.
                                    
                                    new_U[i_global*4 + j_global] += coeff
                                    new_V[j_global*5 + k_global] += coeff # Note: V is 4x5, so index is j*5+k. But we are in 4x4x4 part, k<4.
                                    new_W[k_global*4 + i_global] += coeff # W is 5x4, index is k*4+i.
                                    
            # Add the constructed term to the lists
            U_4.append(new_U)
            V_4.append(new_V)
            W_4.append(new_W)
            
    # Now we have 49 terms for 4x4x4.
    # Add 16 terms for 4x4x1.
    # C[i, 4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            new_U = [0.0] * 16
            new_V = [0.0] * 20
            new_W = [0.0] * 20
            
            new_U[i*4 + j] = 1.0
            new_V[j*5 + 4] = 1.0
            new_W[4*4 + i] = 1.0
            
            U_4.append(new_U)
            V_4.append(new_V)
            W_4.append(new_W)
            
    # Convert to lists of lists (integers/half-integers)
    # The coefficients are integers.
    U_final = [[int(x) for x in row] for row in U_4]
    V_final = [[int(x) for x in row] for row in V_4]
    W_final = [[int(x) for x in row] for row in W_4]
    
    # Verify
    if not verify_decomposition(U_final, V_final, W_final):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sch = [[0] * rank_schoolbook for _ in range(n * m)]
        V_sch = [[0] * rank_schoolbook for _ in range(m * p)]
        W_sch = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j][col] = 1
                    V_sch[j*p+k][col] = 1
                    W_sch[k*n+i][col] = 1
                    col += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}
        
    return {"U": U_final, "V": V_final, "W": W_final}

construct()
