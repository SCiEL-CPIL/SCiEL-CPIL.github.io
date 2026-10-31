"""Program for 4x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4x4 matrix multiplication
    based on the Bini-Capovani-Lotti-Romani (BCLR) algorithm, which is derived
    from applying Strassen's 7-rank algorithm to 2x2 blocks.
    
    The decomposition uses the convention:
    - U has shape (n*m, rank) where n=4, m=4
    - V has shape (m*p, rank) where m=4, p=4
    - W has shape (p*n, rank) where p=4, n=4, with index k*n+i
    
    The tensor T[i,j,k] = delta_{i,k} for matrix multiplication A*B where
    (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    Using Strassen on 2x2 blocks:
    Partition 4x4 matrices into 2x2 blocks of 2x2 matrices.
    Let A = [[A11, A12], [A21, A22]], B = [[B11, B21], [B12, B22]]
    where each sub-block is 2x2.
    
    Strassen's algorithm for 2x2 block multiplication:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mx involves multiplying 2x2 matrices, which can be done with
    Strassen's 7-rank algorithm or schoolbook 4-rank.
    
    For rank minimality, we use Strassen for the 2x2 multiplications too.
    This gives 7 * 7 = 49 rank terms total.
    
    Let's define the indices carefully.
    
    For the 4x4 matrices, we index elements as:
    A[i,j] where i,j in {0,1,2,3}
    B[j,k] where j,k in {0,1,2,3}
    C[i,k] where i,k in {0,1,2,3}
    
    Block structure:
    Block (a,b) where a,b in {0,1} corresponds to rows 2a,2a+1 and cols 2b,2b+1
    
    A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    For each Strassen Mx computation, we need to compute a 2x2 matrix product.
    Using Strassen for the 2x2 case:
    For matrices P (2x2) and Q (2x2), the product R = P*Q has:
    R[0,0] = S1 + S4 - S5 + S7
    R[0,1] = S3 + S5
    R[1,0] = S2 + S4
    R[1,1] = S1 - S2 + S3 + S6
    
    where:
    S1 = (P[0,0]+P[1,1])(Q[0,0]+Q[1,1])
    S2 = (P[1,0]+P[1,1])Q[0,0]
    S3 = P[0,0](Q[0,1]-Q[1,1])
    S4 = P[1,1](Q[1,0]-Q[0,0])
    S5 = (P[0,0]+P[0,1])Q[1,1]
    S6 = (P[1,0]-P[0,0])(Q[0,0]+Q[0,1])
    S7 = (P[0,1]-P[1,1])(Q[1,0]+Q[1,1])
    
    So for each of the 7 block-level Strassen terms Mx, we have 7 scalar
    multiplication terms from the inner Strassen, giving 49 total.
    
    Let me build the U, V, W matrices explicitly.
    
    U[i*m+j, c] = coefficient for A[i,j] in term c
    V[j*p+k, c] = coefficient for B[j,k] in term c
    W[k*n+i, c] = coefficient for C[i,k] in term c (note: k*n+i convention)
    
    We need to track which elements of A and B go into each of the 49 terms.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen combinations at the block level
    # Each Mx is a product of two 2x2 block combinations
    # Mx = Px * Qx where Px is a 2x2 matrix (linear combo of A blocks)
    # and Qx is a 2x2 matrix (linear combo of B blocks)
    
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # Block element (a,b) in block (block_row, block_col) corresponds to:
    # A element (2*block_row + a, 2*block_col + b)
    
    # Define the 7 P matrices (linear combos of A blocks)
    # Each P is a 2x2 matrix where each entry is a linear combo of A block elements
    # P[pr, pc] = sum of coeff * A[block_row, block_col] for specific blocks
    
    # Let's define each Mx term as:
    # Mx = sum over (pr,pc) of Px[pr,pc] * Qx[pr,pc] ... no wait,
    # Mx = Px * Qx where Px, Qx are 2x2 matrices
    # So Mx[r,c] = sum_q Px[r,q] * Qx[q,c]
    
    # But we're using Strassen on the 2x2 multiplication, so:
    # Mx[r,c] is computed via 7 Strassen terms S1..S7
    
    # Let me define the 7 block-level combinations:
    # For M1: P = [[A11, 0],[0, A22]] + [[0,0],[0,0]]... 
    # Actually, let me be more careful.
    
    # M1 = (A11 + A22)(B11 + B22)
    # So P1 = A11 + A22 (as a 2x2 matrix), Q1 = B11 + B22
    # P1[0,0] = A11[0,0] + A22[0,0] = A[0,0] + A[2,2]
    # P1[0,1] = A11[0,1] + A22[0,1] = A[0,1] + A[2,3]
    # P1[1,0] = A11[1,0] + A22[1,0] = A[1,0] + A[3,2]
    # P1[1,1] = A11[1,1] + A22[1,1] = A[1,1] + A[3,3]
    
    # Similarly for Q1 with B elements.
    
    # Let me define all 7 P and Q matrices explicitly.
    
    # Helper: get A element index
    # A[i,j] -> U[i*m+j] = U[i*4+j]
    # B[j,k] -> V[j*p+k] = V[j*4+k]
    # C[i,k] -> W[k*n+i] = W[k*4+i]
    
    # Define the 7 P matrices (each is 2x2, entries are linear combos of A elements)
    # and 7 Q matrices (entries are linear combos of B elements)
    
    # I'll represent each P[r][c] as a dict: {A_index: coeff}
    # A_index = i*4 + j
    
    # M1: P = A11 + A22, Q = B11 + B22
    # M2: P = A21 + A22, Q = B11
    # M3: P = A11, Q = B12 - B22
    # M4: P = A22, Q = B21 - B11
    # M5: P = A11 + A12, Q = B22
    # M6: P = A21 - A11, Q = B11 + B12
    # M7: P = A12 - A22, Q = B21 + B22
    
    # For each Mx, P is a 2x2 matrix. Let me define P as a list of 4 entries
    # P[0]=P[0,0], P[1]=P[0,1], P[2]=P[1,0], P[3]=P[1,1]
    # Each entry is a dict {A_idx: coeff}
    
    def add_to_dict(d, key, val):
        if val != 0:
            d[key] = d.get(key, 0) + val
    
    # Define P matrices
    # A11: elements A[0,0],A[0,1],A[1,0],A[1,1] -> indices 0,1,4,5
    # A12: elements A[0,2],A[0,3],A[1,2],A[1,3] -> indices 2,3,6,7
    # A21: elements A[2,0],A[2,1],A[3,0],A[3,1] -> indices 8,9,12,13
    # A22: elements A[2,2],A[2,3],A[3,2],A[3,3] -> indices 10,11,14,15
    
    # P1 = A11 + A22
    P1 = [
        {0: 1, 10: 1},   # P[0,0] = A[0,0] + A[2,2]
        {1: 1, 11: 1},   # P[0,1] = A[0,1] + A[2,3]
        {4: 1, 14: 1},   # P[1,0] = A[1,0] + A[3,2]
        {5: 1, 15: 1},   # P[1,1] = A[1,1] + A[3,3]
    ]
    
    # P2 = A21 + A22
    P2 = [
        {8: 1, 10: 1},   # P[0,0] = A[2,0] + A[2,2]
        {9: 1, 11: 1},   # P[0,1] = A[2,1] + A[2,3]
        {12: 1, 14: 1},  # P[1,0] = A[3,0] + A[3,2]
        {13: 1, 15: 1},  # P[1,1] = A[3,1] + A[3,3]
    ]
    
    # P3 = A11
    P3 = [
        {0: 1},
        {1: 1},
        {4: 1},
        {5: 1},
    ]
    
    # P4 = A22
    P4 = [
        {10: 1},
        {11: 1},
        {14: 1},
        {15: 1},
    ]
    
    # P5 = A11 + A12
    P5 = [
        {0: 1, 2: 1},    # P[0,0] = A[0,0] + A[0,2]
        {1: 1, 3: 1},    # P[0,1] = A[0,1] + A[0,3]
        {4: 1, 6: 1},    # P[1,0] = A[1,0] + A[1,2]
        {5: 1, 7: 1},    # P[1,1] = A[1,1] + A[1,3]
    ]
    
    # P6 = A21 - A11
    P6 = [
        {8: 1, 0: -1},   # P[0,0] = A[2,0] - A[0,0]
        {9: 1, 1: -1},   # P[0,1] = A[2,1] - A[0,1]
        {12: 1, 4: -1},  # P[1,0] = A[3,0] - A[1,0]
        {13: 1, 5: -1},  # P[1,1] = A[3,1] - A[1,1]
    ]
    
    # P7 = A12 - A22
    P7 = [
        {2: 1, 10: -1},  # P[0,0] = A[0,2] - A[2,2]
        {3: 1, 11: -1},  # P[0,1] = A[0,3] - A[2,3]
        {6: 1, 14: -1},  # P[1,0] = A[1,2] - A[3,2]
        {7: 1, 15: -1},  # P[1,1] = A[1,3] - A[3,3]
    ]
    
    # Define Q matrices (linear combos of B elements)
    # B11: elements B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,4,5
    # B12: elements B[0,2],B[0,3],B[1,2],B[1,3] -> indices 2,3,6,7
    # B21: elements B[2,0],B[2,1],B[3,0],B[3,1] -> indices 8,9,12,13
    # B22: elements B[2,2],B[2,3],B[3,2],B[3,3] -> indices 10,11,14,15
    
    # Q1 = B11 + B22
    Q1 = [
        {0: 1, 10: 1},   # Q[0,0] = B[0,0] + B[2,2]
        {1: 1, 11: 1},   # Q[0,1] = B[0,1] + B[2,3]
        {4: 1, 14: 1},   # Q[1,0] = B[1,0] + B[3,2]
        {5: 1, 15: 1},   # Q[1,1] = B[1,1] + B[3,3]
    ]
    
    # Q2 = B11
    Q2 = [
        {0: 1},
        {1: 1},
        {4: 1},
        {5: 1},
    ]
    
    # Q3 = B12 - B22
    Q3 = [
        {2: 1, 10: -1},  # Q[0,0] = B[0,2] - B[2,2]
        {3: 1, 11: -1},  # Q[0,1] = B[0,3] - B[2,3]
        {6: 1, 14: -1},  # Q[1,0] = B[1,2] - B[3,2]
        {7: 1, 15: -1},  # Q[1,1] = B[1,3] - B[3,3]
    ]
    
    # Q4 = B21 - B11
    Q4 = [
        {8: 1, 0: -1},   # Q[0,0] = B[2,0] - B[0,0]
        {9: 1, 1: -1},   # Q[0,1] = B[2,1] - B[0,1]
        {12: 1, 4: -1},  # Q[1,0] = B[3,0] - B[1,0]
        {13: 1, 5: -1},  # Q[1,1] = B[3,1] - B[1,1]
    ]
    
    # Q5 = B22
    Q5 = [
        {10: 1},
        {11: 1},
        {14: 1},
        {15: 1},
    ]
    
    # Q6 = B11 + B12
    Q6 = [
        {0: 1, 2: 1},    # Q[0,0] = B[0,0] + B[0,2]
        {1: 1, 3: 1},    # Q[0,1] = B[0,1] + B[0,3]
        {4: 1, 6: 1},    # Q[1,0] = B[1,0] + B[1,2]
        {5: 1, 7: 1},    # Q[1,1] = B[1,1] + B[1,3]
    ]
    
    # Q7 = B21 + B22
    Q7 = [
        {8: 1, 10: 1},   # Q[0,0] = B[2,0] + B[2,2]
        {9: 1, 11: 1},   # Q[0,1] = B[2,1] + B[2,3]
        {12: 1, 14: 1},  # Q[1,0] = B[3,0] + B[3,2]
        {13: 1, 15: 1},  # Q[1,1] = B[3,1] + B[3,3]
    ]
    
    # Now for each Mx (x=1..7), we compute the 2x2 product using Strassen.
    # The 7 Strassen terms for multiplying P (2x2) by Q (2x2):
    # S1 = (P[0,0]+P[1,1]) * (Q[0,0]+Q[1,1])
    # S2 = (P[1,0]+P[1,1]) * Q[0,0]
    # S3 = P[0,0] * (Q[0,1]-Q[1,1])
    # S4 = P[1,1] * (Q[1,0]-Q[0,0])
    # S5 = (P[0,0]+P[0,1]) * Q[1,1]
    # S6 = (P[1,0]-P[0,0]) * (Q[0,0]+Q[0,1])
    # S7 = (P[0,1]-P[1,1]) * (Q[1,0]+Q[1,1])
    
    # Result:
    # R[0,0] = S1 + S4 - S5 + S7
    # R[0,1] = S3 + S5
    # R[1,0] = S2 + S4
    # R[1,1] = S1 - S2 + S3 + S6
    
    # For each Mx, the 7 Strassen terms give 7 rank-1 terms.
    # The contribution to C is determined by which block of C the Mx contributes to.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C11 is block (0,0): C[0:2, 0:2]
    # C12 is block (0,1): C[0:2, 2:4]
    # C21 is block (1,0): C[2:4, 0:2]
    # C22 is block (1,1): C[2:4, 2:4]
    
    # For each Mx contributing to a C block with coefficient coeff:
    # The 2x2 result R from Mx contributes coeff * R[r,c] to C block at position (r,c)
    
    # Let's build the 49 terms.
    
    # For Mx, Strassen term sy (y=1..7):
    # The P part of term sy is a linear combo of A elements
    # The Q part of term sy is a linear combo of B elements
    # The contribution to C is: for each (r,c) in the 2x2 result, 
    #   C_block[r,c] gets coeff * (contribution from sy to R[r,c])
    
    # Strassen term contributions to R:
    # S1: R[0,0]+=1, R[1,1]+=1
    # S2: R[1,0]+=1, R[1,1]-=1
    # S3: R[0,1]+=1, R[1,1]+=1
    # S4: R[0,0]+=1, R[1,0]+=1
    # S5: R[0,0]-=1, R[0,1]+=1
    # S6: R[1,1]+=1
    # S7: R[0,0]+=1
    
    # Define the Strassen P and Q linear combos for each term:
    # For a given P (2x2 dict entries) and Q (2x2 dict entries):
    
    def strassen_terms(P, Q):
        """
        Return list of 7 tuples (P_dict, Q_dict) for the 7 Strassen terms.
        P_dict and Q_dict are {index: coeff} dicts.
        """
        # S1: (P[0,0]+P[1,1]) * (Q[0,0]+Q[1,1])
        P1 = {}
        add_to_dict(P1, 0, 1)  # P[0,0]
        add_to_dict(P1, 3, 1)  # P[1,1]
        Q1 = {}
        add_to_dict(Q1, 0, 1)  # Q[0,0]
        add_to_dict(Q1, 3, 1)  # Q[1,1]
        
        # S2: (P[1,0]+P[1,1]) * Q[0,0]
        P2 = {}
        add_to_dict(P2, 2, 1)  # P[1,0]
        add_to_dict(P2, 3, 1)  # P[1,1]
        Q2 = {}
        add_to_dict(Q2, 0, 1)  # Q[0,0]
        
        # S3: P[0,0] * (Q[0,1]-Q[1,1])
        P3 = {}
        add_to_dict(P3, 0, 1)  # P[0,0]
        Q3 = {}
        add_to_dict(Q3, 1, 1)   # Q[0,1]
        add_to_dict(Q3, 3, -1)  # Q[1,1]
        
        # S4: P[1,1] * (Q[1,0]-Q[0,0])
        P4 = {}
        add_to_dict(P4, 3, 1)  # P[1,1]
        Q4 = {}
        add_to_dict(Q4, 2, 1)   # Q[1,0]
        add_to_dict(Q4, 0, -1)  # Q[0,0]
        
        # S5: (P[0,0]+P[0,1]) * Q[1,1]
        P5 = {}
        add_to_dict(P5, 0, 1)  # P[0,0]
        add_to_dict(P5, 1, 1)  # P[0,1]
        Q5 = {}
        add_to_dict(Q5, 3, 1)  # Q[1,1]
        
        # S6: (P[1,0]-P[0,0]) * (Q[0,0]+Q[0,1])
        P6 = {}
        add_to_dict(P6, 2, 1)   # P[1,0]
        add_to_dict(P6, 0, -1)  # P[0,0]
        Q6 = {}
        add_to_dict(Q6, 0, 1)  # Q[0,0]
        add_to_dict(Q6, 1, 1)  # Q[0,1]
        
        # S7: (P[0,1]-P[1,1]) * (Q[1,0]+Q[1,1])
        P7 = {}
        add_to_dict(P7, 1, 1)   # P[0,1]
        add_to_dict(P7, 3, -1)  # P[1,1]
        Q7 = {}
        add_to_dict(Q7, 2, 1)  # Q[1,0]
        add_to_dict(Q7, 3, 1)  # Q[1,1]
        
        return [(P1, Q1), (P2, Q2), (P3, Q3), (P4, Q4), (P5, Q5), (P6, Q6), (P7, Q7)]
    
    # Now we need to combine the block-level P and Q with the Strassen-level P and Q.
    # The block-level P is a 2x2 matrix where each entry is a dict of A elements.
    # The Strassen-level P selects which entries of the block-level P to combine.
    
    def combine_block_and_strassen(block_P, strassen_P_idx_coeffs):
        """
        block_P: list of 4 dicts, block_P[0]=P[0,0], block_P[1]=P[0,1], block_P[2]=P[1,0], block_P[3]=P[1,1]
        strassen_P_idx_coeffs: dict {strassen_idx: coeff} where strassen_idx in 0..3
        Returns a dict of A element indices to coefficients.
        """
        result = {}
        for idx, coeff in strassen_P_idx_coeffs.items():
            for a_idx, a_coeff in block_P[idx].items():
                total = coeff * a_coeff
                if total != 0:
                    result[a_idx] = result.get(a_idx, 0) + total
        return result
    
    # Strassen term P index selections:
    # S1: P[0,0] and P[1,1] -> {0:1, 3:1}
    # S2: P[1,0] and P[1,1] -> {2:1, 3:1}
    # S3: P[0,0] -> {0:1}
    # S4: P[1,1] -> {3:1}
    # S5: P[0,0] and P[0,1] -> {0:1, 1:1}
    # S6: P[1,0] and -P[0,0] -> {2:1, 0:-1}
    # S7: P[0,1] and -P[1,1] -> {1:1, 3:-1}
    
    strassen_P_selections = [
        {0: 1, 3: 1},   # S1
        {2: 1, 3: 1},   # S2
        {0: 1},         # S3
        {3: 1},         # S4
        {0: 1, 1: 1},   # S5
        {2: 1, 0: -1},  # S6
        {1: 1, 3: -1},  # S7
    ]
    
    # Strassen term Q index selections:
    # S1: Q[0,0] and Q[1,1] -> {0:1, 3:1}
    # S2: Q[0,0] -> {0:1}
    # S3: Q[0,1] and -Q[1,1] -> {1:1, 3:-1}
    # S4: Q[1,0] and -Q[0,0] -> {2:1, 0:-1}
    # S5: Q[1,1] -> {3:1}
    # S6: Q[0,0] and Q[0,1] -> {0:1, 1:1}
    # S7: Q[1,0] and Q[1,1] -> {2:1, 3:1}
    
    strassen_Q_selections = [
        {0: 1, 3: 1},   # S1
        {0: 1},         # S2
        {1: 1, 3: -1},  # S3
        {2: 1, 0: -1},  # S4
        {3: 1},         # S5
        {0: 1, 1: 1},   # S6
        {2: 1, 3: 1},   # S7
    ]
    
    # Strassen term contributions to R (2x2 result):
    # R indices: R[0,0]=0, R[0,1]=1, R[1,0]=2, R[1,1]=3
    strassen_R_contributions = [
        {0: 1, 3: 1},    # S1: R[0,0]+=1, R[1,1]+=1
        {2: 1, 3: -1},   # S2: R[1,0]+=1, R[1,1]-=1
        {1: 1, 3: 1},    # S3: R[0,1]+=1, R[1,1]+=1
        {0: 1, 2: 1},    # S4: R[0,0]+=1, R[1,0]+=1
        {0: -1, 1: 1},   # S5: R[0,0]-=1, R[0,1]+=1
        {3: 1},          # S6: R[1,1]+=1
        {0: 1},          # S7: R[0,0]+=1
    ]
    
    # Block-level P and Q matrices
    P_blocks = [P1, P2, P3, P4, P5, P6, P7]
    Q_blocks = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    
    # C block contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C block (cr, cc) where cr,cc in {0,1}
    # C block (cr,cc) corresponds to C elements C[2*cr+r, 2*cc+c] for r,c in {0,1}
    # W index: k*n+i = (2*cc+c)*4 + (2*cr+r)
    
    c_block_coeffs = {
        (0, 0): {1: 1, 4: 1, 5: -1, 7: 1},   # C11 = M1 + M4 - M5 + M7
        (0, 1): {3: 1, 5: 1},                 # C12 = M3 + M5
        (1, 0): {2: 1, 4: 1},                 # C21 = M2 + M4
        (1, 1): {1: 1, 2: -1, 3: 1, 6: 1},   # C22 = M1 - M2 + M3 + M6
    }
    
    # Now build the 49 terms
    term_idx = 0
    for mx in range(1, 8):  # M1 to M7
        P_block = P_blocks[mx - 1]
        Q_block = Q_blocks[mx - 1]
        
        for sy in range(7):  # Strassen terms S1 to S7
            # Get the combined A and B linear combos
            P_sel = strassen_P_selections[sy]
            Q_sel = strassen_Q_selections[sy]
            
            # Combine block P with strassen P selection
            A_dict = combine_block_and_strassen(P_block, P_sel)
            B_dict = combine_block_and_strassen(Q_block, Q_sel)
            
            # Get the R contributions
            R_contrib = strassen_R_contributions[sy]
            
            # For each C block that Mx contributes to:
            for (cr, cc), m_coeffs in c_block_coeffs.items():
                if mx in m_coeffs:
                    block_coeff = m_coeffs[mx]
                    
                    # For each R element that this Strassen term contributes to:
                    for r_idx, r_coeff in R_contrib.items():
                        # r_idx: 0=R[0,0], 1=R[0,1], 2=R[1,0], 3=R[1,1]
                        r_row = r_idx // 2
                        r_col = r_idx % 2
                        
                        # C element: C[2*cr + r_row, 2*cc + r_col]
                        c_i = 2 * cr + r_row
                        c_k = 2 * cc + r_col
                        
                        # W index: k*n + i = c_k * 4 + c_i
                        w_idx = c_k * n + c_i
                        
                        total_coeff = block_coeff * r_coeff
                        
                        if total_coeff != 0:
                            # Add to U, V, W for this term
                            for a_idx, a_coeff in A_dict.items():
                                if a_coeff != 0:
                                    U[a_idx][term_idx] += a_coeff
                            for b_idx, b_coeff in B_dict.items():
                                if b_coeff != 0:
                                    V[b_idx][term_idx] += b_coeff
                            W[w_idx][term_idx] += total_coeff
            
            term_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
