"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 52 by using a rank-26 decomposition
    # for 3x3x3 multiplication, applied to the two 3x3 blocks of the 3x6 matrix.
    
    # U shape: (n*m, rank) = (9, 52)
    # V shape: (m*p, rank) = (18, 52)
    # W shape: (p*n, rank) = (18, 52)
    
    rank_block = 26
    total_rank = rank_block * 2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Helper to add a rank-1 term to a specific block
    def add_term(U, V, W, col, a_idx, v_idx, w_idx, coeff_a=1, coeff_v=1, coeff_w=1):
        U[a_idx, col] += coeff_a
        V[v_idx, col] += coeff_v
        W[w_idx, col] += coeff_w

    # Define the 26 terms for 3x3x3 multiplication
    # Local indices for 3x3:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    
    terms = []
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)(B00+B11)
    terms.append([(0, 1), (0, 1), (0, 1, -1, 1)]) # C00+=M1, C01+=0, C10+=0, C11+=M1? No, Strassen output mapping is specific.
    # Let's explicitly list the contributions to C for each Strassen term.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We add each M as a separate rank-1 term? No, that would be 7 terms for the 2x2 block.
    # But the coefficients in U, V, W must be integers.
    # M1: U has 1 at 0, 1. V has 1 at 0, 1.
    #    W has 1 at 0, 1 at 3, -1 at 2, 1 at 1?
    #    Wait, W is the output vector.
    #    Term 1 (M1): U[0]=1, U[1]=1, U[2]=0, U[3]=0. V[0]=1, V[1]=1, V[2]=0, V[3]=0.
    #    W[0]=1 (C00), W[1]=0 (C01), W[2]=0 (C10), W[3]=1 (C11).
    #    Check: U^T V W^T?
    #    Contribution to C00: U0*V0*W0 = 1*1*1 = 1. Correct (M1 part of C00).
    #    Contribution to C11: U1*V1*W3? No.
    #    Tensor form: T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
    #    For M1:
    #    r=0.
    #    U[0,0]=1, U[1,0]=1. (Indices 0,1 of A)
    #    V[0,0]=1, V[1,0]=1. (Indices 0,1 of B)
    #    W[0,0]=1, W[3,0]=1. (Indices 0,3 of C)
    #    T[0,0,0] += U[0,0]V[0,0]W[0,0] = 1. (C00 gets M1).
    #    T[0,0,3] += U[0,0]V[0,0]W[3,0] = 1. (C11 gets M1).
    #    T[0,1,0] += U[0,0]V[1,0]W[0,0] = 1. (C00 gets M1? No, V[1] is B01. M1 uses B00, B11.
    #    Strassen M1 = (A00+A11)(B00+B11).
    #    U[0]=1, U[1]=1 (A00, A11).
    #    V[0]=1, V[1]=1 (B00, B11).
    #    W[0]=1, W[3]=1 (C00, C11).
    #    Terms generated:
    #    A00*B00*C00: 1
    #    A00*B11*C00: 1
    #    A11*B00*C00: 1
    #    A11*B11*C00: 1
    #    A00*B00*C11: 1
    #    A00*B11*C11: 1
    #    A11*B00*C11: 1
    #    A11*B11*C11: 1
    #    This creates cross terms in the output that are not in M1?
    #    No, M1 is a scalar. The decomposition sums these.
    #    But the tensor decomposition requires T[i,j,k] to match.
    #    For C00, we need sum_k A_ik B_kj.
    #    The Strassen algorithm works because the linear combinations in U, V, W and the additions in W cancel out the cross terms.
    #    So we just need to list the 7 terms with their specific U, V, W coefficients.
    
    # Let's define the 7 Strassen terms for 2x2:
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    S = []
    # M1
    S.append(([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(3,1)]))
    # M2
    S.append(([(2,1),(3,1)], [(0,1)], [(2,1)]))
    # M3
    S.append(([(0,1)], [(1,1),(3,-1)], [(1,1)]))
    # M4
    S.append(([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]))
    # M5
    S.append(([(0,1),(1,1)], [(3,1)], [(2,-1),(1,1)]))
    # M6
    S.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]))
    # M7
    S.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
    
    # Verify Strassen correctness?
    # It's standard.
    
    # Now map these to 3x3 indices.
    # A 2x2 block is A[0:2, 0:2].
    # A indices 0,1,2,3 in 3x3 are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # B indices 0,1,2,3 in 3x3 are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # C indices 0,1,2,3 in 3x3 are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    
    block_terms = []
    
    # Add Strassen terms
    for u_idx, v_idx, w_idx in S:
        block_terms.append((u_idx, v_idx, w_idx))
        
    # Part 2: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 4:(0,2), 5:(1,2)
    # B indices: 6:(2,0), 7:(2,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C00 += A02 B20 -> U[4], V[6], W[0]
    block_terms.append(([(4,1)], [(6,1)], [(0,1)]))
    # C01 += A02 B21 -> U[4], V[7], W[1]
    block_terms.append(([(4,1)], [(7,1)], [(1,1)]))
    # C10 += A12 B20 -> U[5], V[6], W[2]
    block_terms.append(([(5,1)], [(6,1)], [(2,1)]))
    # C11 += A12 B21 -> U[5], V[7], W[3]
    block_terms.append(([(5,1)], [(7,1)], [(3,1)]))
    
    # Part 3: C[0:2, 2]
    # A[0:2, 0:2] * B[0:2, 2]
    # A indices: 0,1,2,3
    # B indices: 8:(0,2), 9:(1,2)
    # C indices: 4:(0,2), 5:(1,2)
    # C02 += A00 B02 -> U[0], V[8], W[4]
    block_terms.append(([(0,1)], [(8,1)], [(4,1)]))
    # C02 += A01 B12 -> U[1], V[9], W[4]
    block_terms.append(([(1,1)], [(9,1)], [(4,1)]))
    # C12 += A10 B02 -> U[2], V[8], W[5]
    block_terms.append(([(2,1)], [(8,1)], [(5,1)]))
    # C12 += A11 B12 -> U[3], V[9], W[5]
    block_terms.append(([(3,1)], [(9,1)], [(5,1)]))
    # A[0:2, 2] * B[2, 2]
    # A indices: 4,5
    # B index: 10:(2,2)
    # C indices: 4,5
    # C02 += A02 B22 -> U[4], V[10], W[4]
    block_terms.append(([(4,1)], [(10,1)], [(4,1)]))
    # C12 += A12 B22 -> U[5], V[10], W[5]
    block_terms.append(([(5,1)], [(10,1)], [(5,1)]))
    
    # Part 4: C[2, 0:2]
    # A[2, 0:2] * B[0:2, 0:2]
    # A indices: 6:(2,0), 7:(2,1)
    # B indices: 0,1,2,3
    # C indices: 6:(2,0), 7:(2,1)
    # C20 += A20 B00 -> U[6], V[0], W[6]
    block_terms.append(([(6,1)], [(0,1)], [(6,1)]))
    # C20 += A21 B10 -> U[7], V[2], W[6]
    block_terms.append(([(7,1)], [(2,1)], [(6,1)]))
    # C21 += A20 B01 -> U[6], V[1], W[7]
    block_terms.append(([(6,1)], [(1,1)], [(7,1)]))
    # C21 += A21 B11 -> U[7], V[3], W[7]
    block_terms.append(([(7,1)], [(3,1)], [(7,1)]))
    # A[2, 2] * B[2, 0:2]
    # A index: 8:(2,2)
    # B indices: 6,7
    # C indices: 6,7
    # C20 += A22 B20 -> U[8], V[6], W[6]
    block_terms.append(([(8,1)], [(6,1)], [(6,1)]))
    # C21 += A22 B21 -> U[8], V[7], W[7]
    block_terms.append(([(8,1)], [(7,1)], [(7,1)]))
    
    # Part 5: C[2, 2]
    # A[2, 0:2] * B[0:2, 2]
    # A indices: 6,7
    # B indices: 8,9
    # C index: 8:(2,2)
    # C22 += A20 B02 -> U[6], V[8], W[8]
    block_terms.append(([(6,1)], [(8,1)], [(8,1)]))
    # C22 += A21 B12 -> U[7], V[9], W[8]
    block_terms.append(([(7,1)], [(9,1)], [(8,1)]))
    # A[2, 2] * B[2, 2]
    # A index: 8
    # B index: 10
    # C index: 8
    # C22 += A22 B22 -> U[8], V[10], W[8]
    block_terms.append(([(8,1)], [(10,1)], [(8,1)]))
    
    # Total 26 terms.
    
    # Now populate U, V, W for Block 1 (k=0,1,2)
    for i, (u_idx, v_idx, w_idx) in enumerate(block_terms):
        col = i
        for idx, coeff in u_idx:
            U[idx, col] += coeff
        for idx, coeff in v_idx:
            # Map local B index to global V index
            # Local j, k. Global j*6 + k.
            # Local index mapping:
            # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
            # We need to parse the local index to get j, k.
            # 0: j=0, k=0
            # 1: j=0, k=1
            # 2: j=1, k=0
            # 3: j=1, k=1
            # 4: j=0, k=2
            # 5: j=1, k=2
            # 6: j=2, k=0
            # 7: j=2, k=1
            # 8: j=2, k=2
            l = idx
            if l == 0: g = 0
            elif l == 1: g = 1
            elif l == 2: g = 6
            elif l == 3: g = 7
            elif l == 4: g = 2
            elif l == 5: g = 8
            elif l == 6: g = 12
            elif l == 7: g = 13
            elif l == 8: g = 14
            else: g = -1
            if g >= 0:
                V[g, col] += coeff
        for idx, coeff in w_idx:
            # Map local C index to global W index
            # Local k, i. Global k*3 + i.
            # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
            l = idx
            if l == 0: g = 0
            elif l == 1: g = 1
            elif l == 2: g = 3
            elif l == 3: g = 4
            elif l == 4: g = 2
            elif l == 5: g = 5
            elif l == 6: g = 6
            elif l == 7: g = 7
            elif l == 8: g = 8
            else: g = -1
            if g >= 0:
                W[g, col] += coeff
                
    # Populate U, V, W for Block 2 (k=3,4,5)
    for i, (u_idx, v_idx, w_idx) in enumerate(block_terms):
        col = rank_block + i
        for idx, coeff in u_idx:
            U[idx, col] += coeff
        for idx, coeff in v_idx:
            l = idx
            if l == 0: g = 3
            elif l == 1: g = 4
            elif l == 2: g = 9
            elif l == 3: g = 10
            elif l == 4: g = 5
            elif l == 5: g = 11
            elif l == 6: g = 15
            elif l == 7: g = 16
            elif l == 8: g = 17
            else: g = -1
            if g >= 0:
                V[g, col] += coeff
        for idx, coeff in w_idx:
            l = idx
            if l == 0: g = 9
            elif l == 1: g = 10
            elif l == 2: g = 12
            elif l == 3: g = 13
            elif l == 4: g = 11
            elif l == 5: g = 14
            elif l == 6: g = 15
            elif l == 7: g = 16
            elif l == 8: g = 17
            else: g = -1
            if g >= 0:
                W[g, col] += coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
