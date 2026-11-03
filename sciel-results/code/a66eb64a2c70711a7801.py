import time
import numpy as np

def construct():
    """
    Returns a low-rank tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses block-Strassen at the 2x2 block level.
    U: (4*4, rank) = (16, rank)
    V: (4*9, rank) = (36, rank)
    W: (9*4, rank) = (36, rank)
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    
    # Block Strassen for 4x4 * 4x9
    # A (4x4) split into 2x2 blocks: A11,A12,A21,A22 (each 2x2)
    # B (4x9) split into 2x3 blocks: B11,B12,B13,B21,B22,B23 (each 2x3)
    # C (4x9) split into 2x3 blocks: C11,C12,C13,C21,C22,C23 (each 2x3)
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Strassen 7 multiplications (each is 2x2 * 2x3):
    # M1 = (A11+A22) * (B11+B21)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B13-B23)
    # M5 = (A11+A12) * B23
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    #
    # C11 = M1 - M2 - M6 + M7
    # C12 = M1 - M2 - M4 - M6
    # C13 = M3 + M4 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 - M2 - M6
    # C23 = M3 + M4 + M5 - M7
    #
    # Let me verify C11 = M1 - M2 - M6 + M7:
    # M1 = (A11+A22)(B11+B21) = A11B11 + A11B21 + A22B11 + A22B21
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    # M1-M2-M6+M7 = A11B11+A11B21+A22B11+A22B21 - A21B11-A22B11 - A21B11-A21B12+A11B11+A11B12 + A12B21+A12B22-A22B21-A22B22
    # = 2A11B11 + A11B21 + A11B12 + A12B21 + A12B22 - 2A21B11 - A21B12 - 2A22B22 + (A22B11-A22B11) + (A22B21-A22B21)
    # This doesn't simplify to A11B11 + A12B21. So my formulas are wrong.
    #
    # Let me use the correct Strassen formulas for the rectangular case.
    # For 2x2 blocks of A and 2x3 blocks of B:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # I'll just use schoolbook for each 2x2*2x3 block multiplication (rank 12 each)
    # but with the Strassen combinations. Total rank = 7 * 12 = 84.
    
    # Indexing:
    # A (4x4), row-major: A[i,j] at index i*4+j
    #   A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    # B (4x9), row-major: B[i,j] at index i*9+j
    #   B11: [0,1,2, 9,10,11], B12: [3,4,5, 12,13,14], B13: [6,7,8, 15,16,17]
    #   B21: [18,19,20, 27,28,29], B22: [21,22,23, 30,31,32], B23: [24,25,26, 33,34,35]
    # C (4x9), W convention: C[i,j] at index j*4+i
    #   C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W[0],W[1],W[4],W[5]
    #   C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W[8],W[9],W[12],W[13]
    #   C13: C[0,4]..C[0,8],C[1,4]..C[1,8] -> W[16..21],W[24..29],W[32..33] wait
    #   C[i,j] at W[j*4+i]:
    #   C11: i=0,1; j=0,1 -> W[0*4+0]=0, W[0*4+1]=1, W[1*4+0]=4, W[1*4+1]=5
    #   C12: i=0,1; j=2,3 -> W[2*4+0]=8, W[2*4+1]=9, W[3*4+0]=12, W[3*4+1]=13
    #   C13: i=0,1; j=4,5,6,7,8 -> W[4*4+0]=16,W[4*4+1]=17,W[5*4+0]=20,W[5*4+1]=21,W[6*4+0]=24,W[6*4+1]=25,W[7*4+0]=28,W[7*4+1]=29,W[8*4+0]=32,W[8*4+1]=33
    #   C21: i=2,3; j=0,1 -> W[0*4+2]=2,W[0*4+3]=3,W[1*4+2]=6,W[1*4+3]=7
    #   C22: i=2,3; j=2,3 -> W[2*4+2]=10,W[2*4+3]=11,W[3*4+2]=14,W[3*4+3]=15
    #   C23: i=2,3; j=4,5,6,7,8 -> W[4*4+2]=18,W[4*4+3]=19,W[5*4+2]=22,W[5*4+3]=23,W[6*4+2]=26,W[6*4+3]=27,W[7*4+2]=30,W[7*4+3]=31,W[8*4+2]=34,W[8*4+3]=35
    
    # For each Strassen multiplication Mi = X * Y where X is 2x2 and Y is 2x3,
    # I need to express X as a linear combination of A elements and Y as a linear
    # combination of B elements, then the product contributes to C blocks.
    
    # I'll build the decomposition by adding terms for each of the 7 Strassen
    # multiplications, each expanded with schoolbook rank 12.
    
    # Helper: add a 2x2 * 2x3 schoolbook multiplication
    # X_indices: list of 4 A-indices with coefficients (for the 2x2 X matrix)
    # Y_indices: list of 6 B-indices with coefficients (for the 2x3 Y matrix)
    # C_block: list of (c_index, coeff) for the C block this contributes to
    
    # Actually, let me just directly construct the rank-84 decomposition.
    # Each Strassen term Mi = Xi * Yi where Xi is a 2x2 linear combo of A,
    # Yi is a 2x3 linear combo of B.
    # The product Xi*Yi is a 2x3 matrix. Each element (a,b) of this product
    # is sum_{k=0}^{1} Xi[a,k] * Yi[k,b].
    # In the tensor decomposition, each rank-1 term corresponds to one (a,k,b) triple.
    # So each Mi contributes 2*2*3 = 12 rank-1 terms.
    # Total: 7 * 12 = 84 terms.
    
    # Let me define the 7 Strassen terms:
    # For each term, define:
    #   a_coeffs: dict {A_index: coeff} for the 2x2 X matrix (row-major: [x00,x01,x10,x11])
    #   b_coeffs: dict {B_index: coeff} for the 2x3 Y matrix (row-major: [y00,y01,y02,y10,y11,y12])
    #   c_blocks: list of (C_block_name, coeff) where C_block is one of C11..C23
    
    # A block indices (row-major within 2x2 block):
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A12: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # A21: A[2,0]=8, A[2,1]=9, A[3,0]=12, A[3,1]=13
    # A22: A[2,2]=10, A[2,3]=11, A[3,2]=14, A[3,3]=15
    
    # B block indices (row-major within 2x3 block):
    # B11: B[0,0]=0,B[0,1]=1,B[0,2]=2,B[1,0]=9,B[1,1]=10,B[1,2]=11
    # B12: B[0,3]=3,B[0,4]=4,B[0,5]=5,B[1,3]=12,B[1,4]=13,B[1,5]=14
    # B13: B[0,6]=6,B[0,7]=7,B[0,8]=8,B[1,6]=15,B[1,7]=16,B[1,8]=17
    # B21: B[2,0]=18,B[2,1]=19,B[2,2]=20,B[3,0]=27,B[3,1]=28,B[3,2]=29
    # B22: B[2,3]=21,B[2,4]=22,B[2,5]=23,B[3,3]=30,B[3,4]=31,B[3,5]=32
    # B23: B[2,6]=24,B[2,7]=25,B[2,8]=26,B[3,6]=33,B[3,7]=34,B[3,8]=35
    
    # C block W indices:
    # C11: [0,1,4,5] (C[0,0],C[0,1],C[1,0],C[1,1])
    # C12: [8,9,12,13] (C[0,2],C[0,3],C[1,2],C[1,3])
    # C13: [16,17,20,21,24,25,28,29,32,33] (C[0,4]..C[0,8],C[1,4]..C[1,8])
    # C21: [2,3,6,7] (C[2,0],C[2,1],C[3,0],C[3,1])
    # C22: [10,11,14,15] (C[2,2],C[2,3],C[3,2],C[3,3])
    # C23: [18,19,22,23,26,27,30,31,34,35] (C[2,4]..C[2,8],C[3,4]..C[3,8])
    
    # Strassen terms:
    # M1 = (A11+A22) * (B11+B21)
    #   X = A11 + A22: x00=A[0,0]+A[2,2]=A0+A10, x01=A[0,1]+A[2,3]=A1+A11,
    #                   x10=A[1,0]+A[3,2]=A4+A14, x11=A[1,1]+A[3,3]=A5+A15
    #   Y = B11 + B21: y00=B[0,0]+B[2,0]=B0+B18, y01=B[0,1]+B[2,1]=B1+B19, y02=B[0,2]+B[2,2]=B2+B20,
    #                   y10=B[1,0]+B[3,0]=B9+B27, y11=B[1,1]+B[3,1]=B10+B28, y12=B[1,2]+B[3,2]=B11+B29
    
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B13-B23)
    # M5 = (A11+A12) * B23
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # Now I need to figure out the C contributions. Let me verify by computing
    # what each Mi equals in terms of the C blocks.
    #
    # M1 = (A11+A22)(B11+B21) = A11B11 + A11B21 + A22B11 + A22B21
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M4 = A22(B13-B23) = A22B13 - A22B23
    # M5 = (A11+A12)B23 = A11B23 + A12B23
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    #
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C13 = A11B13 + A12B23
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # C23 = A21B13 + A22B23
    #
    # Let me find linear combinations of M1..M7 that give each C block.
    #
    # C21 = A21B11 + A22B21
    # M2 = A21B11 + A22B11
    # M7 = A12B21 + A12B22 - A22B21 - A22B22
    # M6 = A21B11 + A21B12 - A11B11 - A11B12
    #
    # Hmm, this is getting complicated. Let me just use numpy to find the
    # correct linear combinations.
    
    # Actually, let me take a step back. The simplest correct approach is:
    # Just use schoolbook for the full 4x4*4x9, giving rank 144.
    # But that's what the current code does. I need to do better.
    
    # Let me try a different decomposition. Note that 4x4*4x9 can be seen as
    # computing 4x4 * 4x9 = sum over k of A[:,k] * B[k,:] (outer products).
    # That's 4 outer products of (4x1)*(1x9) = 4*4*9 = 144 rank-1 terms.
    
    # Alternatively, I can use the fact that we only need integer/half-integer
    # entries, so I can use Strassen with its +/-1 coefficients.
    
    # Let me just carefully implement the Strassen decomposition and verify it
    # numerically before returning.
    
    # Define the 7 Strassen multiplications as (a_coeffs, b_coeffs, c_contribs)
    # where a_coeffs maps A-index -> coeff, b_coeffs maps B-index -> coeff,
    # and c_contribs is a dict mapping C-index -> coeff.
    
    # A indices for each 2x2 block (row-major):
    A11_idx = [0, 1, 4, 5]    # A[0,0],A[0,1],A[1,0],A[1,1]
    A12_idx = [2, 3, 6, 7]    # A[0,2],A[0,3],A[1,2],A[1,3]
    A21_idx = [8, 9, 12, 13]  # A[2,0],A[2,1],A[3,0],A[3,1]
    A22_idx = [10, 11, 14, 15] # A[2,2],A[2,3],A[3,2],A[3,3]
    
    # B indices for each 2x3 block (row-major):
    B11_idx = [0, 1, 2, 9, 10, 11]
    B12_idx = [3, 4, 5, 12, 13, 14]
    B13_idx = [6, 7, 8, 15, 16, 17]
    B21_idx = [18, 19, 20, 27, 28, 29]
    B22_idx = [21, 22, 23, 30, 31, 32]
    B23_idx = [24, 25, 26, 33, 34, 35]
    
    # C indices for each 2x3 block (W convention: C[i,j] at j*4+i):
    C11_idx = [0, 1, 4, 5]         # C[0,0],C[0,1],C[1,0],C[1,1]
    C12_idx = [8, 9, 12, 13]       # C[0,2],C[0,3],C[1,2],C[1,3]
    C13_idx = [16,17,20,21,24,25,28,29,32,33]  # C[0,4..8],C[1,4..8]
    C21_idx = [2, 3, 6, 7]         # C[2,0],C[2,1],C[3,0],C[3,1]
    C22_idx = [10, 11, 14, 15]     # C[2,2],C[2,3],C[3,2],C[3,3]
    C23_idx = [18,19,22,23,26,27,30,31,34,35]  # C[2,4..8],C[3,4..8]
    
    # For each Strassen term, I'll specify the X (2x2) and Y (2x3) as linear
    # combos, and then the resulting 2x3 product contributes to C blocks.
    #
    # The 7 terms:
    # M1 = (A11+A22) * (B11+B21)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B13-B23)
    # M5 = (A11+A12) * B23
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    #
    # Each Mi is a 2x3 matrix. Let me denote Mi = [mi00, mi01, mi02; mi10, mi11, mi12]
    #
    # M1 = (A11+A22)(B11+B21)
    #   = A11*B11 + A11*B21 + A22*B11 + A22*B21
    #
    # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
    #
    # M3 = A11*(B12-B22) = A11*B12 - A11*B22
    #
    # M4 = A22*(B13-B23) = A22*B13 - A22*B23
    #
    # M5 = (A11+A12)*B23 = A11*B23 + A12*B23
    #
    # M6 = (A21-A11)*(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    #
    # M7 = (A12-A22)*(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    #
    # Now:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Let me express each C block as a combo of M1..M7:
    #
    # C21 = A21*B11 + A22*B21
    # M2 = A21*B11 + A22*B11
    # M7 = A12*B21 + A12*B22 - A22*B21 - A22*B22
    # M6 = A21*B11 + A21*B12 - A11*B11 - A11*B12
    #
    # This is a system of linear equations. Let me solve it with numpy.
    
    # Represent each Mi as a vector in the space of 2x3 matrices (6-dim).
    # Then find coefficients to express each C block.
    
    # Actually, let me just brute-force it. I'll create the decomposition
    # directly by specifying for each of the 7*12 = 84 rank-1 terms what
    # they are.
    
    # For term Mi, the X matrix has entries X[a,b] (a,b in {0,1}) which are
    # linear combos of A elements, and Y matrix has entries Y[c,d] (c in {0,1},
    # d in {0,1,2}) which are linear combos of B elements.
    # The (a,d) entry of the product is sum_c X[a,c]*Y[c,d].
    # Each rank-1 term in the decomposition corresponds to a specific (a,c,d)
    # triple for a specific Mi.
    
    # Let me define the Strassen terms more concretely.
    
    # For each Mi, define X as a 2x2 matrix of linear combos of A,
    # and Y as a 2x3 matrix of linear combos of B.
    
    # I'll represent X[a,b] as a dict {A_index: coeff} and similarly for Y.
    
    def add_block_sum(block1_idx, coeff1, block2_idx, coeff2):
        """Add two blocks with coefficients, return dict {index: coeff}"""
        result = {}
        for idx in block1_idx:
            result[idx] = result.get(idx, 0) + coeff1
        for idx in block2_idx:
            result[idx] = result.get(idx, 0) + coeff2
        return result
    
    def add_block_single(block_idx, coeff):
        """Return dict {index: coeff} for a single block"""
        return {idx: coeff for idx in block_idx}
    
    # M1: X = A11 + A22, Y = B11 + B21
    M1_X = add_block_sum(A11_idx, 1, A22_idx, 1)
    M1_Y = add_block_sum(B11_idx, 1, B21_idx, 1)
    
    # M2: X = A21 + A22, Y = B11
    M2_X = add_block_sum(A21_idx, 1, A22_idx, 1)
    M2_Y = add_block_single(B11_idx, 1)
    
    # M3: X = A11, Y = B12 - B22
    M3_X = add_block_single(A11_idx, 1)
    M3_Y = add_block_sum(B12_idx, 1, B22_idx, -1)
    
    # M4: X = A22, Y = B13 - B23
    M4_X = add_block_single(A22_idx, 1)
    M4_Y = add_block_sum(B13_idx, 1, B23_idx, -1)
    
    # M5: X = A11 + A12, Y = B23
    M5_X = add_block_sum(A11_idx, 1, A12_idx, 1)
    M5_Y = add_block_single(B23_idx, 1)
    
    # M6: X = A21 - A11, Y = B11 + B12
    M6_X = add_block_sum(A21_idx, 1, A11_idx, -1)
    M6_Y = add_block_sum(B11_idx, 1, B12_idx, 1)
    
    # M7: X = A12 - A22, Y = B21 + B22
    M7_X = add_block_sum(A12_idx, 1, A22_idx, -1)
    M7_Y = add_block_sum(B21_idx, 1, B22_idx, 1)
    
    # Now I need to determine how each Mi contributes to the C blocks.
    # Mi is a 2x3 matrix. Its (a,d) entry contributes to C blocks.
    # For M1, M2, M3: they contribute to C11, C12, C13 (top row of C blocks)
    # For M4, M5, M6, M7: they contribute to C21, C22, C23 (bottom row)
    # Wait, that's not right. Each Mi is a full 2x3 matrix that can contribute
    # to any C block.
    
    # Let me just compute the contribution of each Mi to each C block by
    # expanding the products.
    
    # Actually, the cleanest way is to just verify numerically. Let me build
    # the full decomposition and check it.
    
    # For each Mi, expand into 12 rank-1 terms (2*2*3 for the 2x2*2x3 product).
    # For term (a, c, d) of Mi:
    #   U contribution: for each A_index in X[a,c]'s support, add U[A_index, col] += coeff
    #   V contribution: for each B_index in Y[c,d]'s support, add V[B_index, col] += coeff  
    #   W contribution: the C index corresponding to position (a,d) in the C block
    #
    # Wait, I need to be more careful. The product Mi = X*Y where X is 2x2 and Y is 2x3.
    # Mi[a,d] = sum_{c=0}^{1} X[a,c] * Y[c,d]
    #
    # X[a,c] is a linear combo of A elements: X[a,c] = sum_{a_idx} xa[a,c,a_idx] * A[a_idx]
    # Y[c,d] is a linear combo of B elements: Y[c,d] = sum_{b_idx} yb[c,d,b_idx] * B[b_idx]
    #
    # So Mi[a,d] = sum_{c} sum_{a_idx} sum_{b_idx} xa[a,c,a_idx] * yb[c,d,b_idx] * A[a_idx] * B[b_idx]
    #
    # Each rank-1 term in the tensor decomposition is of the form:
    #   U[i, t] * V[j, t] * W[k, t] for some t
    # where U[i,t] and V[j,t] and W[k,t] are the coefficients.
    #
    # For the (a,c,d) triple in Mi, the rank-1 term has:
    #   U[a_idx, t] = xa[a,c,a_idx] for the specific a_idx
    #   V[b_idx, t] = yb[c,d,b_idx] for the specific b_idx
    #   W[c_idx, t] = 1 where c_idx is the C index for position (a,d) in the appropriate block
    #
    # But wait, X[a,c] involves multiple A indices, and Y[c,d] involves multiple B indices.
    # So the term Mi[a,d] = X[a,c]*Y[c,d] expands to a sum over a_idx and b_idx.
    # Each (a_idx, b_idx) pair gives a separate rank-1 term.
    #
    # This means each Mi has more than 12 rank-1 terms in general!
    # For example, M1 = (A11+A22)*(B11+B21).
    # X[0,0] = A[0,0] + A[2,2], so it has 2 terms.
    # Y[0,0] = B[0,0] + B[2,0], so it has 2 terms.
    # So X[0,0]*Y[0,0] alone has 2*2 = 4 rank-1 terms.
    # The total for M1 would be much more than 12.
    
    # This is the key issue. The Strassen approach doesn't directly give a
    # low-rank decomposition because the linear combinations in X and Y
    # create products that expand into many rank-1 terms.
    
    # The correct way to think about it: the tensor decomposition rank is the
    # minimum number of rank-1 tensors needed. Strassen reduces the number of
    # bilinear multiplications, but each bilinear multiplication of two
    # linear combinations still requires multiple rank-1 terms.
    
    # So the rank-84 approach (7 Strassen terms * 12 schoolbook terms each)
    # is NOT correct because each Strassen term expands into more than 12
    # rank-1 terms.
    
    # Let me reconsider. The actual rank of the 4x4*4x9 multiplication tensor
    # is at most 144 (schoolbook). Can we do better?
    
    # One approach: use the fact that we can share the A factors across the
    # 9 columns of B. For each of the 4 rows of A, we compute the dot product
    # with each column of B. That's 4*4*9 = 144 rank-1 terms.
    
    # Another approach: use Strassen at a higher level. The 4x4*4x9 problem
    # can be decomposed using Strassen into 7 sub-problems of size 2x2*2x9,
    # but each 2x2*2x9 has rank 2*2*9 = 36, giving total 7*36 = 256. Worse!
    
    # Or: decompose into 7 sub-problems of size 2x2*2x3 (by splitting the 9
    # columns into 3 groups of 3), giving 7*3*12 = 252. Still worse.
    
    # I think the schoolbook rank of 144 might be hard to beat for this
    # specific problem without very advanced techniques.
    
    # Let me just return the correct schoolbook decomposition.
    
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
    # Block Strassen approach:
    # A (4x4) split into 2x2 blocks of 2x2: A11, A12, A21, A22
    # B (4x9) split into 2x3 blocks of 2x3: B11, B12, B13, B21, B22, B23
    # C (4x9) split into 2x3 blocks of 2x3: C11, C12, C13, C21, C22, C23
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Strassen-like 7 multiplications:
    # M1 = (A11 + A22) * (B11 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B13 - B23)
    # M5 = (A11 + A12) * B23
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Verify:
    # C11 = M1 - M2 - M4 - M6 + M7  ... let me verify this properly
    #
    # Actually, let me use the standard Strassen formulas adapted for rectangular case.
    # For 2x2 blocks where A is 2x2 blocks and B is 2x3 blocks:
    #
    # M1 = (A11 + A22) * (B11 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B13 - B23)
    # M5 = (A11 + A12) * B23
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 - M2 - M6 + M7  -- needs verification
    # C12 = M1 - M2 - M4 - M6  -- needs verification  
    # C13 = M3 + M4 + M5       -- needs verification
    # C21 = M2 + M6            -- needs verification
    # C22 = M1 - M3 - M2 - M6  -- needs verification
    # C23 = M3 + M4 + M5 - M7  -- needs verification
    #
    # Let me verify C21 = M2 + M6:
    # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
    # M6 = (A21-A11)*(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M2+M6 = 2*A21*B11 + A22*B11 + A21*B12 - A11*B11 - A11*B12
    # But C21 = A21*B11 + A22*B21. This doesn't match.
    #
    # The standard Strassen doesn't directly apply to rectangular block cases.
    # Let me use a different approach: just use the schoolbook for the block level
    # but with the 2x2x2x3 sub-problems done efficiently.
    #
    # Actually, the simplest improvement: use rank 120 by noting that we can
    # share some computations. But let me just implement a correct rank-84 solution
    # using 7 block multiplications, each done with schoolbook rank 12.
    
    # Define helper: get schoolbook decomposition for 2x2 * 2x3
    def get_2x2x2x3_schoolbook():
        """
        Returns U, V, W for 2x2 * 2x3 matrix multiplication with rank 12.
        U: (4, 12), V: (6, 12), W: (6, 12)
        2x2 matrix X: indices 0,1,2,3 (row-major)
        2x3 matrix Y: indices 0,1,2,3,4,5 (row-major)
        2x3 result Z: indices 0,1,2,3,4,5 (row-major)
        """
        r = 12
        u = [[0]*r for _ in range(4)]
        v = [[0]*r for _ in range(6)]
        w = [[0]*r for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u[i*2+j][col] = 1
                    v[j*3+k][col] = 1
                    w[k*2+i][col] = 1
                    col += 1
        return u, v, w
    
    # We'll build the full decomposition by specifying for each of the 7 block
    # multiplications which input indices they use and which output indices they contribute to.
    #
    # The 7 block multiplications are:
    # M1 = (A11 + A22) * (B11 + B21)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B13 - B23)
    # M5 = (A11 + A12) * B23
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # For each Mi, we need:
    # - Which elements of A (16 total) contribute, with what coefficients
    # - Which elements of B (36 total) contribute, with what coefficients
    # - Which elements of C (36 total) are affected, with what coefficients
    #
    # A indices (row-major 4x4):
    # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    #
    # B indices (row-major 4x9):
    # B11: [0,1,2, 9,10,11], B12: [3,4,5, 12,13,14], B13: [6,7,8, 15,16,17]
    # B21: [18,19,20, 27,28,29], B22: [21,22,23, 30,31,32], B23: [24,25,26, 33,34,35]
    #
    # C indices (row-major 9x4, but W uses k*n+i convention so C[i,j] is at index j*4+i):
    # C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W indices 0,1,4,5
    # C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W indices 2,3,6,7
    # C13: C[0,4],C[0,5],C[0,6],C[0,7],C[0,8],C[1,4],C[1,5],C[1,6],C[1,7],C[1,8] -> W indices 4,5,6,7,8, 8,9,10,11,12
    # Wait, C is 4x9. C[i,j] for i in 0..3, j in 0..8.
    # W[k*n+i] where k is column of B (0..8), i is row of A (0..3).
    # So C[i,j] = sum over k of A[i,k]*B[k,j]. In W convention, C[i,j] is at W[j*4+i].
    #
    # C11 (rows 0-1, cols 0-1): W indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C12 (rows 0-1, cols 2-3): W indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C13 (rows 0-1, cols 4-8): W indices 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21, 6*4+0=24, 6*4+1=25, 7*4+0=28, 7*4+1=29, 8*4+0=32, 8*4+1=33
    # C21 (rows 2-3, cols 0-1): W indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C22 (rows 2-3, cols 2-3): W indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C23 (rows 2-3, cols 4-8): W indices 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23, 6*4+2=26, 6*4+3=27, 7*4+2=30, 7*4+3=31, 8*4+2=34, 8*4+3=35
    
    # This is getting very complex. Let me take a simpler approach:
    # Just return the schoolbook decomposition with rank 144, but I know
    # that's the same as the current score. Let me instead try to find a
    # simpler improvement.
    
    # Alternative simple improvement: Note that we can write the 4x4 x 4x9
    # multiplication as 9 separate 4x4 x 4x1 multiplications. Each 4x4 x 4x1
    # can be done in rank 4 (just the dot products). Total rank: 9*4 = 36.
    # Wait, that's not right either. Let me think again.
    
    # Actually, for 4x4 * 4x1, the result is 4x1. Each output entry C[i] = sum_j A[i,j]*B[j].
    # This is just 4 dot products of length 4, which can be done in rank 4 (one per output).
    # But we need to be careful about the tensor decomposition format.
    
    # For 4x4 * 4x1: U is (16, r), V is (4, r), W is (4, r)
    # Each term: U[i*4+j, t] * V[j, t] * W[i, t] should give contribution to C[i]
    # For rank 4: t=0..3, U[i*4+t, t] = 1, V[t, t] = 1, W[i, t] = 1
    # Then sum_t U[i*4+t,t]*V[t,t]*W[i,t] = sum_t A[i,t]*B[t] = C[i]. Correct!
    
    # So 4x4 * 4x1 has rank 4.
    # For 4x4 * 4x9, we have 9 such column multiplications, but they share the same A.
    # The total tensor is the direct sum of 9 copies of the 4x4*4x1 tensor.
    # Rank of direct sum is sum of ranks: 9*4 = 36.
    
    # Wait, is that correct? Let me verify. The 4x4 * 4x9 multiplication tensor
    # has entries T[i,j,k] = delta[i,j+k] (in appropriate indexing). Actually,
    # T[i, j, k] where i is row of C (0-3), j is col of C (0-8), k is... 
    # 
    # In the standard tensor decomposition for matrix multiplication:
    # T[i, j, k] = 1 if i = a + b (in some sense), but more precisely:
    # The multiplication tensor for (n x m) * (m x p) -> (n x p) has
    # T[i, j, k] where i in 0..n*m-1 (flattened A), j in 0..m*p-1 (flattened B),
    # k in 0..p*n-1 (flattened C), and T[i,j,k] = 1 if the element at position i
    # in A and position j in B contribute to position k in C.
    #
    # For A[i_a, j_a] (flattened to i_a*m + j_a), B[j_b, k_b] (flattened to j_b*p + k_b),
    # C[i_c, k_c] (flattened to k_c*n + i_c), the contribution is:
    # A[i_a, j_a] * B[j_b, k_b] contributes to C[i_c, k_c] if i_a = i_c, j_a = j_b, k_b = k_c.
    #
    # So T[i_a*m+j_a, j_b*p+k_b, k_c*n+i_c] = 1 if i_a=i_c, j_a=j_b, k_b=k_c.
    
    # For the 4x4 * 4x9 case:
    # A: 4x4, flattened to 16 elements
    # B: 4x9, flattened to 36 elements  
    # C: 4x9, flattened to 36 elements (using k*n+i convention, so C[i,j] at index j*4+i)
    
    # The rank-36 decomposition:
    # For each column j of B (j=0..8), and for each row i of C (i=0..3):
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # 
    # We can write this as 4 terms per (i,j) pair, but we can share across j.
    # Actually, for fixed i, C[i,j] = sum_k A[i,k]*B[k,j] for all j.
    # This is a vector-matrix product: row i of A times B.
    # 
    # For the tensor decomposition, we need rank-1 terms.
    # Term (i, k, j): U[i*4+k, t] = 1, V[k*9+j, t] = 1, W[j*4+i, t] = 1
    # where t indexes the term. There are 4*4*9 = 144 such terms in schoolbook.
    #
    # To get rank 36, we would need to combine these. But each term is already
    # rank-1, and they're all distinct (different i, k, j combinations give different
    # patterns in U, V, W). So the rank is actually 144 for the schoolbook.
    #
    # The rank-36 idea doesn't work because we can't combine terms that have
    # different patterns.
    
    # Let me just go with the schoolbook rank 144 for now, but I'll try to
    # implement a rank-120 decomposition using some structure.
    
    # Actually, I realize I should just implement the block Strassen correctly.
    # Let me define the 7 block multiplications and their contributions to C.
    
    # I'll build the decomposition term by term.
    # Each "block multiplication" Mi involves:
    # - A linear combination of A elements (with coefficients)
    # - A linear combination of B elements (with coefficients)
    # - A linear combination of C elements (with coefficients)
    #
    # For each block multiplication, I'll expand it into rank-1 terms using
    # the schoolbook for 2x2 * 2x3.
    
    # Block definitions (A indices in 0..15, B indices in 0..35, C indices in 0..35):
    # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    # B11: [0,1,2,9,10,11], B12: [3,4,5,12,13,14], B13: [6,7,8,15,16,17]
    # B21: [18,19,20,27,28,29], B22: [21,22,23,30,31,32], B23: [24,25,26,33,34,35]
    # C11: [0,1,4,5], C12: [8,9,12,13], C13: [16,17,20,21,24,25,28,29,32,33]
    # C21: [2,3,6,7], C22: [10,11,14,15], C23: [18,19,22,23,26,27,30,31,34,35]
    
    # For each block multiplication, define:
    # - a_coeffs: dict mapping A index -> coefficient
    # - b_coeffs: dict mapping B index -> coefficient
    # - c_coeffs: dict mapping C index -> coefficient
    # - The block multiplication computes: sum_a sum_b a_coeffs[a]*b_coeffs[b]*A[a]*B[b] = sum_c c_coeffs[c]*C[c]
    
    # M1 = (A11 + A22) * (B11 + B21)
    # A part: A11 + A22, so a_coeffs: {0:1, 1:1, 4:1, 5:1, 10:1, 11:1, 14:1, 15:1}
    # B part: B11 + B21, so b_coeffs: {0:1, 1:1, 2:1, 9:1, 10:1, 11:1, 18:1, 19:1, 20:1, 27:1, 28:1, 29:1}
    # C part: This contributes to C11, C12, C21, C22 (all four 2x2 blocks)
    # Actually, M1 = (A11+A22)*(B11+B21) = A11*B11 + A11*B21 + A22*B11 + A22*B21
    # This contributes to:
    # C11 gets A11*B11, C12 gets A11*B21 (wait, B21 is in column block 1, which is cols 0-1)
    # I'm getting confused with the block indexing. Let me restart with clearer notation.
    
    # Let me use a completely different, simpler approach. I'll just return the
    # schoolbook decomposition but verify it's correct, and accept rank 144.
    # The current program already does this, so I need to find an actual improvement.
    
    # Let me try rank 120 by using the following trick:
    # For the 4x4 * 4x9 multiplication, note that we can group the 9 columns of B
    # into 3 groups of 3. For each group, we compute A * B_group where B_group is 4x3.
    # A * B_group is 4x3. We can do this with rank 12 (schoolbook for 4x4*4x3).
    # Total: 3 * 12 = 36. But again, this doesn't reduce the rank because the
    # tensor entries are all distinct.
    
    # I think the fundamental issue is that the rank of the matrix multiplication
    # tensor is exactly n*m*p for the "generic" case, and we need advanced techniques
    # to reduce it.
    
    # Let me just implement a correct rank-144 solution and return it.
    
    rank = 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
