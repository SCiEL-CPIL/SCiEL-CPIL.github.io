import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 28 (for 4x4x4) + 16 (for 4x4x1) = 44
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 multiplication using Strassen (rank 28)
    # A is 4x4, B_left is 4x4 (cols 0-3 of B)
    # Split A into A00, A01, A10, A11 (2x2 blocks)
    # Split B_left into B00, B01, B10, B11 (2x2 blocks)
    
    # Strassen terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need a helper to add a rank-1 term for a 2x2x2 schoolbook multiplication
    # given the linear coefficients for the input matrices.
    # Input: 
    #   U_coeffs: dict mapping (i, j) to coeff for A block
    #   V_coeffs: dict mapping (j, k) to coeff for B block
    #   W_coeffs: dict mapping (i, k) to coeff for C block
    #   i_off, j_off: offset for A block (row, col)
    #   j_off, k_off: offset for B block (row, col)
    #   i_off_c, k_off_c: offset for C block (row, col)
    
    def add_2x2x2_block(u_coeffs, v_coeffs, w_coeffs, i_off, j_off, j_off_b, k_off_b, i_off_c, k_off_c):
        nonlocal col
        # u_coeffs: { (r, c): val } for A block
        # v_coeffs: { (r, c): val } for B block
        # w_coeffs: { (r, c): val } for C block
        # Schoolbook for 2x2x2:
        # C_rc = sum_d A_rd B_dc
        for a in range(2):
            for c in range(2):
                for b in range(2):
                    # Term: A[a,c] * B[c,b] contributes to C[a,b]
                    # Coeff for A is u_coeffs.get((a,c), 0)
                    # Coeff for B is v_coeffs.get((c,b), 0)
                    # Coeff for C is w_coeffs.get((a,b), 0)
                    
                    ua = u_coeffs.get((a, c), 0.0)
                    vb = v_coeffs.get((c, b), 0.0)
                    wc = w_coeffs.get((a, b), 0.0)
                    
                    if ua == 0 or vb == 0 or wc == 0:
                        continue
                        
                    # Global indices
                    # U index: (i_off + a) * m + (j_off + c)
                    u_idx = (i_off + a) * m + (j_off + c)
                    # V index: (j_off_b + c) * p + (k_off_b + b)
                    v_idx = (j_off_b + c) * p + (k_off_b + b)
                    # W index: (k_off_c + b) * n + (i_off_c + a)
                    w_idx = (k_off_c + b) * n + (i_off_c + a)
                    
                    U[u_idx][col] += ua
                    V[v_idx][col] += vb
                    W[w_idx][col] += wc
                    
                    col += 1

    # Define the blocks
    # A00: i_off=0, j_off=0
    # A01: i_off=0, j_off=2
    # A10: i_off=2, j_off=0
    # A11: i_off=2, j_off=2
    
    # B00: j_off=0, k_off=0
    # B01: j_off=0, k_off=2
    # B10: j_off=2, k_off=0
    # B11: j_off=2, k_off=2
    
    # C00: i_off=0, k_off=0
    # C01: i_off=0, k_off=2
    # C10: i_off=2, k_off=0
    # C11: i_off=2, k_off=2

    # M1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11
    # Y = B00 + B11
    # We need to add 4 terms for X*Y contributing to C00, C01, C10, C11?
    # No, M1 is a 2x2 matrix.
    # M1 contributes to C00, C01, C10, C11.
    # C00 += M1
    # C01 += M1
    # C10 += M1
    # C11 += M1
    # Wait, Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 (coeff 1), C01 (0), C10 (0), C11 (1).
    # But M1 is the product of X and Y.
    # X is a 2x2 matrix, Y is a 2x2 matrix.
    # The product M1 = X Y is a 2x2 matrix.
    # The elements of M1 are linear combinations of A and B.
    # We need to add the rank-1 terms for the multiplication X Y.
    # And the result is added to C00 and C11.
    
    # This means we need to call add_2x2x2_block for each output block that M1 contributes to.
    # But the U and V parts are the same for all output blocks?
    # No, U and V define the inputs. W defines the outputs.
    # If M1 contributes to C00 and C11, we need to add the W coefficients for both.
    # But U and V are the same.
    # So we can add the U and V parts once, and W parts for each output?
    # No, each rank-1 term in the tensor is a triple (u, v, w).
    # If the same u, v pair is used for multiple w's, they are different rank-1 terms?
    # No, $T = \sum u_r \otimes v_r \otimes w_r$.
    # If $u_1 \otimes v_1$ appears with $w_1$ and $w_2$, it's two terms:
    # $u_1 \otimes v_1 \otimes w_1 + u_1 \otimes v_1 \otimes w_2$.
    # This is valid.
    
    # So for M1:
    # u_coeffs for X: A00 (1) + A11 (1)
    # v_coeffs for Y: B00 (1) + B11 (1)
    # w_coeffs for C00: all 1s (2x2 identity)
    # w_coeffs for C11: all 1s (2x2 identity)
    
    # We will call add_2x2x2_block for C00, then for C11.
    # But this will duplicate U and V entries in different columns?
    # No, `col` is global.
    # If I call it for C00, it uses col, col+1, col+2, col+3.
    # If I call it for C11, it uses col+4, col+5, col+6, col+7.
    # This increases the rank!
    # Strassen's rank 7 relies on the fact that the *same* multiplications (M1..M7) are reused for different output blocks.
    # In the tensor decomposition, if I reuse the same U and V vectors, I can have different W vectors.
    # But the rank is the number of terms in the sum.
    # If I write $M_1 \otimes C_{00}$ and $M_1 \otimes C_{11}$, that's two terms.
    # But $M_1$ is itself a sum of 4 terms (schoolbook).
    # So $M_1 = \sum_{a,b,c} A_{ac} B_{cb} C_{ab}$.
    # So $M_1$ contributes 4 rank-1 terms to the tensor.
    # If $M_1$ contributes to $C_{00}$ and $C_{11}$, do we add 4 terms for $C_{00}$ and 4 terms for $C_{11}$?
    # Yes, because the W part is different.
    # So rank cost for M1 is $4 \times 2 = 8$?
    # No, wait.
    # The tensor is $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
    # If $u, v$ are fixed, and we have $w_1, w_2$, we have two terms.
    # So if M1 (4 terms) is used in C00 and C11, we have $4 + 4 = 8$ terms.
    # Total terms for 7 M's:
    # M1: 2 outputs (C00, C11) -> 8 terms
    # M2: 2 outputs (C10, C11) -> 8 terms
    # M3: 2 outputs (C01, C11) -> 8 terms
    # M4: 2 outputs (C00, C10) -> 8 terms
    # M5: 2 outputs (C00, C01) -> 8 terms
    # M6: 1 output (C11) -> 4 terms
    # M7: 1 output (C00) -> 4 terms
    # Total: $8+8+8+8+8+4+4 = 48$.
    # This is worse than 28!
    
    # Why is Strassen rank 7?
    # Because the "multiplication" is of the *blocks*.
    # The rank of the tensor $M_{<4,4,4>}$ is the minimum number of rank-1 tensors to sum to it.
    # Strassen's algorithm says that the *computational complexity* (number of scalar mults) is 28.
    # But the tensor rank is different.
    # The tensor rank of $M_{<4,4,4>}$ is 21.
    # My block-strassen approach gives a decomposition of rank 48?
    # No, if I use Strassen for the inner $2 \times 2 \times 2$ as well, the rank is different.
    # If I use schoolbook for inner, the rank is 48.
    # If I use Strassen for inner (rank 7), then each M_r is rank 7.
    # M1 is used in 2 outputs. So $7 \times 2 = 14$ terms?
    # Still high.
    
    # The key is that the tensor rank of $M_{<4,4,4>}$ is 21, which is less than 28.
    # This means there is a more clever decomposition.
    # Since I don't have the rank 21 decomposition memorized, I will stick to the simple schoolbook for the whole thing?
    # No, 80 is bad.
    # Can I use a known rank 25 or 28 decomposition?
    # Actually, the "block-strassen" with schoolbook inner is rank 48.
    # 48 is better than 80.
    # Let's use rank 48 for the $4 \times 4 \times 4$ part.
    # And rank 16 for the rest.
    # Total 64.
    
    # Wait, is there a simpler improvement?
    # What if I just use Strassen for $2 \times 2 \times 2$ on the whole matrix by padding?
    # No.
    
    # Let's try to implement the rank 48 construction.
    # It is straightforward.
    
    # Re-evaluating the cost:
    # M1: C00, C11. Cost 8.
    # M2: C10, C11. Cost 8.
    # M3: C01, C11. Cost 8.
    # M4: C00, C10. Cost 8.
    # M5: C00, C01. Cost 8.
    # M6: C11. Cost 4.
    # M7: C00. Cost 4.
    # Sum: 48.
    
    # This is valid.
    
    # Part 2: 4x4x1 (rank 16)
    # Total rank 64.
    
    # This is better than 80.
    
    # Let's code it.

    # Helper to add a block multiplication
    def add_block_mult(u_coeffs, v_coeffs, w_coeffs, i_off, j_off, j_off_b, k_off_b, i_off_c, k_off_c):
        nonlocal col
        for a in range(2):
            for c in range(2):
                for b in range(2):
                    ua = u_coeffs.get((a, c), 0.0)
                    vb = v_coeffs.get((c, b), 0.0)
                    wc = w_coeffs.get((a, b), 0.0)
                    if ua == 0 or vb == 0 or wc == 0:
                        continue
                    u_idx = (i_off + a) * m + (j_off + c)
                    v_idx = (j_off_b + c) * p + (k_off_b + b)
                    w_idx = (k_off_c + b) * n + (i_off_c + a)
                    U[u_idx][col] += ua
                    V[v_idx][col] += vb
                    W[w_idx][col] += wc
                    col += 1

    # M1 = (A00 + A11)(B00 + B11)
    u1 = {(0,0):1, (0,1):0, (1,0):0, (1,1):1, (2,0):1, (2,1):0, (3,0):0, (3,1):1} # No, coeffs are relative to block
    # u_coeffs keys are (a,c) where a,c in 0..1
    u1 = {(0,0):1, (1,1):1} # A00[0,0]+A11[0,0] etc? No.
    # X = A00 + A11.
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    # So u_coeffs for X: {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # But we need to map this to global A.
    # My add_block_mult assumes u_coeffs are for the *block* and handles offsets.
    # So u_coeffs = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # And we call it for A00 (i_off=0, j_off=0) and A11 (i_off=2, j_off=2)?
    # No, X is a single 2x2 matrix formed by summing two blocks.
    # The linear form is $X_{ac} = A_{00, ac} + A_{11, ac}$.
    # So we need to add terms for A00 and A11 separately?
    # Yes.
    # So for M1, we have 2 sources for U: A00 and A11.
    # And 2 sources for V: B00 and B11.
    # And 2 targets for W: C00 and C11.
    
    # This is getting complicated to manage with the helper.
    # I will just iterate over all combinations.
    
    # For each Strassen term M_r:
    # 1. Define list of (u_block, coeff)
    # 2. Define list of (v_block, coeff)
    # 3. Define list of (w_block, coeff)
    # 4. For each u_block, v_block, w_block:
    #    For each a,c,b in 2x2x2:
    #       Add term.
    
    strassen_terms = [
        # M1
        {
            'u': [('A00', 1), ('A11', 1)],
            'v': [('B00', 1), ('B11', 1)],
            'w': [('C00', 1), ('C11', 1)]
        },
        # M2
        {
            'u': [('A10', 1), ('A11', 1)],
            'v': [('B00', 1)],
            'w': [('C10', 1), ('C11', 1)]
        },
        # M3
        {
            'u': [('A00', 1)],
            'v': [('B01', 1), ('B11', -1)],
            'w': [('C01', 1), ('C11', 1)]
        },
        # M4
        {
            'u': [('A11', 1)],
            'v': [('B10', 1), ('B00', -1)],
            'w': [('C00', 1), ('C10', 1)]
        },
        # M5
        {
            'u': [('A00', 1), ('A01', 1)],
            'v': [('B11', 1)],
            'w': [('C00', 1), ('C01', 1)]
        },
        # M6
        {
            'u': [('A10', 1), ('A00', -1)],
            'v': [('B00', 1), ('B01', 1)],
            'w': [('C11', 1)]
        },
        # M7
        {
            'u': [('A01', 1), ('A11', -1)],
            'v': [('B10', 1), ('B11', 1)],
            'w': [('C00', 1)]
        }
    ]
    
    block_offsets = {
        'A00': (0, 0), 'A01': (0, 2), 'A10': (2, 0), 'A11': (2, 2),
        'B00': (0, 0), 'B01': (0, 2), 'B10': (2, 0), 'B11': (2, 2),
        'C00': (0, 0), 'C01': (0, 2), 'C10': (2, 0), 'C11': (2, 2)
    }
    
    for term in strassen_terms:
        for u_name, u_coeff in term['u']:
            u_off_i, u_off_j = block_offsets[u_name]
            for v_name, v_coeff in term['v']:
                v_off_j, v_off_k = block_offsets[v_name]
                for w_name, w_coeff in term['w']:
                    w_off_i, w_off_k = block_offsets[w_name]
                    # Add 2x2x2 schoolbook
                    for a in range(2):
                        for c in range(2):
                            for b in range(2):
                                u_idx = (u_off_i + a) * m + (u_off_j + c)
                                v_idx = (v_off_j + c) * p + (v_off_k + b)
                                w_idx = (w_off_k + b) * n + (w_off_i + a)
                                U[u_idx][col] += u_coeff
                                V[v_idx][col] += v_coeff
                                W[w_idx][col] += w_coeff
                                col += 1

    # Part 2: 4x4x1
    # C[:, 4] = A B[:, 4]
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
