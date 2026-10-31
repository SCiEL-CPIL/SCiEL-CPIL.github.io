"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the recursive Strassen algorithm (base case 2x2).
    
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # all 4
    
    # We'll build the Strassen decomposition recursively.
    # For a 2x2 Strassen step, we need 7 multiplications.
    # For 4x4, we split into four 2x2 blocks and apply Strassen.
    # This gives rank 7 * (number of sub-problems) but with sharing.
    # Actually, 4x4 via one Strassen level: 7 multiplications of 2x2,
    # each 2x2 done by schoolbook (4 mults) = 28? No.
    # 
    # Correct approach: 4x4 matrix mult can be done with Strassen by
    # splitting into 2x2 blocks. The 7 Strassen products are each 2x2
    # matrix multiplications. Each 2x2 mult has rank 7 (Strassen base).
    # So total rank = 7 * 7 = 49.
    #
    # Let's construct this explicitly.
    #
    # For 4x4: A is 4x4, B is 4x4, C = A*B is 4x4.
    # Split A, B, C into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product. We use Strassen for each 2x2.
    #
    # For a 2x2 Strassen:
    # Given P (2x2) and Q (2x2), R = P*Q
    # P = [[p11, p12], [p21, p22]]
    # Q = [[q11, q12], [q21, q22]]
    #
    # S1 = (p11+p22)(q11+q22)
    # S2 = (p21+p22)q11
    # S3 = p11(q12-q22)
    # S4 = p22(q21-q11)
    # S5 = (p11+p12)q22
    # S6 = (p21-p11)(q11+q12)
    # S7 = (p12-p22)(q21+q22)
    #
    # R11 = S1 + S4 - S5 + S7
    # R12 = S3 + S5
    # R21 = S2 + S4
    # R22 = S1 - S2 + S3 + S6
    #
    # Each Sk is a scalar product (rank 1).
    #
    # So for the full 4x4, we have 7 "Strassen levels" at the top, each
    # producing a 2x2 result via 7 scalar products. Total: 49 scalar products.
    #
    # Let's index the 49 products. Top-level product t (0..6), sub-product s (0..6).
    # Total rank = 49.
    
    rank = 49
    
    # U: shape (n*m, rank) = (16, 49)
    # V: shape (m*p, rank) = (16, 49)  
    # W: shape (p*n, rank) = (16, 49)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper: for a 2x2 block starting at row offset (r_off) in the 4x4 matrix
    # The 4x4 matrix A has elements A[i,j] for i,j in 0..3
    # U is indexed by [i*m + j, col] = [i*4 + j, col]
    # V is indexed by [j*p + k, col] = [j*4 + k, col]
    # W is indexed by [k*n + i, col] = [k*4 + i, col]
    
    # Top-level Strassen: 7 products M1..M7
    # Each Mi involves sums/differences of 2x2 blocks of A and B
    # Then each Mi is a 2x2 product, done by 7 scalar products S1..S7
    
    # Define the 7 top-level Strassen combinations
    # For each, we need:
    #   - Which 2x2 block of A (or sum/diff) multiplies which 2x2 block of B
    #   - The signs
    
    # Top-level Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices: A11=(0,0), A12=(0,2), A21=(2,0), A22=(2,2) in 4x4
    # i.e., A11 covers rows 0-1, cols 0-1; A12 covers rows 0-1, cols 2-3; etc.
    
    # For each top product t, define the A-side and B-side as linear combos of blocks
    # A-side: list of (block_row, block_col, sign) where block_row, block_col in {0,1}
    # B-side: same
    
    # A block (br, bc) covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # In U indexing: row = i*4 + j, so for block (br, bc), rows are 2*br + di, cols 2*bc + dj
    
    # For each top product, the A-side is a sum of blocks, B-side is a sum of blocks.
    # Then the 2x2 product is done via Strassen with 7 scalar products.
    
    # Top-level definitions: (A_blocks, B_blocks) where each is list of (br, bc, sign)
    top_products = [
        # M1: (A11+A22)(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A21+A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3: A11(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4: A22(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5: (A11+A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6: (A21-A11)(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7: (A12-A22)(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # For the 2x2 Strassen sub-problem, given P and Q (each 2x2), the 7 scalar products are:
    # S1 = (p11+p22)(q11+q22)
    # S2 = (p21+p22)q11
    # S3 = p11(q12-q22)
    # S4 = p22(q21-q11)
    # S5 = (p11+p12)q22
    # S6 = (p21-p11)(q11+q12)
    # S7 = (p12-p22)(q21+q22)
    
    # For each scalar product s, we need:
    #   - P-side: linear combo of the 4 elements of P (p11, p12, p21, p22)
    #   - Q-side: linear combo of the 4 elements of Q
    
    # P element (pi, pj) where pi, pj in {0,1} corresponds to:
    #   In the 2x2 block starting at (2*br, 2*bc) of the 4x4 matrix A:
    #   element at global position (2*br+pi, 2*bc+pj)
    #   U index: (2*br+pi)*4 + (2*bc+pj)
    
    # Q element (qi, qj) where qi, qj in {0,1} corresponds to:
    #   In the 2x2 block starting at (2*br, 2*bc) of the 4x4 matrix B:
    #   element at global position (2*br+qi, 2*bc+qj)
    #   V index: (2*br+qi)*4 + (2*bc+qj)
    
    # The result R = P*Q is a 2x2 matrix. R element (ri, rj) contributes to
    # C at position (2*br+ri, 2*bc+rj) ... wait, no.
    # 
    # Actually, the top-level product Mi gives a 2x2 result that contributes to
    # one of the C blocks. Let me reconsider.
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # So M1 contributes to C11 and C22, M2 to C21 and C22, etc.
    #
    # Let me define for each top product t, which C block(s) it contributes to and with what sign.
    
    # C block (cr, cc) covers rows 2*cr to 2*cr+1, cols 2*cc to 2*cc+1
    # W index: (k*4 + i) where k is col, i is row. So for C block (cr, cc),
    # element (ri, rj) within the block is at global (2*cr+ri, 2*cc+rj),
    # W index = (2*cc+rj)*4 + (2*cr+ri)
    
    # For each top product t, the contribution to C:
    # t=0 (M1): C11 (+1), C22 (+1)
    # t=1 (M2): C21 (+1), C22 (-1)
    # t=2 (M3): C12 (+1), C22 (+1)
    # t=3 (M4): C11 (+1), C21 (+1)
    # t=4 (M5): C11 (-1), C12 (+1)
    # t=5 (M6): C22 (+1)
    # t=6 (M7): C11 (+1)
    
    c_contributions = [
        [(0,0,1), (1,1,1)],   # M1
        [(1,0,1), (1,1,-1)],  # M2
        [(0,1,1), (1,1,1)],   # M3
        [(0,0,1), (1,0,1)],   # M4
        [(0,0,-1), (0,1,1)],  # M5
        [(1,1,1)],            # M6
        [(0,0,1)],            # M7
    ]
    
    # 2x2 Strassen scalar products: (P_coeffs, Q_coeffs)
    # P_coeffs: list of (pi, pj, sign) for p11, p12, p21, p22
    # Q_coeffs: list of (qi, qj, sign)
    strassen_2x2 = [
        # S1 = (p11+p22)(q11+q22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # S2 = (p21+p22)q11
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # S3 = p11(q12-q22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # S4 = p22(q21-q11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # S5 = (p11+p12)q22
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # S6 = (p21-p11)(q11+q12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # S7 = (p12-p22)(q21+q22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Now build U, V, W
    for t in range(7):
        A_blocks, B_blocks = top_products[t]
        
        for s in range(7):
            col = t * 7 + s
            P_coeffs, Q_coeffs = strassen_2x2[s]
            
            # For each (pi, pj, sign_p) in P_coeffs, and for each A block (ab_r, ab_c, sign_a):
            # The element is at global (2*ab_r + pi, 2*ab_c + pj)
            # U index: (2*ab_r + pi) * 4 + (2*ab_c + pj)
            # Coefficient: sign_p * sign_a
            for (pi, pj, sign_p) in P_coeffs:
                for (ab_r, ab_c, sign_a) in A_blocks:
                    row_u = (2 * ab_r + pi) * 4 + (2 * ab_c + pj)
                    U[row_u, col] += sign_p * sign_a
            
            # For each (qi, qj, sign_q) in Q_coeffs, and for each B block (bb_r, bb_c, sign_b):
            # The element is at global (2*bb_r + qi, 2*bb_c + qj)
            # V index: (2*bb_r + qi) * 4 + (2*bb_c + qj)
            # Coefficient: sign_q * sign_b
            for (qi, qj, sign_q) in Q_coeffs:
                for (bb_r, bb_c, sign_b) in B_blocks:
                    row_v = (2 * bb_r + qi) * 4 + (2 * bb_c + qj)
                    V[row_v, col] += sign_q * sign_b
            
            # W: for each C block contribution (cb_r, cb_c, sign_c):
            # The 2x2 Strassen result R has elements R(ri, rj) for ri, rj in {0,1}
            # R(ri, rj) = sum over the 7 scalar products of their contributions
            # 
            # Actually, the scalar product s contributes to specific R elements.
            # Let me reconsider. The 2x2 Strassen gives:
            # R11 = S1 + S4 - S5 + S7
            # R12 = S3 + S5
            # R21 = S2 + S4
            # R22 = S1 - S2 + S3 + S6
            #
            # So for scalar product s, its contribution to R(ri, rj) is determined.
            
            # Let me define for each s, which R elements it contributes to and with what sign.
            # s=0 (S1): R11(+1), R22(+1)
            # s=1 (S2): R21(+1), R22(-1)
            # s=2 (S3): R12(+1), R22(+1)
            # s=3 (S4): R11(+1), R21(+1)
            # s=4 (S5): R11(-1), R12(+1)
            # s=5 (S6): R22(+1)
            # s=6 (S7): R11(+1)
            
            r_contributions = [
                [(0,0,1), (1,1,1)],   # S1
                [(1,0,1), (1,1,-1)],  # S2
                [(0,1,1), (1,1,1)],   # S3
                [(0,0,1), (1,0,1)],   # S4
                [(0,0,-1), (0,1,1)],  # S5
                [(1,1,1)],            # S6
                [(0,0,1)],            # S7
            ]
            
            for (ri, rj, sign_r) in r_contributions[s]:
                for (cb_r, cb_c, sign_c) in c_contributions[t]:
                    # R(ri, rj) is at global position (2*cb_r + ri, 2*cb_c + rj)
                    # W index: (k*4 + i) = (2*cb_c + rj)*4 + (2*cb_r + ri)
                    row_w = (2 * cb_c + rj) * 4 + (2 * cb_r + ri)
                    W[row_w, col] += sign_r * sign_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
