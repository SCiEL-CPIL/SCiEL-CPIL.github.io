"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 4

    # Strassen decomposition for 4x4 matrix multiplication
    # We build a rank-49 decomposition by applying Strassen to 4x4 matrices
    # viewed as 2x2 blocks of 2x2 matrices.
    
    # 2x2 Strassen factors (for A*B where A,B are 2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A11-A10)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6

    # For 4x4: A is 4x4, split into 4 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Each block multiplication uses Strassen with 7 scalar multiplications.
    # Total rank = 7 * 7 = 49
    
    # Build U, V, W for the 4x4 x 4x4 -> 4x4 case
    # U has shape (16, 49) for 4x4 input
    # V has shape (16, 49) for 4x4 input
    # W has shape (16, 49) for 4x4 output
    
    # Index convention: for matrix of size dxd, element (i,j) maps to index i*d+j
    # For 4x4: A[i,j] -> index i*4+j
    
    U_44 = np.zeros((16, 49), dtype=np.float64)
    V_44 = np.zeros((16, 49), dtype=np.float64)
    W_44 = np.zeros((16, 49), dtype=np.float64)
    
    # Strassen coefficients for 2x2 case
    # Each M_k is a product of linear combinations of A elements and B elements
    # A elements indexed as (a00, a01, a10, a11) -> positions 0,1,2,3 in a 2x2
    # B elements indexed as (b00, b01, b10, b11) -> positions 0,1,2,3 in a 2x2
    
    # For the outer Strassen (4x4 as 2x2 of 2x2 blocks):
    # Block indices: 00, 01, 10, 11
    # Within each 2x2 block, local indices: 00, 01, 10, 11 -> local positions 0,1,2,3
    
    # Outer Strassen linear forms on blocks:
    # For A:
    #   L1_A = A00 + A11
    #   L2_A = A10 + A11
    #   L3_A = A00
    #   L4_A = A11
    #   L5_A = A00 + A10
    #   L6_A = A10 - A00
    #   L7_A = A11 - A10
    #
    # For B:
    #   L1_B = B00 + B11
    #   L2_B = B00
    #   L3_B = B10 - B11
    #   L4_B = B01 - B00
    #   L5_B = B11
    #   L6_B = B00 + B01
    #   L7_B = B10 + B11
    
    # Inner Strassen within each 2x2 block multiplication:
    # When we compute, e.g., (A00+A11) * (B00+B11), each is a 2x2 matrix.
    # The product of two 2x2 matrices uses 7 scalar multiplications via Strassen.
    # But wait - (A00+A11) is a 2x2 matrix, and (B00+B11) is a 2x2 matrix.
    # Their product is computed via 7 scalar multiplications.
    # So each outer Strassen term contributes 7 scalar multiplications.
    # Total = 7 * 7 = 49.
    
    # Let's define the inner Strassen for multiplying two 2x2 matrices X*Y:
    # x00,x01,x10,x11 are elements of X (local indices 0,1,2,3)
    # y00,y01,y10,y11 are elements of Y (local indices 0,1,2,3)
    # 
    # Inner Strassen:
    # S1 = (x00+x11)*(y00+y11)
    # S2 = (x10+x11)*y00
    # S3 = x00*(y10-y11)
    # S4 = x11*(y01-y00)
    # S5 = (x00+x10)*y11
    # S6 = (x10-x00)*(y00+y01)
    # S7 = (x11-x10)*(y10+y11)
    #
    # Result:
    # R00 = S1 + S4 - S5 - S7
    # R01 = S3 + S7
    # R10 = S2 + S5
    # R11 = S1 - S2 + S3 + S6

    # For each outer term t (0..6), the X matrix is a linear combination of A-blocks,
    # and Y matrix is a linear combination of B-blocks.
    # Each element of X is a sum of elements from specific A-blocks with coefficients.
    # Similarly for Y.
    
    # Let's define the outer linear forms more carefully.
    # Outer A-forms (each is a 2x2 matrix, i.e., 4 elements):
    outer_A_forms = []
    outer_B_forms = []
    
    # A block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Each block has local elements 0..3 (local 00,01,10,11)
    # Global index for block b, local element l: b*4 + l
    
    # For outer Strassen term t:
    # t=0: X = A00 + A11, Y = B00 + B11
    # t=1: X = A10 + A11, Y = B00
    # t=2: X = A00, Y = B10 - B11
    # t=3: X = A11, Y = B01 - B00
    # t=4: X = A00 + A10, Y = B11
    # t=5: X = A10 - A00, Y = B00 + B01
    # t=6: X = A11 - A10, Y = B10 + B11
    
    # Define which A-blocks contribute to X with what coefficient
    # And which B-blocks contribute to Y with what coefficient
    
    outer_def = [
        # (a_blocks: [(block_idx, coeff)], b_blocks: [(block_idx, coeff)])
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # t=0
        ([(2, 1), (3, 1)], [(0, 1)]),            # t=1
        ([(0, 1)], [(2, 1), (3, -1)]),           # t=2
        ([(3, 1)], [(1, 1), (0, -1)]),           # t=3
        ([(0, 1), (2, 1)], [(3, 1)]),            # t=4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # t=5
        ([(3, 1), (2, -1)], [(2, 1), (3, 1)]),   # t=6
    ]
    
    # Inner Strassen S-coefficients
    # For X elements (x00,x01,x10,x11) at local indices (0,1,2,3):
    # S1: x00*1 + x11*1, y00*1 + y11*1
    # S2: x10*1 + x11*1, y00*1
    # S3: x00*1, y10*1 + y11*(-1)
    # S4: x11*1, y01*1 + y00*(-1)
    # S5: x00*1 + x10*1, y11*1
    # S6: x10*1 + x00*(-1), y00*1 + y01*1
    # S7: x11*1 + x10*(-1), y10*1 + y11*1
    
    inner_x = [
        {0: 1, 3: 1},      # S1
        {2: 1, 3: 1},      # S2
        {0: 1},            # S3
        {3: 1},            # S4
        {0: 1, 2: 1},      # S5
        {2: 1, 0: -1},     # S6
        {3: 1, 2: -1},     # S7
    ]
    inner_y = [
        {0: 1, 3: 1},      # S1
        {0: 1},            # S2
        {2: 1, 3: -1},     # S3
        {1: 1, 0: -1},     # S4
        {3: 1},            # S5
        {0: 1, 1: 1},      # S6
        {2: 1, 3: 1},      # S7
    ]
    
    # Inner result coefficients:
    # R00: S1*1 + S4*1 + S5*(-1) + S7*(-1)
    # R01: S3*1 + S7*1
    # R10: S2*1 + S5*1
    # R11: S1*1 + S2*(-1) + S3*1 + S6*1
    
    inner_result = [
        {0: 1, 3: 1, 4: -1, 6: -1},  # R00 (local output 0)
        {2: 1, 6: 1},                 # R01 (local output 1)
        {1: 1, 4: 1},                 # R10 (local output 2)
        {0: 1, 1: -1, 2: 1, 5: 1},   # R11 (local output 3)
    ]
    
    col = 0
    for t in range(7):
        a_blocks, b_blocks = outer_def[t]
        for s in range(7):
            # This scalar multiplication corresponds to U_col * V_col contributing to W
            # U elements: for each local element l in X, sum over contributing A-blocks
            # V elements: for each local element l in Y, sum over contributing B-blocks
            
            # Build U contribution
            # For each local element lx in inner_x[s], we need to map to global A indices
            # For each (block_idx, coeff) in a_blocks, and for each local element:
            #   global_a_index = block_idx * 4 + local_element
            for lx, cx in inner_x[s].items():
                for (bidx, coeff) in a_blocks:
                    global_a = bidx * 4 + lx
                    U_44[global_a, col] += cx * coeff
            
            # Build V contribution
            for ly, cy in inner_y[s].items():
                for (bidx, coeff) in b_blocks:
                    global_b = bidx * 4 + ly
                    V_44[global_b, col] += cy * coeff
            
            # Build W contribution
            # The result of S_s goes into the 2x2 result at position determined by inner_result
            # But we need to figure out which output block and local position
            # The output C block position is determined by outer Strassen result combination
            # Wait, I need to reconsider.
            
            # Actually, the outer Strassen gives:
            # C00 = M0 + M3 - M4 - M6  (using t=0,3,4,6 with signs)
            # C01 = M2 + M6
            # C10 = M1 + M4
            # C11 = M0 - M1 + M2 + M5
            #
            # Where M_t is the 2x2 product from outer term t.
            # Each M_t is a 2x2 matrix with elements from inner Strassen.
            
            # So for outer term t, the 2x2 result R has local elements R00,R01,R10,R11.
            # These contribute to specific C-blocks with specific signs.
            
            outer_result_def = [
                # For each outer term t, list of (c_block_idx, local_out, sign)
                [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)],  # t=0: M0 -> C00(+1), C11(+1)
                [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, -1)], # t=1: M1 -> C10(+1), C11(-1)
                [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)],  # t=2: M2 -> C01(+1), C11(+1)
                [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)],  # t=3: M3 -> C00(+1)
                [(0, 0, -1), (0, 1, 0), (1, 0, 1), (1, 1, 0)], # t=4: M4 -> C00(-1), C10(+1)
                [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],  # t=5: M5 -> C11(+1)
                [(0, 0, -1), (0, 1, 1), (1, 0, 0), (1, 1, 0)], # t=6: M6 -> C00(-1), C01(+1)
            ]
            
            # For this (t, s), the inner S_s contributes to specific local output positions
            for (c_block, local_out, sign_t) in outer_result_def[t]:
                # Find which inner_result entries include S_s
                for local_r_idx, result_coeffs in enumerate(inner_result):
                    if s in result_coeffs:
                        sign_s = result_coeffs[s]
                        # The output goes to C block c_block, local element local_out
                        # But local_r_idx is the result position (0=R00, 1=R01, 2=R10, 3=R11)
                        # Wait, I need to match: inner_result[r_idx] gives coefficients for output r_idx
                        # S_s contributes to output r_idx with coefficient result_coeffs[s]
                        # So for each r_idx where s appears, we add to output at local position r_idx
                        pass
            
            # Let me redo this more carefully.
            # inner_result[r_idx] is a dict mapping s_idx -> coefficient
            # So for output local position r_idx, the contribution from S_s is inner_result[r_idx].get(s, 0)
            
            for r_idx in range(4):
                coeff_rs = inner_result[r_idx].get(s, 0)
                if coeff_rs != 0:
                    for (c_block, local_out, sign_t) in outer_result_def[t]:
                        # r_idx corresponds to local output position within the 2x2 result
                        # local_out should equal r_idx for the contribution to be nonzero
                        # Actually, outer_result_def[t] lists which (c_block, local_out) positions get contributions
                        # Let me reconsider the structure.
                        
                        # The 2x2 result from outer term t has 4 elements at local positions 0,1,2,3.
                        # These map to C-block c_block at local positions 0,1,2,3 respectively.
                        # The outer_result_def should just indicate which C-blocks get which of the 4 result elements.
                        pass
            
            # I think my outer_result_def is wrong. Let me redefine.
            # For outer term t, the 2x2 result M_t contributes to:
            # C00: if t=0 (+1), t=3 (+1), t=4 (-1), t=6 (-1) -> element [0,0] of M_t
            # C01: if t=2 (+1), t=6 (+1) -> element [0,1] of M_t
            # C10: if t=1 (+1), t=4 (+1) -> element [1,0] of M_t
            # C11: if t=0 (+1), t=1 (-1), t=2 (+1), t=5 (+1) -> element [1,1] of M_t
            
            # So the mapping is:
            # (t, local_result_pos) -> [(c_block, sign), ...]
            # local_result_pos: 0=M_t[0,0], 1=M_t[0,1], 2=M_t[1,0], 3=M_t[1,1]
            
            outer_contrib = {
                (0, 0): [(0, 1), (3, 1)],      # M0[0,0] -> C00(+1), C11(+1)
                (0, 1): [],                      # M0[0,1] -> nothing
                (0, 2): [],                      # M0[1,0] -> nothing
                (0, 3): [],                      # M0[1,1] -> nothing
                (1, 0): [],                      # M1[0,0] -> nothing
                (1, 1): [],                      # M1[0,1] -> nothing
                (1, 2): [(2, 1)],               # M1[1,0] -> C10(+1)
                (1, 3): [(3, -1)],              # M1[1,1] -> C11(-1)
                (2, 0): [],                      # M2[0,0] -> nothing
                (2, 1): [(1, 1)],               # M2[0,1] -> C01(+1)
                (2, 2): [],                      # M2[1,0] -> nothing
                (2, 3): [(3, 1)],               # M2[1,1] -> C11(+1)
                (3, 0): [(0, 1)],               # M3[0,0] -> C00(+1)
                (3, 1): [],                      # M3[0,1] -> nothing
                (3, 2): [],                      # M3[1,0] -> nothing
                (3, 3): [],                      # M3[1,1] -> nothing
                (4, 0): [(0, -1)],              # M4[0,0] -> C00(-1)
                (4, 1): [],                      # M4[0,1] -> nothing
                (4, 2): [(2, 1)],               # M4[1,0] -> C10(+1)
                (4, 3): [],                      # M4[1,1] -> nothing
                (5, 0): [],                      # M5[0,0] -> nothing
                (5, 1): [],                      # M5[0,1] -> nothing
                (5, 2): [],                      # M5[1,0] -> nothing
                (5, 3): [(3, 1)],               # M5[1,1] -> C11(+1)
                (6, 0): [(0, -1)],              # M6[0,0] -> C00(-1)
                (6, 1): [(1, 1)],               # M6[0,1] -> C01(+1)
                (6, 2): [],                      # M6[1,0] -> nothing
                (6, 3): [],                      # M6[1,1] -> nothing
            }
            
            # Now for this (t, s), for each local output position r_idx:
            for r_idx in range(4):
                coeff_rs = inner_result[r_idx].get(s, 0)
                if coeff_rs == 0:
                    continue
                contribs = outer_contrib.get((t, r_idx), [])
                for (c_block, sign_t) in contribs:
                    # Output goes to C block c_block, local position r_idx
                    global_c = c_block * 4 + r_idx
                    W_44[global_c, col] += coeff_rs * sign_t
            
            col += 1
    
    # Now we have U_44 (16x49), V_44 (16x49), W_44 (16x49) for 4x4x4
    # We need to extract for 3x4x4: n=3, m=4, p=4
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # For 3x4 matrix A: indices i in 0..2, j in 0..3 -> global index i*4+j (0..11)
    # For 4x4 matrix B: indices j in 0..3, k in 0..3 -> global index j*4+k (0..15)
    # For 3x4 output C: indices k in 0..3, i in 0..2 -> W index k*3+i (0..11)
    
    # The 4x4 decomposition uses:
    # U_44 index: i*4+j for i in 0..3, j in 0..3
    # V_44 index: j*4+k for j in 0..3, k in 0..3
    # W_44 index: k*4+i for k in 0..3, i in 0..3
    
    # For 3x4x4:
    # U_34 index: i*4+j for i in 0..2, j in 0..3 -> same as first 12 rows of U_44
    # V_34 index: j*4+k for j in 0..3, k in 0..3 -> same as all 16 rows of V_44
    # W_34 index: k*3+i for k in 0..3, i in 0..2 -> need to remap from W_44
    
    # Extract U: first 12 rows
    U_34 = U_44[:12, :]
    
    # Extract V: all 16 rows
    V_34 = V_44[:16, :]
    
    # Remap W: W_44 has index k*4+i, we need k*3+i
    W_34 = np.zeros((12, 49), dtype=np.float64)
    for k in range(4):
        for i in range(3):
            W_34[k * 3 + i, :] = W_44[k * 4 + i, :]
    
    # Filter out columns where U is all zeros (since n=3, if all 3 rows are zero, the term is useless)
    # Also filter if V or W is all zero
    
    non_zero_cols = []
    for c in range(49):
        if np.any(U_34[:, c] != 0) and np.any(V_34[:, c] != 0) and np.any(W_34[:, c] != 0):
            non_zero_cols.append(c)
    
    rank = len(non_zero_cols)
    U_final = U_34[:, non_zero_cols]
    V_final = V_34[:, non_zero_cols]
    W_final = W_34[:, non_zero_cols]
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}

# EVOLVE-BLOCK-END
