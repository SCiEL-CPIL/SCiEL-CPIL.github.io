import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank 65 decomposition: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x1x4)
    rank = 65
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for A (n x m): i * m + j
    # Indices for B (m x p): j * p + k
    # Indices for C (p x n): k * n + i
    
    # --- Part 1: Strassen 4x4x4 (First 4 columns of B) ---
    # We use the 7x7 Strassen structure.
    
    # Define the 7 Strassen patterns for 2x2 block multiplication.
    # Each pattern k defines:
    # X_k: 2x2 matrix of linear forms in A_blocks (A11, A12, A21, A22)
    # Y_k: 2x2 matrix of linear forms in B_blocks (B11, B12, B21, B22)
    # Z_k: 2x2 matrix of coefficients for C_blocks (C11, C12, C21, C22)
    
    # A_blocks indices in A (4x4):
    # A11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A21: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A22: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # B_blocks indices in B (4x4, first 4 cols):
    # B11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # B12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # B21: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # B22: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # C_blocks indices in C (4x4, first 4 cols):
    # C11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # C12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # C21: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # C22: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # We will generate the 49 terms.
    # The 7 outer Strassen products P1..P7.
    # For each Pk, we have a 2x2 matrix X and Y.
    # We then decompose X*Y using 7 inner Strassen terms.
    
    # Define the 7 outer patterns (X, Y, Z)
    # X and Y are 2x2 arrays of coefficients for the 4 A/B blocks.
    # Block indices: 0=A11/B11, 1=A12/B12, 2=A21/B21, 3=A22/B22
    
    outer_patterns = [
        # P1
        (
            [[1, 0], [0, 1]], # X11=1, X12=0, X21=0, X22=1 (A11+A22)
            [[1, 0], [0, 1]], # Y11=1, Y12=0, Y21=0, Y22=1 (B11+B22)
            [[1, 0], [0, 1]]  # Z11=1, Z12=0, Z21=0, Z22=1
        ),
        # P2
        (
            [[0, 0], [1, 1]], # X11=0, X12=0, X21=1, X22=1 (A21+A22)
            [[1, 0], [0, 0]], # Y11=1, Y12=0, Y21=0, Y22=0 (B11)
            [[0, 0], [1, 0]]  # Z11=0, Z12=0, Z21=1, Z22=0
        ),
        # P3
        (
            [[1, 0], [0, 0]], # X11=1, X12=0, X21=0, X22=0 (A11)
            [[0, 1], [0, -1]],# Y11=0, Y12=1, Y21=0, Y22=-1 (B12-B22)
            [[0, 1], [0, 0]]  # Z11=0, Z12=1, Z21=0, Z22=0
        ),
        # P4
        (
            [[0, 0], [0, 1]], # X11=0, X12=0, X21=0, X22=1 (A22)
            [[-1, 0], [1, 0]],# Y11=-1, Y12=0, Y21=1, Y22=0 (B21-B11)
            [[1, 0], [1, 0]]  # Z11=1, Z12=0, Z21=1, Z22=0
        ),
        # P5
        (
            [[1, 1], [0, 0]], # X11=1, X12=1, X21=0, X22=0 (A11+A12)
            [[0, 0], [0, 1]], # Y11=0, Y12=0, Y21=0, Y22=1 (B22)
            [[-1, 1], [0, 0]] # Z11=-1, Z12=1, Z21=0, Z22=0
        ),
        # P6
        (
            [[-1, 0], [1, 0]],# X11=-1, X12=0, X21=1, X22=0 (A21-A11)
            [[1, 1], [0, 0]], # Y11=1, Y12=1, Y21=0, Y22=0 (B11+B12)
            [[0, 0], [0, 1]]  # Z11=0, Z12=0, Z21=0, Z22=1
        ),
        # P7
        (
            [[0, 1], [0, -1]],# X11=0, X12=1, X21=0, X22=-1 (A12-A22)
            [[0, 0], [1, 1]], # Y11=0, Y12=0, Y21=1, Y22=1 (B21+B22)
            [[1, 0], [0, 0]]  # Z11=1, Z12=0, Z21=0, Z22=0
        )
    ]
    
    # Inner Strassen patterns for 2x2 scalar multiplication
    # s1..s7
    # X, Y are 2x2 matrices of scalars (linear forms)
    # We need to map the "linear forms" to the actual A/B indices.
    
    # Helper to get the list of (index, coeff) for a block
    def get_block_indices(block_id, is_A):
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22
        # is_A: True for A, False for B
        if is_A:
            # A is 4x4. m=4.
            # A11: rows 0,1 cols 0,1
            # A12: rows 0,1 cols 2,3
            # A21: rows 2,3 cols 0,1
            # A22: rows 2,3 cols 2,3
            r_start = (block_id // 2) * 2
            c_start = (block_id % 2) * 2
            indices = []
            for r in range(2):
                for c in range(2):
                    i = r_start + r
                    j = c_start + c
                    indices.append(i * m + j)
            return indices
        else:
            # B is 4x4 (first 4 cols). p=5, but we use first 4.
            # B11: rows 0,1 cols 0,1
            # B12: rows 0,1 cols 2,3
            # B21: rows 2,3 cols 0,1
            # B22: rows 2,3 cols 2,3
            r_start = (block_id // 2) * 2
            c_start = (block_id % 2) * 2
            indices = []
            for r in range(2):
                for c in range(2):
                    j = r_start + r
                    k = c_start + c
                    indices.append(j * p + k)
            return indices

    # Helper to get the list of (index, coeff) for a C block
    def get_c_block_indices(block_id):
        # C is 4x4 (first 4 cols). n=4.
        # C11: rows 0,1 cols 0,1 (k=0,1; i=0,1)
        # C12: rows 0,1 cols 2,3 (k=2,3; i=0,1)
        # C21: rows 2,3 cols 0,1 (k=0,1; i=2,3)
        # C22: rows 2,3 cols 2,3 (k=2,3; i=2,3)
        # Note: W index is k * n + i
        r_start = (block_id // 2) * 2 # k start
        c_start = (block_id % 2) * 2  # i start
        indices = []
        for r in range(2):
            for c in range(2):
                k = r_start + r
                i = c_start + c
                indices.append(k * n + i)
        return indices

    # Inner Strassen terms for 2x2
    # s1 = (X00+X11)(Y00+Y11)
    # s2 = (X10+X11)Y00
    # s3 = X00(Y01-Y11)
    # s4 = X11(Y10-Y00)
    # s5 = (X00+X01)Y11
    # s6 = (X10-X00)(Y00+Y01)
    # s7 = (X01-X11)(Y10+Y11)
    
    # Z reconstruction:
    # Z00 = s1 + s4 - s5 + s7
    # Z01 = s3 + s5
    # Z10 = s2 + s4
    # Z11 = s1 - s2 + s3 + s6
    
    inner_defs = [
        # (X00, X01, X10, X11, Y00, Y01, Y10, Y11, Z00, Z01, Z10, Z11)
        # s1
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # s2
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        # s3
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),
        # s4
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # s5
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # s6
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # s7
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0)
    ]
    
    col = 0
    for outer_idx, (X, Y, Z) in enumerate(outer_patterns):
        # X, Y are 2x2 lists of block coefficients
        # We need to expand each of the 7 inner terms.
        for inner_idx, (x00, x01, x10, x11, y00, y01, y10, y11, z00, z01, z10, z11) in enumerate(inner_defs):
            # Construct the linear forms for U and V
            
            # U linear form:
            # U = x00 * X00 + x01 * X01 + x10 * X10 + x11 * X11
            # X00 is a block of A. X[0][0] is the coeff for A_block_0?
            # In outer_patterns, X is [[c00, c01], [c10, c11]] where c_ij is coeff for A_block_ij?
            # No, in my definition above:
            # X[0][0] corresponds to A11 (block 0)
            # X[0][1] corresponds to A12 (block 1)
            # X[1][0] corresponds to A21 (block 2)
            # X[1][1] corresponds to A22 (block 3)
            
            # So X_block_coeffs = [X[0][0], X[0][1], X[1][0], X[1][1]]
            # And we combine them with x00, x01, x10, x11?
            # Wait, the inner Strassen formula uses the *entries* of the 2x2 matrix X.
            # The entries of X are themselves linear combinations of A_blocks.
            # Let X_entry_00 = X[0][0] * A11 + X[0][1] * A12 + ...?
            # No, in the outer definition, X is the 2x2 matrix of *blocks*.
            # X[0][0] is the coefficient for A11 in the top-left block of the "X matrix" for this Strassen product.
            # So, the top-left entry of the X matrix (call it $X_{TL}$) is:
            # $X_{TL} = X[0][0] \cdot A_{11} + X[0][1] \cdot A_{12} + X[1][0] \cdot A_{21} + X[1][1] \cdot A_{22}$?
            # No, that's not right.
            # The outer Strassen product P_k is defined as $X_k Y_k$.
            # $X_k$ is a 2x2 matrix where each entry is a 2x2 matrix (linear combo of A_blocks).
            # In my `outer_patterns`, I defined X as a 2x2 array of *scalars* (coeffs).
            # This implies that for P_k, the X matrix is formed by:
            # $X_{TL} = \text{coeff} \cdot A_{11}$?
            # Let's re-read my outer_patterns definition.
            # For P1: X = [[1, 0], [0, 1]].
            # This means $X_{TL} = 1 \cdot A_{11}$, $X_{TR} = 0 \cdot A_{12}$, etc?
            # No, $X_{TL}$ should be $A_{11} + A_{22}$?
            # In P1, $X = A_{11} + A_{22}$ is a scalar sum? No, $A_{11}$ is a 2x2 matrix.
            # $X_{TL}$ is the top-left 2x2 block of the X matrix.
            # If $X = A_{11} + A_{22}$, then $X_{TL} = (A_{11} + A_{22})_{TL}$?
            # No, the Strassen formula operates on the *blocks*.
            # $P_1 = (A_{11} + A_{22}) (B_{11} + B_{22})$.
            # Here, $A_{11} + A_{22}$ is a 2x2 matrix (sum of two 2x2 matrices).
            # So $X$ in $X Y$ is the 2x2 matrix $A_{11} + A_{22}$.
            # So $X_{TL} = (A_{11} + A_{22})_{0,0} = A_{11,0,0} + A_{22,0,0}$.
            # $X_{TR} = (A_{11} + A_{22})_{0,1} = A_{11,0,1} + A_{22,0,1}$.
            # Etc.
            
            # So, for each outer product, we have 4 blocks of A contributing to the X matrix entries.
            # My `outer_patterns` X matrix entries are coefficients for the *entire* A blocks?
            # If X = [[1, 0], [0, 1]], it means:
            # $X_{TL} = 1 \cdot A_{11} + 0 \cdot A_{12} + 0 \cdot A_{21} + 0 \cdot A_{22}$?
            # No, $X_{TL}$ is a 2x2 matrix.
            # If I say $X_{TL} = A_{11}$, then the coefficients for A11's entries are 1, and for others 0.
            # If I say $X_{TL} = A_{11} + A_{22}$, then coefficients for A11 and A22 are 1.
            
            # So, for P1, $X_{TL} = A_{11} + A_{22}$.
            # $X_{TR} = A_{12} + A_{22}$? No.
            # $X = A_{11} + A_{22}$ is a single 2x2 matrix.
            # So $X_{TL} = (A_{11} + A_{22})_{0,0}$.
            # $X_{TR} = (A_{11} + A_{22})_{0,1}$.
            # $X_{BL} = (A_{11} + A_{22})_{1,0}$.
            # $X_{BR} = (A_{11} + A_{22})_{1,1}$.
            
            # So, the "X matrix" in the inner Strassen step is actually the 2x2 matrix formed by the sum/difference of the A blocks.
            # My `outer_patterns` X definition `[[1, 0], [0, 1]]` is ambiguous.
            # It should represent the linear combination of the 4 A blocks that forms the *entire* X matrix.
            # Since the A blocks are 2x2, and the X matrix is 2x2, the mapping is direct.
            # $X_{TL}$ comes from $A_{11}$ and $A_{22}$?
            # No, $A_{11}$ is a 2x2 matrix. $A_{22}$ is a 2x2 matrix.
            # $X = A_{11} + A_{22}$ is a 2x2 matrix.
            # So $X_{TL} = A_{11,0,0} + A_{22,0,0}$.
            # $X_{TR} = A_{11,0,1} + A_{22,0,1}$.
            # $X_{BL} = A_{11,1,0} + A_{22,1,0}$.
            # $X_{BR} = A_{11,1,1} + A_{22,1,1}$.
            
            # So, for each of the 4 entries of the X matrix ($X_{TL}, X_{TR}, X_{BL}, X_{BR}$), we have a linear form in the 16 A entries.
            # Then, the inner Strassen step combines these 4 linear forms.
            
            # I will implement this by:
            # 1. For each of the 4 positions in the X matrix (00, 01, 10, 11), determine the linear form in A.
            # 2. For each of the 4 positions in the Y matrix (00, 01, 10, 11), determine the linear form in B.
            # 3. Apply the inner Strassen formulas to these 4+4 linear forms.
            
            # Let's define the linear forms for X and Y entries.
            # X_entry[pos] = sum(coeff * A_block)
            # For P1: X = A11 + A22.
            # X[0,0] (TL) = A11[0,0] + A22[0,0]
            # X[0,1] (TR) = A11[0,1] + A22[0,1]
            # X[1,0] (BL) = A11[1,0] + A22[1,0]
            # X[1,1] (BR) = A11[1,1] + A22[1,1]
            
            # My `outer_patterns` X is `[[1, 0], [0, 1]]`.
            # This should be interpreted as:
            # X[0,0] (TL) uses A11 (coeff 1) and A22 (coeff 0)?
            # No, that would mean X = diag(A11, A22).
            # But P1 is (A11+A22)(B11+B22).
            # So X = A11 + A22.
            # This means X[0,0] = A11[0,0] + A22[0,0].
            # So the coefficient for A11 in X[0,0] is 1.
            # The coefficient for A22 in X[0,0] is 1.
            
            # So, for each outer pattern, I need to define the 2x2 matrix of *block coefficients* for the *entire* X matrix.
            # Let's call it `X_blocks`.
            # For P1: `X_blocks` = `[[1, 0], [0, 1]]`? No.
            # If `X_blocks[0,0] = 1`, does it mean X[0,0] = A11[0,0]?
            # And `X_blocks[1,1] = 1` means X[1,1] = A22[1,1]?
            # Then X = diag(A11, A22).
            # But we want X = A11 + A22.
            # So X[0,0] = A11[0,0] + A22[0,0].
            # This requires X_blocks[0,0] to have coeff 1 for A11 AND 1 for A22?
            # No, the block structure is:
            # A = [[A11, A12], [A21, A22]]
            # X = A11 + A22 is a 2x2 matrix.
            # X[0,0] = A11[0,0] + A22[0,0]
            # X[0,1] = A11[0,1] + A22[0,1]
            # X[1,0] = A11[1,0] + A22[1,0]
            # X[1,1] = A11[1,1] + A22[1,1]
            
            # So, for each entry (r,c) of the X matrix, the linear form is:
            # L_X[r,c] = sum_{b in blocks} coeff[b] * A_b[r,c]
            
            # For P1, coeff[0] (A11) = 1, coeff[3] (A22) = 1. Others 0.
            # So L_X[0,0] = 1*A11[0,0] + 1*A22[0,0].
            
            # I will define `X_coeff` as a 4-element vector [c0, c1, c2, c3] for A11, A12, A21, A22.
            # And `Y_coeff` as a 4-element vector for B11, B12, B21, B22.
            
            outer_coeffs = [
                ([1, 0, 0, 1], [1, 0, 0, 1]), # P1
                ([0, 0, 1, 1], [1, 0, 0, 0]), # P2
                ([1, 0, 0, 0], [0, 1, 0, -1]),# P3
                ([0, 0, 0, 1], [-1, 0, 1, 0]),# P4
                ([1, 1, 0, 0], [0, 0, 0, 1]), # P5
                ([-1, 0, 1, 0], [1, 1, 0, 0]),# P6
                ([0, 1, 0, -1], [0, 0, 1, 1]) # P7
            ]
            
            for xc, yc in outer_coeffs:
                # Generate 7 inner terms
                for inner_idx, (x00, x01, x10, x11, y00, y01, y10, y11, z00, z01, z10, z11) in enumerate(inner_defs):
                    # Construct U linear form
                    # U = x00 * L_X[0,0] + x01 * L_X[0,1] + x10 * L_X[1,0] + x11 * L_X[1,1]
                    # L_X[r,c] = sum_{b} xc[b] * A_b[r,c]
                    
                    U_lin = {}
                    for r in range(2):
                        for c in range(2):
                            # Get coeff for this X entry
                            if r == 0 and c == 0: coeff_x = x00
                            elif r == 0 and c == 1: coeff_x = x01
                            elif r == 1 and c == 0: coeff_x = x10
                            else: coeff_x = x11
                            
                            if coeff_x == 0: continue
                            
                            for b in range(4):
                                if xc[b] == 0: continue
                                # Get indices for A_block b at pos (r,c)
                                # A_block b is 2x2.
                                # r, c are local indices in the 2x2 block.
                                # Global A index:
                                # A11 (b=0): rows 0,1 cols 0,1
                                # A12 (b=1): rows 0,1 cols 2,3
                                # A21 (b=2): rows 2,3 cols 0,1
                                # A22 (b=3): rows 2,3 cols 2,3
                                r_start = (b // 2) * 2
                                c_start = (b % 2) * 2
                                i = r_start + r
                                j = c_start + c
                                idx = i * m + j
                                U_lin[idx] = U_lin.get(idx, 0) + coeff_x * xc[b]
                    
                    # Construct V linear form
                    V_lin = {}
                    for r in range(2):
                        for c in range(2):
                            if r == 0 and c == 0: coeff_y = y00
                            elif r == 0 and c == 1: coeff_y = y01
                            elif r == 1 and c == 0: coeff_y = y10
                            else: coeff_y = y11
                            
                            if coeff_y == 0: continue
                            
                            for b in range(4):
                                if yc[b] == 0: continue
                                r_start = (b // 2) * 2
                                c_start = (b % 2) * 2
                                j = r_start + r
                                k = c_start + c
                                idx = j * p + k
                                V_lin[idx] = V_lin.get(idx, 0) + coeff_y * yc[b]
                    
                    # Construct W linear form
                    # W = z00 * L_Z[0,0] + ...
                    # L_Z[r,c] = sum_{b} zc[b] * C_b[r,c]
                    # But zc depends on the outer pattern.
                    # Wait, the Z coefficients are fixed for the outer pattern.
                    # I need to define Z_coeffs for each outer pattern.
                    
                    # P1: Z = C11 + C22? No.
                    # C11 = P1 + P4 - P5 + P7
                    # C22 = P1 - P2 + P3 + P6
                    # So for P1, Z contributes to C11 and C22.
                    # Z[0,0] (TL) = C11[0,0]?
                    # No, Z is the 2x2 matrix of C blocks.
                    # Z[0,0] is the C11 block.
                    # Z[0,1] is the C12 block.
                    # Z[1,0] is the C21 block.
                    # Z[1,1] is the C22 block.
                    
                    # So for P1, Z[0,0] has coeff 1, Z[1,1] has coeff 1. Others 0.
                    
                    z_coeffs = [
                        [1, 0, 0, 1], # P1: C11, C22
                        [0, 0, 1, 0], # P2: C21
                        [0, 1, 0, 0], # P3: C12
                        [1, 0, 1, 0], # P4: C11, C21
                        [-1, 1, 0, 0],# P5: C11, C12
                        [0, 0, 0, 1], # P6: C22
                        [1, 0, 0, 0]  # P7: C11
                    ]
                    
                    zc = z_coeffs[outer_idx]
                    
                    W_lin = {}
                    for r in range(2):
                        for c in range(2):
                            if r == 0 and c == 0: coeff_z = z00
                            elif r == 0 and c == 1: coeff_z = z01
                            elif r == 1 and c == 0: coeff_z = z10
                            else: coeff_z = z11
                            
                            if coeff_z == 0: continue
                            
                            for b in range(4):
                                if zc[b] == 0: continue
                                r_start = (b // 2) * 2
                                c_start = (b % 2) * 2
                                k = r_start + r
                                i = c_start + c
                                idx = k * n + i
                                W_lin[idx] = W_lin.get(idx, 0) + coeff_z * zc[b]
                    
                    # Add to U, V, W
                    for idx, val in U_lin.items():
                        U[idx][col] = val
                    for idx, val in V_lin.items():
                        V[idx][col] = val
                    for idx, val in W_lin.items():
                        W[idx][col] = val
                    col += 1
    
    # --- Part 2: Schoolbook for 5th column (Rank 16) ---
    # A is 4x4. B5 is 4x1. C5 is 4x1.
    # C5[i] = sum_j A[i,j] * B5[j]
    # W index for C5: k=4, i=0..3 -> 4*4 + i = 16 + i
    # U index: i*m + j
    # V index: j*p + 4
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
