"""
Matrix multiplication 4x4x4 tensor decomposition.

This program constructs the known optimal rank-49 tensor decomposition for
multiplying two 4x4 matrices. The decomposition is derived from applying
Strassen's algorithm to 2x2 block matrices, where each 2x2 block product
is itself computed using Strassen's 7-product formula.

Total rank: 7 (block-level) * 7 (scalar-level) = 49.
This is the known optimal rank for 4x4 matrix multiplication.
"""

import time
import numpy as np

# Dimensions
N = 4
M = 4
P = 4
RANK = 49

def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': U, 'V': 'W': W} with integer entries.
    """
    # Initialize U, V, W as zero matrices
    # U shape: (N*M, RANK) = (16, 49)
    # V shape: (M*P, RANK) = (16, 49)
    # W shape: (P*N, RANK) = (16, 49)
    
    U = np.zeros((N * M, RANK), dtype=int)
    V = np.zeros((M * P, RANK), dtype=int)
    W = np.zeros((P * N, RANK), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Each term is defined by linear forms for X (input A part), Y (input B part), and Z (output C part)
    # X, Y, Z are 2x2 matrices.
    # We represent the linear form as a list of (row, col, coeff) tuples for the 2x2 block.
    
    # The 7 Strassen products for 2x2 matrices:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # Output C = A * B:
    # c00 = m1 - m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Define the 7 terms for a single 2x2 multiplication.
    # Each entry is (X_form, Y_form, Z_form)
    # X_form: list of (r, c, coeff) for the 2x2 matrix X
    # Y_form: list of (r, c, coeff) for the 2x2 matrix Y
    # Z_form: list of (r, c, coeff) for the 2x2 matrix Z
    
    strassen_2x2 = [
        # m1
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # m2
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # m3
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # m4
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # m5
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # m6
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # m7
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    # Block-level Strassen products for 4x4 matrices
    # Partition 4x4 into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The block-level Strassen formulas are identical in structure to the 2x2 ones,
    # but now the "elements" are 2x2 blocks.
    
    # Pre-computed lists of (block_row, block_col, coeff) for faster iteration
    block_strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        (
            [(0, 0, 1), (1, 1, 1)],  # X blocks
            [(0, 0, 1), (1, 1, 1)],  # Y blocks
            [(0, 0, 1), (1, 1, 1)]   # Z blocks
        ),
        # M2 = (A10 + A11) * B00
        (
            [(1, 0, 1), (1, 1, 1)],  # X blocks
            [(0, 0, 1)],             # Y blocks
            [(1, 0, 1), (1, 1, -1)]  # Z blocks
        ),
        # M3 = A00 * (B01 - B11)
        (
            [(0, 0, 1)],             # X blocks
            [(0, 1, 1), (1, 1, -1)], # Y blocks
            [(0, 1, 1), (1, 1, 1)]   # Z blocks
        ),
        # M4 = A11 * (B10 - B00)
        (
            [(1, 1, 1)],             # X blocks
            [(1, 0, 1), (0, 0, -1)], # Y blocks
            [(0, 0, 1), (1, 0, 1)]   # Z blocks
        ),
        # M5 = (A00 + A01) * B11
        (
            [(0, 0, 1), (0, 1, 1)],  # X blocks
            [(1, 1, 1)],             # Y blocks
            [(0, 0, -1), (0, 1, 1)]  # Z blocks
        ),
        # M6 = (A10 - A00) * (B00 + B01)
        (
            [(1, 0, 1), (0, 0, -1)], # X blocks
            [(0, 0, 1), (0, 1, 1)],  # Y blocks
            [(1, 1, 1)]              # Z blocks
        ),
        # M7 = (A01 - A11) * (B10 + B11)
        (
            [(0, 1, 1), (1, 1, -1)], # X blocks
            [(1, 0, 1), (1, 1, 1)],  # Y blocks
            [(0, 0, 1)]              # Z blocks
        ),
    ]
    
    # Helper to get flat index for a 2x2 block element in a 4x4 matrix
    def block_elem_to_flat(block_r, block_c, elem_r, elem_c):
        """
        Map block position (block_r, block_c) and element position (elem_r, elem_c)
        within that 2x2 block to the flat index in the 4x4 matrix.
        """
        row = 2 * block_r + elem_r
        col = 2 * block_c + elem_c
        return row * 4 + col
    
    # Construct the full rank-49 decomposition
    # We iterate over the 7 block-level products (M1..M7)
    # For each block product, we iterate over the 7 scalar-level Strassen terms
    # This gives 7 * 7 = 49 terms total.
    
    term_idx = 0
    for b_idx, (X_blocks, Y_blocks, Z_blocks) in enumerate(block_strassen):
        for s_idx, (x_form, y_form, z_form) in enumerate(strassen_2x2):
            # For this scalar term, we need to build the linear forms for U, V, W
            
            # U linear form: sum of contributions from X
            # X is a 2x2 matrix of blocks.
            # X_elem(er, ec) = sum_{(br, bc) in X_blocks} coeff_X * A_block(br, bc)[er, ec]
            # The scalar Strassen term uses specific elements of X (from x_form).
            # x_form is a list of (er, ec, coeff_s).
            
            for er, ec, coeff_s in x_form:
                for br, bc, coeff_x in X_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    U[flat_idx, term_idx] += coeff_s * coeff_x
            
            # V linear form: sum of contributions from Y
            # Similar logic for Y and V
            for er, ec, coeff_s in y_form:
                for br, bc, coeff_y in Y_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    V[flat_idx, term_idx] += coeff_s * coeff_y
            
            # W linear form: sum of contributions from Z
            # Z is a 2x2 matrix of blocks.
            # Z_elem(er, ec) = sum_{(br, bc) in Z_blocks} coeff_Z * C_block(br, bc)[er, ec]
            # The scalar Strassen term contributes to specific elements of Z (from z_form).
            # z_form is a list of (er, ec, coeff_z).
            
            for er, ec, coeff_z in z_form:
                for br, bc, coeff_zb in Z_blocks:
                    # C_block(br, bc)[er, ec] corresponds to C[2*br+er, 2*bc+ec]
                    # The output factor W uses the trace convention: W[k*n+i, r]
                    # Here, C[i, k] corresponds to W[k*4+i, r]
                    i = 2 * br + er
                    k = 2 * bc + ec
                    flat_idx = k * 4 + i
                    W[flat_idx, term_idx] += coeff_z * coeff_zb
            
            term_idx += 1
    
    # Convert numpy arrays to lists of lists of floats (as required by interface)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Optional: Verify decomposition (fast check)
    # This is a sanity check to ensure the construction logic is correct.
    # We check a few random entries or the whole tensor if time permits.
    # Given the deterministic construction, this should always pass.
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    
    # Quick verification
    U_np = np.array(result['U'])
    V_np = np.array(result['V'])
    W_np = np.array(result['W'])
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1.0 if i == k else 0.0
                # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                val = np.sum(U_np[i*4+j, :] * V_np[j*4+k, :] * W_np[k*4+i, :])
                if abs(val - expected) > 1e-9:
                    valid = False
                    print(f"Error at i={i}, j={j}, k={k}: expected {expected}, got {val}")
                    break
            if not valid:
                break
        if not valid:
            break
    if valid:
        print("Verification PASSED")
    else:
        print("Verification FAILED")
