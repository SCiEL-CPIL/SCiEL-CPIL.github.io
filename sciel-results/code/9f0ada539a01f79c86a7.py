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
    Returns {'U': U, 'V': V, 'W': W} with integer entries.
    """
    start_time = time.time()
    
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
    
    block_strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        (
            {(0, 0): 1, (1, 1): 1},  # X blocks: A00, A11
            {(0, 0): 1, (1, 1): 1},  # Y blocks: B00, B11
            {(0, 0): 1, (1, 1): 1}   # Z blocks: C00, C11
        ),
        # M2 = (A10 + A11) * B00
        (
            {(1, 0): 1, (1, 1): 1},  # X blocks: A10, A11
            {(0, 0): 1},             # Y blocks: B00
            {(1, 0): 1, (1, 1): -1}  # Z blocks: C10, C11
        ),
        # M3 = A00 * (B01 - B11)
        (
            {(0, 0): 1},             # X blocks: A00
            {(0, 1): 1, (1, 1): -1}, # Y blocks: B01, B11
            {(0, 1): 1, (1, 1): 1}   # Z blocks: C01, C11
        ),
        # M4 = A11 * (B10 - B00)
        (
            {(1, 1): 1},             # X blocks: A11
            {(1, 0): 1, (0, 0): -1}, # Y blocks: B10, B00
            {(0, 0): 1, (1, 0): 1}   # Z blocks: C00, C10
        ),
        # M5 = (A00 + A01) * B11
        (
            {(0, 0): 1, (0, 1): 1},  # X blocks: A00, A01
            {(1, 1): 1},             # Y blocks: B11
            {(0, 0): -1, (0, 1): 1}  # Z blocks: C00, C01
        ),
        # M6 = (A10 - A00) * (B00 + B01)
        (
            {(1, 0): 1, (0, 0): -1}, # X blocks: A10, A00
            {(0, 0): 1, (0, 1): 1},  # Y blocks: B00, B01
            {(1, 1): 1}              # Z blocks: C11
        ),
        # M7 = (A01 - A11) * (B10 + B11)
        (
            {(0, 1): 1, (1, 1): -1}, # X blocks: A01, A11
            {(1, 0): 1, (1, 1): 1},  # Y blocks: B10, B11
            {(0, 0): 1}              # Z blocks: C00
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
            for er, ec, coeff_s in x_form:
                for (br, bc), coeff_x in X_blocks.items():
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    U[flat_idx, term_idx] += coeff_s * coeff_x
            
            # V linear form: sum of contributions from Y
            for er, ec, coeff_s in y_form:
                for (br, bc), coeff_y in Y_blocks.items():
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    V[flat_idx, term_idx] += coeff_s * coeff_y
            
            # W linear form: sum of contributions from Z
            for er, ec, coeff_z in z_form:
                for (br, bc), coeff_zb in Z_blocks.items():
                    # C_block(br, bc)[er, ec] corresponds to C[2*br+er, 2*bc+ec]
                    # The output factor W uses the trace convention: W[k*n+i, r]
                    # Here, C[i, k] corresponds to W[k*4+i, r]
                    i = 2 * br + er
                    k = 2 * bc + ec
                    flat_idx = k * 4 + i
                    W[flat_idx, term_idx] += coeff_z * coeff_zb
            
            term_idx += 1
            
            # Time guard (though this loop is very fast)
            if time.time() - start_time > 100:
                break
        if time.time() - start_time > 100:
            break
    
    # Convert numpy arrays to lists of lists of floats (as required by interface)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
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
