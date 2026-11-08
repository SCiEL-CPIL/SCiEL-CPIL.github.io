import time
import numpy as np

def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer entries.
    """
    N, M, P = 4, 4, 4
    RANK = 49
    
    U = np.zeros((N * M, RANK), dtype=int)
    V = np.zeros((M * P, RANK), dtype=int)
    W = np.zeros((P * N, RANK), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # X, Y, Z are 2x2 matrices.
    # Format: list of (row, col, coeff)
    
    strassen_2x2 = [
        # m1 = (a00 + a11) * (b00 + b11)
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # m2 = (a10 + a11) * b00
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # m3 = a00 * (b01 - b11)
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # m4 = a11 * (b10 - b00)
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # m5 = (a00 + a01) * b11
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # m6 = (a10 - a00) * (b00 + b01)
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # m7 = (a01 - a11) * (b10 + b11)
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    # Block-level Strassen products for 4x4 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    block_strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # M2 = (A10 + A11) * B00
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # M3 = A00 * (B01 - B11)
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # M4 = A11 * (B10 - B00)
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # M5 = (A00 + A01) * B11
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # M6 = (A10 - A00) * (B00 + B01)
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # M7 = (A01 - A11) * (B10 + B11)
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    def block_elem_to_flat(block_r, block_c, elem_r, elem_c):
        """
        Map block position (block_r, block_c) and element position (elem_r, elem_c)
        within that 2x2 block to the flat index in the 4x4 matrix.
        """
        row = 2 * block_r + elem_r
        col = 2 * block_c + elem_c
        return row * 4 + col
    
    term_idx = 0
    for b_idx, (X_blocks, Y_blocks, Z_blocks) in enumerate(block_strassen):
        for s_idx, (x_form, y_form, z_form) in enumerate(strassen_2x2):
            # U linear form
            for er, ec, coeff_s in x_form:
                for br, bc, coeff_x in X_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    U[flat_idx, term_idx] += coeff_s * coeff_x
            
            # V linear form
            for er, ec, coeff_s in y_form:
                for br, bc, coeff_y in Y_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    V[flat_idx, term_idx] += coeff_s * coeff_y
            
            # W linear form
            for er, ec, coeff_z in z_form:
                for br, bc, coeff_zb in Z_blocks:
                    i = 2 * br + er
                    k = 2 * bc + ec
                    flat_idx = k * 4 + i
                    W[flat_idx, term_idx] += coeff_z * coeff_zb
            
            term_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
