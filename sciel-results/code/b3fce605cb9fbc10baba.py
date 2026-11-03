for block in range(4):
    for entry in range(4):
        a_idx = block * 4 + entry
        U_4x4[a_idx][col] = A_coeffs[block] * u_2x2[entry]
