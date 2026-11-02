def add_to_w(k, i, val):
    W[k*n + i][col] += val

# P1
add_to_w(k0, i0, 1)
add_to_w(k1, i0, 1)
add_to_w(k0, i1, 1)
add_to_w(k1, i1, 1)

# P2 (M2) -> C21 (C10)
add_to_w(k0, i1, 1)

# P3 (M3) -> C12 (C01)
add_to_w(k1, i0, 1)

# P4 (M4) -> C21 (C10) and C11 (C00)?
# M4 = A22(B21-B11).
# A22 is row 1. B21 is col 0. B11 is col 0.
# So it contributes to C10.
# Formula: C21 = M2 + M4.
# So C10 gets M4.
# Does C00 get M4? No.
add_to_w(k0, i1, 1)

# P5 (M5) -> C12 (C01)
# Formula: C12 = M3 + M5.
add_to_w(k1, i0, 1)

# P6 (M6) -> C22 (C11)
# Formula: C22 = M1 - M2 + M3 + M6.
add_to_w(k1, i1, 1)

# P7 (M7) -> C11 (C00)
# Formula: C11 = M1 + M4 - M5 + M7.
add_to_w(k0, i0, 1)
