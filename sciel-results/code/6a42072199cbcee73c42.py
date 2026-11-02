# For 2x2 matrices A, B with C = A*B:
# M1 = (A00 + A11) * (B00 + B11)
# M2 = (A10 + A11) * B00
# M3 = A00 * (B10 - B11)
# M4 = A11 * (B01 - B00)
# M5 = (A00 + A01) * B11
# M6 = (A10 - A00) * (B00 + B01)
# M7 = (A01 - A11) * (B10 + B11)
#
# C00 = M1 - M3 - M7 + M6
# C01 = M5 + M7
# C10 = M2 + M4
# C11 = M1 - M2 - M4 + M5
