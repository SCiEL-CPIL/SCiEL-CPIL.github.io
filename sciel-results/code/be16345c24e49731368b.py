# 2x2 Strassen at block level, 3x3 schoolbook at element level
# A is 2x2 blocks of 3x3, B is 2x2 blocks of 3x3
# Block (bi, bj) of A is A[bi*3:(bi+1)*3, bj*3:(bj+1)*3]

# Strassen terms for 2x2 block mult:
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)

# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
