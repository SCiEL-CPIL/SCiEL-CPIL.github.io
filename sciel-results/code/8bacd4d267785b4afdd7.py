# Laderman 3x3 terms
# Each term is a list of (U_indices, V_indices, W_indices)
# U_indices: list of (i, j, coeff)
# V_indices: list of (j, k, coeff)
# W_indices: list of (k, i, coeff) -> Note: W index is k*n+i
# In 3x3, n=3. W index is k*3+i.

laderman_terms = []

# 1
laderman_terms.append([[(0,0,1)], [(0,0,1)], [(0,0,1)]])
# 2
laderman_terms.append([[(1,1,1)], [(1,1,1)], [(1,1,1)]])
# 3
laderman_terms.append([[(2,2,1)], [(2,2,1)], [(2,2,1)]])
# 4
laderman_terms.append([[(0,1,1)], [(1,0,1)], [(0,0,1)]])
# 5
laderman_terms.append([[(1,2,1)], [(2,0,1)], [(0,1,1)]]) # C[1,0] -> W[0,1]
# 6
laderman_terms.append([[(2,0,1)], [(0,1,1)], [(1,2,1)]]) # C[2,1] -> W[1,2]
# 7
laderman_terms.append([[(0,2,1)], [(2,1,1)], [(1,0,1)]]) # C[0,1] -> W[1,0]
# 8
laderman_terms.append([[(1,0,1)], [(0,2,1)], [(2,1,1)]]) # C[1,2] -> W[2,1]
# 9
laderman_terms.append([[(2,1,1)], [(1,2,1)], [(2,2,1)]]) # C[2,2] -> W[2,2]

# 10: (a00+a10)(b00+b10)
# A: (0,0), (1,0)
# B: (0,0), (1,0)
# C: (0,0) from a00b00, (1,0) from a10b10
laderman_terms.append([[(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)], [(0,0,1), (0,1,1)]])

# 11: (a01+a11)(b10+b11)
# A: (0,1), (1,1)
# B: (1,0), (1,1)
# C: (0,0) from a01b10, (0,1) from a01b11, (1,1) from a11b11
laderman_terms.append([[(0,1,1), (1,1,1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,0,1), (1,1,1)]])

# 12: (a02+a12)(b20+b21)
# A: (0,2), (1,2)
# B: (2,0), (2,1)
# C: (0,1) from a02b21, (1,0) from a12b20, (1,1) from a12b21
laderman_terms.append([[(0,2,1), (1,2,1)], [(2,0,1), (2,1,1)], [(1,0,1), (0,1,1), (1,1,1)]])

# 13: (a00+a20)(b00+b20)
# A: (0,0), (2,0)
# B: (0,0), (2,0)
# C: (0,0) from a00b00, (2,0) from a20b20
laderman_terms.append([[(0,0,1), (2,0,1)], [(0,0,1), (2,0,1)], [(0,0,1), (0,2,1)]])

# 14: (a01+a21)(b10+b20)
# A: (0,1), (2,1)
# B: (1,0), (2,0)
# C: (0,0) from a01b10, (2,0) from a21b20
laderman_terms.append([[(0,1,1), (2,1,1)], [(1,0,1), (2,0,1)], [(0,0,1), (0,2,1)]])

# 15: (a02+a22)(b20+b21)
# A: (0,2), (2,2)
# B: (2,0), (2,1)
# C: (0,1) from a02b21, (2,1) from a22b21
laderman_terms.append([[(0,2,1), (2,2,1)], [(2,0,1), (2,1,1)], [(1,0,1), (1,2,1)]])

# 16: (a10+a20)(b01+b21)
# A: (1,0), (2,0)
# B: (0,1), (2,1)
# C: (1,1) from a10b01, (2,1) from a20b21
laderman_terms.append([[(1,0,1), (2,0,1)], [(0,1,1), (2,1,1)], [(1,1,1), (1,2,1)]])

# 17: (a11+a21)(b11+b21)
# A: (1,1), (2,1)
# B: (1,1), (2,1)
# C: (1,1) from a11b11, (2,1) from a21b21
laderman_terms.append([[(1,1,1), (2,1,1)], [(1,1,1), (2,1,1)], [(1,1,1), (1,2,1)]])

# 18: (a12+a22)(b21+b22)
# A: (1,2), (2,2)
# B: (2,1), (2,2)
# C: (1,1) from a12b21, (2,2) from a22b22
laderman_terms.append([[(1,2,1), (2,2,1)], [(2,1,1), (2,2,1)], [(1,1,1), (2,2,1)]])

# 19: (a00+a01+a02)(b00+b10+b20)
# A: (0,0), (0,1), (0,2)
# B: (0,0), (1,0), (2,0)
# C: (0,0) from a00b00, (0,0) from a01b10, (0,0) from a02b20 -> 3 to C00?
# Wait, a01b10 is C00. a02b20 is C00.
# So C00 gets 1+1+1 = 3?
# And other terms?
# a00b10 -> mismatch.
# a00b20 -> mismatch.
# a01b00 -> mismatch.
# a01b20 -> mismatch.
# a02b00 -> mismatch.
# a02b10 -> mismatch.
# So only C00 gets contributions?
# This seems unlikely to be a valid term if it only contributes to one entry with coeff 3.
# Actually, Laderman's 19th term is often $(a_{00}+a_{01}+a_{02})(b_{00}+b_{10}+b_{20})$?
# No, the last term is usually a sum that corrects the diagonal or something.

# Given the high risk of error in the coefficients, I will NOT implement Laderman.
# I will keep the current program.
