"""Matrix multiplication 2x3x8 with rank-minimal decomposition using Winograd/Strassen for 2x3x2 blocks."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose into 4 blocks of 2x3x2 (column pairs of B)
    # For each 2x3x2 block, use a rank-6 Winograd-like decomposition
    # Total rank = 4 * 6 = 24
    
    # For 2x3x2 matrix multiplication:
    # A is 2x3, B is 3x2, C = A*B is 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Winograd-style rank-6 decomposition for 2x3x2:
    # We use 6 bilinear forms. Let's define them:
    # 
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B = [[b00, b01], [b10, b11], [b20, b21]]
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # Winograd approach: use sums and differences
    # M1 = (a00+a10)*(b00+b01)
    # M2 = (a01+a11)*(b10+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*(b00-b01)
    # M5 = (a01-a11)*(b10-b11)
    # M6 = (a02-a12)*(b20-b21)
    #
    # Then:
    # C[0,0] = (M1+M4)/2 + (M2+M5)/2 + (M3+M6)/2
    # C[0,1] = (M1-M4)/2 + (M2-M5)/2 + (M3-M6)/2
    # C[1,0] = (M1+M4)/2 - (M2+M5)/2 - (M3+M6)/2  ... wait, let me redo this
    
    # Actually, let me use a cleaner Winograd decomposition.
    # For 2x2x2, Strassen uses 7 multiplications.
    # For 2x3x2, we can do better than 12.
    
    # Let's use the following rank-6 decomposition:
    # M1 = (a00+a10) * (b00+b01)
    # M2 = (a00-a10) * (b00-b01)
    # M3 = (a01+a11) * (b10+b11)
    # M4 = (a01-a11) * (b10-b11)
    # M5 = (a02+a12) * (b20+b21)
    # M6 = (a02-a12) * (b20-b21)
    #
    # C[0,0] = (M1+M2)/2 + (M3+M4)/2 + (M5+M6)/2
    # C[0,1] = (M1-M2)/2 + (M3-M4)/2 + (M5-M6)/2
    # C[1,0] = (M1+M2)/2 - (M3+M4)/2 - (M5+M6)/2  -- this doesn't look right
    
    # Let me verify:
    # M1 = (a00+a10)(b00+b01) = a00*b00 + a00*b01 + a10*b00 + a10*b01
    # M2 = (a00-a10)(b00-b01) = a00*b00 - a00*b01 - a10*b00 + a10*b01
    # M1+M2 = 2*a00*b00 + 2*a10*b01
    # M1-M2 = 2*a00*b01 + 2*a10*b00
    
    # M3 = (a01+a11)(b10+b11) = a01*b10 + a01*b11 + a11*b10 + a11*b11
    # M4 = (a01-a11)(b10-b11) = a01*b10 - a01*b11 - a11*b10 + a11*b11
    # M3+M4 = 2*a01*b10 + 2*a11*b11
    # M3-M4 = 2*a01*b11 + 2*a11*b10
    
    # M5 = (a02+a12)(b20+b21) = a02*b20 + a02*b21 + a12*b20 + a12*b21
    # M6 = (a02-a12)(b20-b21) = a02*b20 - a02*b21 - a12*b20 + a12*b21
    # M5+M6 = 2*a02*b20 + 2*a12*b21
    # M5-M6 = 2*a02*b21 + 2*a12*b20
    
    # So:
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    #        = (M1+M2)/4 + (M3+M4)/4 + (M5+M6)/4  -- no, that gives a00*b00/2 + a10*b01/2 + ...
    
    # Let me be more careful:
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # From M1, M2:
    # a00*b00 = (M1+M2)/4 + (M1-M2)/4 = ... let's just express each term
    
    # a00*b00 = (M1+M2)/4 + (M1-M2)/4 -- no
    # M1+M2 = 2*a00*b00 + 2*a10*b01
    # M1-M2 = 2*a00*b01 + 2*a10*b00
    # So: a00*b00 = (M1+M2 - (M1-M2))/4 + ... this is getting complicated.
    
    # Let me just use the direct expressions:
    # a00*b00 = (M1+M2)/4 + (M1-M2)/4 - (M1-M2)/4 + (M1+M2)/4 -- no
    
    # Simpler: 
    # a00*b00 = (M1+M2)/4 + (M1-M2)/4 = M1/2 -- no, that's not right either.
    
    # Let me just solve the linear system. We have 6 Ms and 4 Cs.
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # In terms of Ms:
    # M1 = a00*b00 + a00*b01 + a10*b00 + a10*b01
    # M2 = a00*b00 - a00*b01 - a10*b00 + a10*b01
    # M3 = a01*b10 + a01*b11 + a11*b10 + a11*b11
    # M4 = a01*b10 - a01*b11 - a11*b10 + a11*b11
    # M5 = a02*b20 + a02*b21 + a12*b20 + a12*b21
    # M6 = a02*b20 - a02*b21 - a12*b20 + a12*b21
    
    # Let's denote:
    # x1 = a00*b00, x2 = a00*b01, x3 = a10*b00, x4 = a10*b01
    # x5 = a01*b10, x6 = a01*b11, x7 = a11*b10, x8 = a11*b11
    # x9 = a02*b20, x10 = a02*b21, x11 = a12*b20, x12 = a12*b21
    
    # M1 = x1 + x2 + x3 + x4
    # M2 = x1 - x2 - x3 + x4
    # M3 = x5 + x6 + x7 + x8
    # M4 = x5 - x6 - x7 + x8
    # M5 = x9 + x10 + x11 + x12
    # M6 = x9 - x10 - x11 + x12
    
    # C[0,0] = x1 + x5 + x9
    # C[1,0] = x3 + x7 + x11
    # C[0,1] = x2 + x6 + x10
    # C[1,1] = x4 + x8 + x12
    
    # We need to express C[i,k] as linear combinations of M1..M6.
    # C[0,0] = x1 + x5 + x9
    # x1 = (M1+M2)/2 - x4  -- no, we can't express x1 alone from M1, M2 without x4.
    
    # Actually, we have 12 unknowns (x1..x12) and 6 equations (M1..M6). 
    # The C's are 4 linear combinations of the x's. We need to check if the C's 
    # are in the span of the M's.
    
    # Let's write C in terms of M:
    # C[0,0] = x1 + x5 + x9
    # C[1,0] = x3 + x7 + x11
    # C[0,1] = x2 + x6 + x10
    # C[1,1] = x4 + x8 + x12
    
    # M1+M2 = 2*x1 + 2*x4
    # M1-M2 = 2*x2 + 2*x3
    # M3+M4 = 2*x5 + 2*x8
    # M3-M4 = 2*x6 + 2*x7
    # M5+M6 = 2*x9 + 2*x12
    # M5-M6 = 2*x10 + 2*x11
    
    # So:
    # x1 + x4 = (M1+M2)/2
    # x2 + x3 = (M1-M2)/2
    # x5 + x8 = (M3+M4)/2
    # x6 + x7 = (M3-M4)/2
    # x9 + x12 = (M5+M6)/2
    # x10 + x11 = (M5-M6)/2
    
    # C[0,0] = x1 + x5 + x9
    # C[1,0] = x3 + x7 + x11
    # C[0,1] = x2 + x6 + x10
    # C[1,1] = x4 + x8 + x12
    
    # We can't directly express these from the Ms alone without additional constraints.
    # This means the rank-6 decomposition with this particular choice of Ms doesn't work.
    
    # Let me try a different approach. Maybe use a known rank-6 decomposition for 2x3x2.
    
    # Actually, I recall that M(2,3,2) = 6 is achievable. Let me look for the explicit decomposition.
    
    # One known decomposition:
    # M1 = a00*b00
    # M2 = a00*b01
    # M3 = a01*b10
    # M4 = a01*b11
    # M5 = a02*b20
    # M6 = a02*b21
    # But that's just 6 of the 12 terms, missing the a1j terms.
    
    # I think I need to use a more sophisticated approach. Let me just use the schoolbook 
    # for now with rank 48, but structure it clearly.
    
    # Actually, let me try to use rank 24 by using a rank-6 decomposition per block.
    # I'll use the following Winograd-like scheme:
    
    # For each block (columns 2t, 2t+1 of B):
    # A is 2x3, B_block is 3x2
    # 
    # M1 = (a00+a10) * (b00+b01)
    # M2 = (a00-a10) * (b00-b01)
    # M3 = (a01+a11) * (b10+b11)
    # M4 = (a01-a11) * (b10-b11)
    # M5 = (a02+a12) * (b20+b21)
    # M6 = (a02-a12) * (b20-b21)
    #
    # C[0,0] = (M1+M2)/4 + (M3+M4)/4 + (M5+M6)/4
    # C[1,0] = (M1+M2)/4 - (M3+M4)/4 - (M5+M6)/4  -- still not working
    
    # I'll just go with schoolbook rank 48 for correctness.
    
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
