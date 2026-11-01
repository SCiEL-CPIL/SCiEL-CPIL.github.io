import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # p=9, so 3 blocks of 3
    # Each block is 2x3x3
    # We use rank 11 for each 2x3x3 block
    rank_block = 11
    total_rank = 3 * rank_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Indices for A (2x3): 0..5
    # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
    # Indices for B (3x3) for one block: 0..8
    # b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    # Indices for C (2x3): 0..5
    # c00=0, c01=1, c02=2, c10=3, c11=4, c12=5
    
    # Strassen for 2x2:
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a10)(b01+b11)
    # M3 = (a00)(b00)
    # M4 = (a11)(b11)
    # M5 = (a00+a11)(b10+b11)
    # M6 = (a01-a11)(b10+b11)
    # M7 = (a10+a11)(b00-b10)
    
    # Terms for 2x2 product:
    # c00 = M3 + M5 - M4 - M7
    # c01 = M1 - M5 + M4 + M6
    # c10 = M2 - M1 + M3 + M7
    # c11 = M1 - M2 + M4 - M6
    
    # Additional terms from 3rd column/row:
    # c00 += a02*b20
    # c01 += a02*b21
    # c10 += a12*b20
    # c11 += a12*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # We need to compute:
    # a02*b20, a02*b21, a12*b20, a12*b21
    # a00*b02, a01*b12, a02*b22
    # a10*b02, a11*b12, a12*b22
    
    # Let's use 4 mults for the 2x2 outer product part (a02, a12) x (b20, b21):
    # M8 = (a02+a12)(b20+b21)
    # M9 = (a02-a12)(b20-b21)
    # a02*b20 = (M8+M9)/2
    # a02*b21 = (M8-M9)/2
    # a12*b20 = (M8-M9)/2 ? No.
    # a12*b20 = (M8-M9)/2 is wrong.
    # M8 = a02*b20 + a02*b21 + a12*b20 + a12*b21
    # M9 = a02*b20 - a02*b21 - a12*b20 + a12*b21
    # M8+M9 = 2*a02*b20
    # M8-M9 = 2*a12*b21
    # We need a02*b21 and a12*b20.
    # We need 2 more mults.
    # M10 = a02*b21
    # M11 = a12*b20
    # Then a02*b20 = (M8+M9)/2 - M11? No.
    # a02*b20 = (M8+M9)/2.
    # a12*b21 = (M8-M9)/2.
    # So we have a02*b20, a12*b21, and we use M10, M11 for the other two.
    
    # Now for c02 and c12:
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # We need a00*b02, a01*b12, a02*b22, a10*b02, a11*b12, a12*b22.
    # This is 6 terms.
    # We can use 3 more mults?
    # M12 = (a00+a10)*b02
    # M13 = (a01+a11)*b12
    # M14 = (a02+a12)*b22
    # Then a00*b02 = (M12 - a10*b02).
    # We don't have a10*b02.
    # This is getting complicated.
    
    # Let's stick to a simpler, verified rank.
    # I will use the schoolbook rank 54 if I cannot ensure correctness of a lower rank.
    # But 54 is the seed.
    # I will use a rank 27 decomposition (3x9) if I can.
    # Actually, I will use the rank 7 decomposition for 2x3x3 which is known to exist.
    # I will hardcode a rank 7 decomposition.
    
    pass
