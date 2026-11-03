"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = 120.0
    
    # The tensor decomposition for matrix multiplication C = A*B
    # where A is n x m, B is m x p, C is n x p
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook gives rank n*m*p = 48
    # Winograd 2x2 can reduce this. Let's use a verified approach.
    
    # The key insight: for 2x2 matrix multiplication, we can use 7 multiplications
    # instead of 8. For 3x4 * 4x4, we can partition into 2x2 blocks where possible.
    
    # Let me use a different approach: build the decomposition using the 
    # tensor rank properties and try to find a lower rank decomposition.
    
    # Actually, let me just carefully implement Winograd 2x2.
    # The correct Winograd 2x2 scheme (verified):
    # For C = A*B where A, B, C are 2x2:
    # m0 = (A[0,0] + A[0,1]) * (B[0,0] - B[1,0])
    # m1 = (A[1,0] + A[1,1]) * (B[1,0] + B[1,1])
    # m2 = A[0,0] * (B[0,1] - B[1,1])
    # m3 = A[1,1] * (B[0,0] + B[1,0])
    # m4 = A[0,1] * (B[0,1] + B[1,1])
    # m5 = (A[1,0] - A[0,0]) * B[0,0]
    # m6 = (A[0,1] - A[1,1]) * B[1,1]
    
    # C[0,0] = m0 + m1 + m3 + m5
    # C[0,1] = m2 + m4 + m3 + m6
    # C[1,0] = m0 + m1 + m5 + m3
    # C[1,1] = m2 + m4 + m0 + m6
    
    # Wait, let me verify this properly.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # m0 = (a+b)(e-g) = ae - ag + be - bg
    # m1 = (c+d)(g+h) = cg + ch + dg + dh
    # m2 = a(f-h) = af - ah
    # m3 = d(e+g) = de + dg
    # m4 = b(f+h) = bf + bh
    # m5 = (c-a)e = ce - ae
    # m6 = (b-d)h = bh - dh
    
    # C[0,0] = ae + bg
    # m0 + m1 + m3 + m5 = (ae-ag+be-bg) + (cg+ch+dg+dh) + (de+dg) + (ce-ae)
    # = ae - ag + be - bg + cg + ch + dg + dh + de + dg + ce - ae
    # = -ag + be - bg + cg + ch + 2dg + dh + de + ce
    # This is not equal to ae + bg. So my scheme is wrong.
    
    # Let me use a known correct Winograd 2x2 scheme.
    # From standard references:
    # m1 = (A[0,0] + A[0,1]) * (B[0,0] - B[1,0])
    # m2 = (A[1,0] + A[1,1]) * (B[1,0] + B[1,1])
    # m3 = A[0,0] * (B[0,1] - B[1,1])
    # m4 = A[1,1] * (B[0,0] + B[1,0])
    # m5 = A[0,1] * (B[0,1] + B[1,1])
    # m6 = (A[1,0] - A[0,0]) * B[0,0]
    # m7 = (A[0,1] - A[1,1]) * B[1,1]
    
    # C[0,0] = m1 + m2 + m4 + m6
    # C[0,1] = m3 + m5 + m4 + m7
    # C[1,0] = m1 + m2 + m6 + m4
    # C[1,1] = m3 + m5 + m1 + m7
    
    # Let me verify with A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    # C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[5+14, 6+16],[15+28, 18+32]] = [[19,22],[43,50]]
    
    # m1 = (1+2)*(5-7) = 3*(-2) = -6
    # m2 = (3+4)*(7+8) = 7*15 = 105
    # m3 = 1*(6-8) = -2
    # m4 = 4*(5+7) = 48
    # m5 = 2*(6+8) = 28
    # m6 = (3-1)*5 = 10
    # m7 = (2-4)*8 = -16
    
    # C[0,0] = m1+m2+m4+m6 = -6+105+48+10 = 157. Expected 19. Still wrong!
    
    # I'm clearly making an error. Let me just use the schoolbook approach which is guaranteed correct.
    # The rank is 48.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
