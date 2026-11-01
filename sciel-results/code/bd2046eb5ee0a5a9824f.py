import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 32: 4 blocks of (Strassen 2x2x2 + 1 term for j=2)
    R = 32
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01+B11)
    # M5: (A00+A01)B11
    # M6: (A01-A10)(B00+B01)
    # M7: (A10-A00)(B11-B10)
    
    # U coeffs: [i, j] -> coeff
    # V coeffs: [j, k'] -> coeff
    # W coeffs: [k', i] -> coeff (sign for C)
    
    # Term 0 (M1)
    U0 = {(0,0): 1, (1,1): 1}
    V0 = {(0,0): 1, (1,1): 1}
    W0 = {(0,0): 1, (1,1): 1} # C00 += M1, C11 += M1
    
    # Term 1 (M2)
    U1 = {(1,0): 1, (1,1): 1}
    V1 = {(0,0): 1}
    W1 = {(0,0): 1, (1,1): -1} # C10 += M2, C11 -= M2
    
    # Term 2 (M3)
    U2 = {(0,0): 1}
    V2 = {(1,0): 1, (1,1): -1}
    W2 = {(0,1): 1, (1,1): 1} # C01 += M3, C11 += M3
    
    # Term 3 (M4)
    U3 = {(1,1): 1}
    V3 = {(0,1): 1, (1,1): 1}
    W3 = {(0,0): 1, (1,0): 1} # C00 += M4, C10 += M4
    
    # Term 4 (M5)
    U4 = {(0,0): 1, (0,1): 1}
    V4 = {(1,1): 1}
    W4 = {(0,0): -1, (0,1): 1} # C00 -= M5, C01 += M5
    
    # Term 5 (M6)
    U5 = {(0,1): 1, (1,0): -1}
    V5 = {(0,0): 1, (0,1): 1}
    W5 = {(1,0): 1, (1,1): -1} # C10 += M6? No.
    # Let's re-verify Strassen formulas.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # My W assignments above:
    # M1: C00(+), C11(+) -> OK
    # M2: C10(+), C11(-) -> OK
    # M3: C01(+), C11(+) -> OK
    # M4: C00(+), C10(+) -> OK
    # M5: C00(-), C01(+) -> OK
    # M6: Need C10? No, M6 is not in standard Strassen?
    # Standard Strassen uses 7 products.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01+B11)
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B00+B01)
    # M7 = (A10-A00)(B11-B10)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So M6 is NOT used?
    # Wait, Strassen algorithm:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01+b11)
    # p5 = (a00+a01)b11
    # p6 = (a01-a10)(b00+b01)
    # p7 = (a10-a00)(b11-b10)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    
    # p6 is not in the sum?
    # Ah, I recall Strassen uses 7 products, but maybe my mapping is wrong.
    # Let's check p6.
    # Actually, the standard 7 products are as above.
    # But c00..c11 do not use p6?
    # That would be 6 products?
    # No, the 7th product is used.
    # Let's check a source.
    # Wikipedia:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01+b11)
    # p5 = (a00+a01)b11
    # p6 = (a01-a10)(b00+b01)
    # p7 = (a10-a00)(b11-b10)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    
    # It seems p6 is not used in the reconstruction?
    # That can't be right. If p6 is not used, why compute it?
    # Maybe I copied the formulas wrong.
    # Let's try to derive or find correct formulas.
    # Actually, there are different variants.
    # One variant:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)(b00+b01)
    # p3 = (a00+a01)(b10+b11)
    # p4 = (a10-a00)(b01-b00)
    # p5 = (a01-a11)(b11-b10)
    # p6 = a00(b00-b11)
    # p7 = a11(b11-b00)
    
    # c00 = p1 + p4 - p5 + p6
    # c01 = p3 + p5 - p1 + p7 ? No.
    
    # Let's stick to the first set and check if p6 is redundant or I missed it.
    # If p6 is not used, rank is 6? No, rank is 7.
    # Maybe c11 uses p6?
    # Let's assume the standard 7 products are correct and find the correct linear combination.
    
    # Actually, I can just use the known rank-7 decomposition.
    # I will implement the specific coefficients from a reliable source.
    
    # For now, I will use the schoolbook for the 2x2x2 part to be safe, 
    # but the prompt implies Strassen is possible.
    # Schoolbook for 2x2x2 is rank 8.
    # 4 blocks * 8 = 32. Plus 4 * 1 = 4. Total 36.
    # 36 is better than 44.
    # Let's implement rank 36 first.
    
    R = 36
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for s in range(4):
        # Block s: k in {2s, 2s+1}
        
        # Part 1: j in {0, 1}. 2x2x2 Schoolbook.
        for i in range(2):
            for j in range(2):
                for k_prime in range(2):
                    k = 2*s + k_prime
                    U[i*3 + j][col] = 1
                    V[j*8 + k][col] = 1
                    W[k*2 + i][col] = 1
                    col += 1
        # Part 2: j=2. 2x1x2. Rank 1.
        for i in range(2):
            for k_prime in range(2):
                k = 2*s + k_prime
                # We can combine these into 1 term?
                # U[i,2] V[2,k] W[k,i]
                # If we use 1 term, U[i,2]=1, V[2,k]=1, W[k,i]=1.
                # Sum over r? No, r is fixed.
                # We need sum_r U V W = 1 for each (i,k).
                # If we use 1 term r:
                # U[0,2,r]=1, U[1,2,r]=1
                # V[2,2s,r]=1, V[2,2s+1,r]=1
                # W[2s,0,r]=1, W[2s,1,r]=1, W[2s+1,0,r]=1, W[2s+1,1,r]=1
                # Then for i=0, k=2s: 1*1*1 = 1.
                # For i=1, k=2s: 1*1*1 = 1.
                # Yes, 1 term works for the whole 2x1x2 block.
                pass
        
        # So 8 terms for Part 1, 1 term for Part 2.
        # Total 9 terms per block?
        # Wait, schoolbook for 2x2x2 is 8 terms.
        # So 8 + 1 = 9 terms per block.
        # 4 blocks * 9 = 36 terms.
        
        # Let's re-verify the 1 term for Part 2.
        # We need to set U, V, W for this term.
        r = col
        for i in range(2):
            U[i*3 + 2][r] = 1
        for k_prime in range(2):
            k = 2*s + k_prime
            V[2*8 + k][r] = 1
            for i in range(2):
                W[k*2 + i][r] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
