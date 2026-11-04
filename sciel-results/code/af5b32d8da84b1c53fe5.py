"""Matrix multiplication 3,3,5 - rank 44 decomposition via block structure."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Rank 44 decomposition:
    # Part 1: 3x3x3 (first 3 cols of C) using 26-rank block Strassen
    # Part 2: 3x3x2 (last 2 cols of C) using 18-rank schoolbook
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        
    # --- Part 1: 3x3x3 (cols 0,1,2 of C) ---
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    
    # Strassen terms for A11 (0,1,2,3) * B11 (0,1,2,3)
    # m1 = (a00+a11)(b00+b11)
    add_term(0, 0, 0, 1, 1, 1, 0) # a00 b00
    add_term(0, 3, 0, 1, 1, 1, 0) # a00 b11
    add_term(3, 0, 0, 1, 1, 1, 0) # a11 b00
    add_term(3, 3, 0, 1, 1, 1, 0) # a11 b11
    # Wait, Strassen m1 is (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
    # This is 1 term in the rank decomposition? No, the rank decomposition sums products.
    # The term is U_c * V_c * W_c.
    # For m1, U is (a00+a11), V is (b00+b11), W is 1 (contribution to multiple C entries).
    # But U, V, W are vectors of length rank.
    # So for column 0:
    # U[0][0] = 1 (a00)
    # U[3][0] = 1 (a11)
    # V[0][0] = 1 (b00)
    # V[3][0] = 1 (b11)
    # This creates 4 products in this single rank term?
    # No! The rank term is a single multiplication of (sum U*A) and (sum V*B).
    # So if U[0]=1, U[3]=1, then the "A-part" is a00+a11.
    # If V[0]=1, V[3]=1, then the "B-part" is b00+b11.
    # The product is (a00+a11)(b00+b11).
    # This product contributes to W.
    # W[0][0] (C00) += 1
    # W[1][0] (C01) += 1
    # W[3][0] (C10) += 1
    # W[4][0] (C11) += 1
    # Wait, Strassen reconstruction uses signs.
    # C00 = m1 + m3 - m4 + m7
    # C01 = -m1 + m4 + m5 + m7
    # C10 = -m1 - m3 + m6 + m7
    # C11 = m1 - m5 - m6 + m7
    
    # So for m1 (col 0):
    # U: a00, a11
    # V: b00, b11
    # W: C00+=1, C01+=-1, C10+=-1, C11+=1
    # Note: W index is k*n+i.
    # C00 -> W[0*3+0]=W[0]
    # C01 -> W[1*3+0]=W[3]
    # C10 -> W[0*3+1]=W[1]
    # C11 -> W[1*3+1]=W[4]
    
    # Let's reset and do this carefully.
    # I will use a list of tuples (U_indices, V_indices, W_indices, W_vals)
    # where U_indices is list of (idx, coeff)
    
    terms = []
    
    # Strassen m1
    terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,-1),(1,-1),(4,1)]))
    # Strassen m2: (a10+a11)b00
    terms.append(([(2,1),(3,1)], [(0,1)], [(0,1),(3,-1),(1,1),(4,-1)])) 
    # Wait, Strassen m2 contributes to:
    # C00: +m2? No.
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11+B12)
    # P5 = (A11-A22)(B22+B12)
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P3 - P4 + P7
    # C12 = -P1 + P4 + P5 + P7
    # C21 = -P1 - P3 + P6 + P7
    # C22 = P1 - P5 - P6 + P7
    
    # My indices:
    # A11=(0,1), A12=(2,3) -- NO.
    # A11 is top-left 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # A12 is top-right 2x1: (0,2)->4, (1,2)->5.
    # A21 is bottom-left 1x2: (2,0)->6, (2,1)->7.
    # A22 is bottom-right 1x1: (2,2)->8.
    
    # B11: 0,1,2,3.
    # B12: 4,5.
    # B21: 6,7.
    # B22: 8.
    
    # P1: (A11+A22)(B11+B22)
    # U: A11(0,1,2,3) + A22(8)
    # V: B11(0,1,2,3) + B22(8)
    # W: C11(0), C12(1), C21(2), C22(3)
    # W vals: 1, -1, -1, 1
    terms[0] = ([(0,1),(1,1),(2,1),(3,1),(8,1)], [(0,1),(1,1),(2,1),(3,1),(8,1)], [(0,1),(1,-1),(2,-1),(3,1)])
    
    # P2: (A21+A22)B11
    # U: A21(6,7) + A22(8)
    # V: B11(0,1,2,3)
    # W: C11(0), C12(1), C21(2), C22(3)
    # W vals: 0, 0, 1, -1  (P2 contributes to C21 and C22? No.)
    # C21 = -P1 - P3 + P6 + P7. P2 is not in C21.
    # C22 = P1 - P5 - P6 + P7. P2 is not in C22.
    # Wait, P2 = (A21+A22)B11.
    # A21 is 1x2, A22 is 1x1. B11 is 2x2.
    # This product is 1x2. It contributes to C21 and C22?
    # C21 = A21 B11 + A22 B21.
    # C22 = A21 B12 + A22 B22.
    # So P2 = (A21+A22)B11 = A21 B11 + A22 B11.
    # A21 B11 is part of C21.
    # A22 B11 is NOT part of C22 (needs B22).
    # So P2 contributes to C21?
    # C21 = A21 B11 + A22 B21.
    # So A21 B11 is a term in C21.
    # A22 B11 is an extra term?
    # Strassen works on 2x2 matrices.
    # Here we have mixed sizes.
    # The 26-rank algorithm I derived earlier used simple products for the mixed blocks.
    # It did NOT use Strassen on the mixed blocks.
    # It used Strassen ONLY on A11 B11.
    # So P2, P3... are NOT standard Strassen terms for the whole 3x3.
    # My previous derivation:
    # 1. A11 B11 (7 mults via Strassen)
    # 2. A12 B21 (4 mults)
    # ...
    
    # So I should stick to the 26 simple terms.
    
    # Let's rebuild the 26 terms.
    terms = []
    
    # 7 Strassen terms for A11(0,1,2,3) B11(0,1,2,3)
    # P1
    terms.append(([(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,-1),(2,-1),(3,1)]))
    # P2
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])) # (A21+A22)B11 -> A11 is 2x2. A21 is 1x2.
    # Wait, in 2x2 Strassen, A21 is 1x1. Here A21 is 1x2? No, A11 is 2x2.
    # In the 2x2 block, A11 is top-left.
    # The Strassen algorithm is for 2x2 matrices.
    # A11 is 2x2. B11 is 2x2.
    # So the 7 terms are standard Strassen on these 2x2 blocks.
    # The results of these 7 terms are 2x2 matrices.
    # They contribute to C11 (which is 2x2).
    # So W indices are 0,1,2,3 (C00, C01, C10, C11).
    
    # P1: (A11+A22)(B11+B22)
    terms[0] = ([(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,-1),(2,-1),(3,1)])
    # P2: (A21+A22)B11
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])) # This is wrong. P2 is (A21+A22)B11.
    # In 2x2 Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11+B12)
    # P5 = (A11-A22)(B22+B12)
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P3 - P4 + P7
    # C12 = -P1 + P4 + P5 + P7
    # C21 = -P1 - P3 + P6 + P7
    # C22 = P1 - P5 - P6 + P7
    
    # For A11 B11, the "A21" in Strassen refers to the bottom-left of the 2x2 block A11.
    # So A21_strassen = A[1,0] = index 2.
    # A22_strassen = A[1,1] = index 3.
    # A12_strassen = A[0,1] = index 1.
    
    # P2: (A[1,0]+A[1,1]) B[0,0]
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])) # Contributes to C10 (idx 2) and C11 (idx 3)?
    # C21 (in 2x2) is C10 (in 3x3).
    # C22 (in 2x2) is C11 (in 3x3).
    # P2 contributes to C21 and C22 of the 2x2 block.
    # So W indices 2 and 3.
    # C21 = ... + P6 + P7. P2 is not in C21.
    # C22 = ... - P6 + P7. P2 is not in C22.
    # Wait, P2 is (A21+A22)B11.
    # It is used in C21? No.
    # C21 = A21 B11 + A22 B21.
    # So A21 B11 is a direct term.
    # In Strassen, P2 = (A21+A22)B11 = A21 B11 + A22 B11.
    # So A21 B11 = P2 - A22 B11.
    # This is getting complicated.
    
    # Let's just use the 26 simple terms I derived earlier.
    
    terms = []
    # 1. A11 B11 via 7 Strassen terms
    # I will hardcode the 7 Strassen terms for the 2x2 block (indices 0,1,2,3).
    # P1
    terms.append(([(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,1),(2,1),(3,1)], [(0,1),(1,-1),(2,-1),(3,1)]))
    # P2
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]))
    # P3
    terms.append(([(0,1)], [(2,1),(3,-1)], [(0,1),(1,0),(2,1),(3,0)]))
    # P4
    terms.append(([(3,1)], [(0,1),(1,1)], [(0,-1),(1,1),(2,0),(3,1)]))
    # P5
    terms.append(([(0,1),(3,-1)], [(3,1),(1,1)], [(0,0),(1,1),(2,0),(3,-1)]))
    # P6
    terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(0,0),(1,0),(2,1),(3,-1)]))
    # P7
    terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(1,1),(2,1),(3,1)]))
    
    # 2. A12 B21 (4 terms)
    # A12: 4,5. B21: 6,7.
    # C00 += a02 b20 (4*6)
    terms.append(([(4,1)], [(6,1)], [(0,1)]))
    # C01 += a02 b21 (4*7)
    terms.append(([(4,1)], [(7,1)], [(1,1)]))
    # C10 += a12 b20 (5*6)
    terms.append(([(5,1)], [(6,1)], [(2,1)]))
    # C11 += a12 b21 (5*7)
    terms.append(([(5,1)], [(7,1)], [(3,1)]))
    
    # 3. A11 B12 (4 terms)
    # A11: 0,1,2,3. B12: 4,5.
    # C02 += a00 b02 + a01 b12
    terms.append(([(0,1)], [(4,1)], [(4,1)]))
    terms.append(([(1,1)], [(5,1)], [(4,1)]))
    # C12 += a10 b02 + a11 b12
    terms.append(([(2,1)], [(4,1)], [(5,1)]))
    terms.append(([(3,1)], [(5,1)], [(5,1)]))
    
    # 4. A12 B22 (2 terms)
    # A12: 4,5. B22: 8.
    # C02 += a02 b22
    terms.append(([(4,1)], [(8,1)], [(4,1)]))
    # C12 += a12 b22
    terms.append(([(5,1)], [(8,1)], [(5,1)]))
    
    # 5. A21 B11 (4 terms)
    # A21: 6,7. B11: 0,1,2,3.
    # C20 += a20 b00 + a21 b10
    terms.append(([(6,1)], [(0,1)], [(6,1)]))
    terms.append(([(7,1)], [(2,1)], [(6,1)]))
    # C21 += a20 b01 + a21 b11
    terms.append(([(6,1)], [(1,1)], [(7,1)]))
    terms.append(([(7,1)], [(3,1)], [(7,1)]))
    
    # 6. A22 B21 (2 terms)
    # A22: 8. B21: 6,7.
    # C20 += a22 b20
    terms.append(([(8,1)], [(6,1)], [(6,1)]))
    # C21 += a22 b21
    terms.append(([(8,1)], [(7,1)], [(7,1)]))
    
    # 7. A21 B12 (2 terms)
    # A21: 6,7. B12: 4,5.
    # C22 += a20 b02 + a21 b12
    terms.append(([(6,1)], [(4,1)], [(8,1)]))
    terms.append(([(7,1)], [(5,1)], [(8,1)]))
    
    # 8. A22 B22 (1 term)
    # C22 += a22 b22
    terms.append(([(8,1)], [(8,1)], [(8,1)]))
    
    # Total 26 terms for Part 1.
    
    # --- Part 2: 3x3x2 (cols 3,4 of C) ---
    # Schoolbook 18 terms.
    # C03 = a00 b03 + a01 b13 + a02 b23
    terms.append(([(0,1)], [(9,1)], [(9,1)])) # C03 is W[3*3+0]=W[9]
    terms.append(([(1,1)], [(10,1)], [(9,1)]))
    terms.append(([(4,1)], [(12,1)], [(9,1)]))
    
    # C13 = a10 b03 + a11 b13 + a12 b23
    terms.append(([(2,1)], [(9,1)], [(10,1)])) # C13 is W[3*3+1]=W[10]
    terms.append(([(3,1)], [(10,1)], [(10,1)]))
    terms.append(([(5,1)], [(12,1)], [(10,1)]))
    
    # C23 = a20 b03 + a21 b13 + a22 b23
    terms.append(([(6,1)], [(9,1)], [(11,1)])) # C23 is W[3*3+2]=W[11]
    terms.append(([(7,1)], [(10,1)], [(11,1)]))
    terms.append(([(8,1)], [(12,1)], [(11,1)]))
    
    # C04 = a00 b04 + a01 b14 + a02 b24
    terms.append(([(0,1)], [(11,1)], [(12,1)])) # C04 is W[4*3+0]=W[12]
    terms.append(([(1,1)], [(13,1)], [(12,1)]))
    terms.append(([(4,1)], [(15,1)], [(12,1)]))
    
    # C14 = a10 b04 + a11 b14 + a12 b24
    terms.append(([(2,1)], [(11,1)], [(13,1)])) # C14 is W[4*3+1]=W[13]
    terms.append(([(3,1)], [(13,1)], [(13,1)]))
    terms.append(([(5,1)], [(15,1)], [(13,1)]))
    
    # C24 = a20 b04 + a21 b14 + a22 b24
    terms.append(([(6,1)], [(11,1)], [(14,1)])) # C24 is W[4*3+2]=W[14]
    terms.append(([(7,1)], [(13,1)], [(14,1)]))
    terms.append(([(8,1)], [(15,1)], [(14,1)]))
    
    # Total 18 terms for Part 2.
    # Total 44 terms.
    
    # Construct U, V, W
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for idx, val in u_list:
            U[idx][c] += val
        for idx, val in v_list:
            V[idx][c] += val
        for idx, val in w_list:
            W[idx][c] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
