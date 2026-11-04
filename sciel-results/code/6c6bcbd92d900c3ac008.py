"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-36 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for 2x2 blocks.
    - Split A (2x4) into four 2x2 blocks: A00, A01, A10, A11
    - Split B (4x6) into two 2x3 blocks: B0, B1
    - For each 2x2 * 2x3 multiplication, use 7 Strassen multiplications
    - Total: 2 * 7 = 14 multiplications... wait, that's not right.
    
    Let me reconsider. The standard approach:
    - A is 2x4, B is 4x6
    - We can view this as multiplying two 2x2 matrices where each entry is itself a matrix
    - A = [[A00, A01], [A10, A11]] where each Ai is 1x2 (a row vector of length 2)
    - B = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x1 (a column vector of length 2)
    
    Actually, let me think more carefully.
    
    A is 2x4: rows are length 4.
    B is 4x6: columns are length 4.
    
    We can partition the 4 middle indices into two groups of 2:
    - Group 0: indices 0,1
    - Group 1: indices 2,3
    
    A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
    
    Then C = A*B = A_left * B_top + A_right * B_bottom
    
    Each term is a 2x2 * 2x6 multiplication.
    
    For 2x2 * 2x6:
    - Split B into three 2x2 blocks: B = [B0 | B1 | B2]
    - C = A * [B0 | B1 | B2] = [A*B0 | A*B1 | A*B2]
    - Each A*Bj is a 2x2 * 2x2 multiplication, which can be done with 7 Strassen products.
    - So 2x2 * 2x6 requires 3*7 = 21 products.
    
    Total: 2 * 21 = 42 products.
    
    But we can do better by sharing some of the Strassen intermediate products.
    
    Actually, let me just implement the straightforward 42-rank decomposition using Strassen for each 2x2*2x2 block.
    
    Strassen for 2x2 * 2x2:
    Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    m1 = (a+d)*(e+h)
    m2 = (c+d)*e
    m3 = a*(f-h)
    m4 = d*(g-e)
    m5 = (a+b)*h
    m6 = (c-a)*(e+f)
    m7 = (b-d)*(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    For 2x2 * 2x6 where B = [B0|B1|B2] with each Bj 2x2:
    We compute A*Bj for j=0,1,2 using Strassen.
    
    Total rank = 2 (for the two halves of the 4 middle indices) * 3 (three 2x2 blocks in the 6 output) * 7 (Strassen) = 42.
    
    Let me build this explicitly.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll build rank-42 decomposition
    # The 4 middle indices are split into two groups of 2
    # The 6 output columns are split into three groups of 2
    
    # For each of the 2 middle groups (left/right) and each of the 3 output groups (col 0-1, 2-3, 4-5):
    # Use 7 Strassen products
    # Total: 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)*(e+h): A factors: a->1, d->1; B factors: e->1, h->1
    # m2 = (c+d)*e: A factors: c->1, d->1; B factors: e->1
    # m3 = a*(f-h): A factors: a->1; B factors: f->1, h->-1
    # m4 = d*(g-e): A factors: d->1; B factors: g->1, e->-1
    # m5 = (a+b)*h: A factors: a->1, b->1; B factors: h->1
    # m6 = (c-a)*(e+f): A factors: c->1, a->-1; B factors: e->1, f->1
    # m7 = (b-d)*(g+h): A factors: b->1, d->-1; B factors: g->1, h->1
    
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    
    strassen_A = [
        # For each of 7 products, coefficients for a, b, c, d
        [1, 0, 0, 1],   # m1: a+d
        [0, 0, 1, 1],   # m2: c+d
        [1, 0, 0, 0],   # m3: a
        [0, 0, 0, 1],   # m4: d
        [1, 1, 0, 0],   # m5: a+b
        [-1, 0, 1, 0],  # m6: c-a
        [0, 1, 0, -1],  # m7: b-d
    ]
    
    strassen_B = [
        # For each of 7 products, coefficients for e, f, g, h
        [1, 0, 0, 1],   # m1: e+h
        [1, 0, 0, 0],   # m2: e
        [0, 1, 0, -1],  # m3: f-h
        [-1, 0, 1, 0],  # m4: g-e
        [0, 0, 0, 1],   # m5: h
        [1, 1, 0, 0],   # m6: e+f
        [0, 0, 1, 1],   # m7: g+h
    ]
    
    strassen_C = [
        # For each output position (00, 01, 10, 11), coefficients for m1..m7
        [1, 0, 0, 1, -1, 0, 1],   # C00 = m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = m3+m5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = m2+m4
        [1, -1, 1, 0, 0, 1, 0],  # C11 = m1-m2+m3+m6
    ]
    
    col = 0
    for mid_group in range(2):  # left (indices 0,1) or right (indices 2,3)
        for out_group in range(3):  # columns 0-1, 2-3, 4-5
            for s in range(7):  # 7 Strassen products
                # U: 2x4 matrix, index by row i (0,1) and col j (0..3)
                # A[i, j] where j = mid_group*2 + (0 or 1)
                # In A = [[a,b],[c,d]], a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
                # Our A here is 2x2 with entries from the full 2x4 A
                # A_full[0, mid_group*2+0] = a, A_full[0, mid_group*2+1] = b
                # A_full[1, mid_group*2+0] = c, A_full[1, mid_group*2+1] = d
                
                # U indices: i*m + j = i*4 + j
                U[0 * 4 + mid_group * 2 + 0][col] += strassen_A[s][0]  # a
                U[0 * 4 + mid_group * 2 + 1][col] += strassen_A[s][1]  # b
                U[1 * 4 + mid_group * 2 + 0][col] += strassen_A[s][2]  # c
                U[1 * 4 + mid_group * 2 + 1][col] += strassen_A[s][3]  # d
                
                # V: 4x6 matrix, index by row j (0..3) and col k (0..5)
                # B[j, k] where j = mid_group*2 + (0 or 1), k = out_group*2 + (0 or 1)
                # In B = [[e,f],[g,h]], e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
                # Our B here is 2x2 with entries from the full 4x6 B
                # B_full[mid_group*2+0, out_group*2+0] = e
                # B_full[mid_group*2+0, out_group*2+1] = f
                # B_full[mid_group*2+1, out_group*2+0] = g
                # B_full[mid_group*2+1, out_group*2+1] = h
                
                # V indices: j*p + k = j*6 + k
                V[(mid_group * 2 + 0) * 6 + out_group * 2 + 0][col] += strassen_B[s][0]  # e
                V[(mid_group * 2 + 0) * 6 + out_group * 2 + 1][col] += strassen_B[s][1]  # f
                V[(mid_group * 2 + 1) * 6 + out_group * 2 + 0][col] += strassen_B[s][2]  # g
                V[(mid_group * 2 + 1) * 6 + out_group * 2 + 1][col] += strassen_B[s][3]  # h
                
                # W: output C is 2x6, indexed by k*n + i = k*2 + i
                # C[i, k] where i is row (0,1), k is col (0..5)
                # For this 2x2*2x2 block, output positions are:
                # C_full[0, out_group*2+0] = C00
                # C_full[0, out_group*2+1] = C01
                # C_full[1, out_group*2+0] = C10
                # C_full[1, out_group*2+1] = C11
                
                # W indices: k*n + i = k*2 + i
                # C00 contributes to C[0, out_group*2+0]: W[(out_group*2+0)*2 + 0]
                # C01 contributes to C[0, out_group*2+1]: W[(out_group*2+1)*2 + 0]
                # C10 contributes to C[1, out_group*2+0]: W[(out_group*2+0)*2 + 1]
                # C11 contributes to C[1, out_group*2+1]: W[(out_group*2+1)*2 + 1]
                
                W[(out_group * 2 + 0) * 2 + 0][col] += strassen_C[0][s]  # C00
                W[(out_group * 2 + 1) * 2 + 0][col] += strassen_C[1][s]  # C01
                W[(out_group * 2 + 0) * 2 + 1][col] += strassen_C[2][s]  # C10
                W[(out_group * 2 + 1) * 2 + 1][col] += strassen_C[3][s]  # C11
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
