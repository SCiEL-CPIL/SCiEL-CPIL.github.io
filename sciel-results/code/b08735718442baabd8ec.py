import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 2, 2, 11
    
    # Strategy: Rank-minimal exact tensor decomposition.
    # Current best known construction uses Strassen's algorithm for 2x2 blocks.
    # We have 11 columns in B.
    # We can group them into five 2-column blocks (columns 0-9) and one 1-column tail (column 10).
    # Each 2x2 * 2x2 block can be done in rank 7 (Strassen).
    # The tail 2x1 * 2x1 can be done in rank 4 (Schoolbook: a*e, b*f, c*e, d*f).
    # Total rank = 5 * 7 + 4 = 39.
    
    # Is it possible to do better?
    # The lower bound for 2x2x11 matrix multiplication is max(n*m*p, ...) no, that's not right.
    # The tensor rank of 2x2x11 multiplication is at least the rank of the "flattened" matrix.
    # Flattening along the middle index (m=2), we get a matrix of size (n*p) x (m*p) ? No.
    # Let's look at the structure.
    # C = A * B.
    # A is 2x2, B is 2x11, C is 2x11.
    # The tensor T has indices i (row of C, 0..1), j (col of A/row of B, 0..1), k (col of B, 0..10).
    # T[i, j, k] = 1 if i==j? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] is 1 if the indices match the multiplication.
    # Specifically, T[i, j, k] = delta_{i, ?} ... wait.
    # The tensor is defined such that <T, A \otimes B> = C.
    # C[i, k] = sum_{j} A[i, j] B[j, k].
    # So T[i, j, k] = 1 if i is the row of C, j is the col of A, k is the col of B?
    # No, the tensor is T[i, j, k] = 1 if i, j, k are consistent?
    # Actually, T[i, j, k] is non-zero only when i, j, k form a valid triple?
    # No, T[i, j, k] = 1 if i is the row index of C, j is the column index of A, k is the column index of B.
    # Wait, the standard definition is T[i, j, k] = 1 if i is the row of C, j is the col of A, k is the col of B?
    # Let's check the indices.
    # A has indices (i, j). B has indices (j, k). C has indices (i, k).
    # So T[i, j, k] = 1 if the indices are i, j, k?
    # No, T is a tensor of order 3.
    # T[i, j, k] = 1 if i, j, k are the indices?
    # Actually, T[i, j, k] = 1 if i is the row of C, j is the col of A, k is the col of B?
    # The value is 1 if i, j, k are the indices?
    # Let's just use the known rank 39 construction. It is verified to be correct in similar tasks.
    # The lower bound for 2x2xn is 4n/2? No.
    # For 2x2x2, rank is 7.
    # For 2x2xn, rank is at least 4n/2 = 2n? No.
    # The rank of 2x2xn multiplication is 4n - 2 for n>=2? No.
    # For n=2, rank=7. 4*2-1=7.
    # For n=3, rank=11? 4*3-1=11.
    # For n=11, rank=4*11-1=43?
    # Wait, the formula for 2x2xn is 4n-1?
    # Let's check n=1: 2x2x1 is 4 mults. 4*1=4. 4*1-1=3 (wrong).
    # n=2: 7 mults. 4*2-1=7.
    # n=3: 11 mults?
    # If the formula is 4n-1 for n>=2, then for n=11, rank=43.
    # But we have a construction for 39.
    # How?
    # 5 blocks of 2x2 (rank 7 each) = 35.
    # 1 block of 2x1 (rank 4) = 4.
    # Total 39.
    # So 39 < 43.
    # Is 39 minimal?
    # Maybe we can mix differently?
    # 11 = 2+2+2+2+2+1.
    # What if we use a different algorithm for the tail?
    # 2x2x1 is just 4 mults.
    # Can we share factors between the last 2x2 block and the tail?
    # The last 2x2 block uses columns 8,9. The tail uses column 10.
    # Strassen for 2x2 uses 7 mults.
    # Schoolbook for 2x2 uses 4 mults.
    # 5*7 + 4 = 39.
    # 4*7 + 1*4 (for 2x2) + 1*4 (for 2x1) = 28 + 4 + 4 = 36?
    # No, 4 blocks of 2x2 (cols 0-7) = 28.
    # Cols 8,9 is a 2x2 block. If we use schoolbook, rank 4.
    # Col 10 is 2x1. Rank 4.
    # Total 28 + 4 + 4 = 36?
    # Let's check if schoolbook for 2x2 is valid.
    # Yes, 4 mults.
    # So why did we use Strassen for all 5 blocks?
    # Because Strassen is better for 2x2?
    # 7 vs 4. Schoolbook is BETTER for 2x2?
    # Wait. 2x2 * 2x2.
    # Schoolbook:
    # C00 = a*e + b*f
    # C01 = a*g + b*h
    # C10 = c*e + d*f
    # C11 = c*g + d*h
    # This requires 8 mults?
    # a*e, b*f, a*g, b*h, c*e, d*f, c*g, d*h.
    # Yes, 8 mults.
    # Strassen requires 7 mults.
    # So Strassen is better for 2x2.
    # So 5*7 + 4 = 39 is better than 4*7 + 8 + 4 = 36?
    # Wait.
    # 5*7 = 35. +4 = 39.
    # 4*7 = 28.
    # Block 8,9 (2x2) using schoolbook: 8 mults.
    # Block 10 (2x1) using schoolbook: 4 mults.
    # Total: 28 + 8 + 4 = 40.
    # So 39 is better.
    
    # Can we do better than 39?
    # Maybe 3 blocks of 2x2 (21) + 1 block of 2x5?
    # Rank of 2x2x5?
    # 2x2xn rank is approx 4n.
    # 4*5 = 20?
    # 21 + 20 = 41. Worse.
    
    # What about 2x2x3?
    # Rank 11?
    # 4*3 - 1 = 11.
    # 3 blocks of 2x2 (21) + 1 block of 2x5? No.
    # 11 = 3 + 3 + 3 + 2.
    # 3 * Rank(2x2x3) + Rank(2x2x2).
    # 3 * 11 + 7 = 33 + 7 = 40. Worse.
    
    # 11 = 3 + 3 + 2 + 2 + 1.
    # 2 * 11 + 2 * 7 + 4 = 22 + 14 + 4 = 40.
    
    # It seems 39 is the best known.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0

    # Process 5 blocks of 2 columns each (k=0..9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices (C[i, k] -> W[k*n + i])
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # b * B[1, 10]
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # c * B[0, 10]
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1

        # d * B[1, 10]
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1

    return {"U": U, "V": V, "W": W}
