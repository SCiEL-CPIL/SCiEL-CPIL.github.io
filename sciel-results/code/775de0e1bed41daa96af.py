"""Matrix multiplication 2x2 * 2x14 via Strassen block decomposition."""
DIMENSIONS = [2, 2, 14]

import time

# Strassen's 7 multiplications for 2x2 * 2x2:
# Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
# Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
#
# For multiplying 2x2 matrix A by 2x14 matrix B:
# Split B into 7 blocks of 2x2: B = [B0, B1, B2, B3, B4, B5, B6]
# Result = A * B = [A*B0, A*B1, ..., A*B6]
#
# Each A*Bj uses 7 Strassen multiplications, so total rank = 7*7 = 49.
#
# The tensor decomposition has rank 49.
# U has shape (n*m, rank) = (4, 49)
# V has shape (m*p, rank) = (28, 49)
# W has shape (p*n, rank) = (28, 49)
#
# Indexing conventions:
# U[i*m + j, c] where i in [0,n), j in [0,m) -> row index i*2+j
# V[j*p + k, c] where j in [0,m), k in [0,p) -> row index j*14+k
# W[k*n + i, c] where k in [0,p), i in [0,n) -> row index k*2+i
#
# The multiplication tensor T[i,j,k] = 1 for all i,j,k.
# We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
#
# For Strassen: A*B where A is 2x2, B is 2x2.
# A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
# B indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
#
# In our notation:
# A[i,j] with i in {0,1}, j in {0,1} -> U row = i*m+j = i*2+j
# B[j,k] with j in {0,1}, k in {0,1} -> V row = j*p_local+k where p_local=2 for each block
#   But wait, V is indexed by j*p + k where p=14. For block bj, the k index is offset by bj*2.
#   So V row = j*14 + (bj*2 + k)
# Result C[i,k] with i in {0,1}, k in {0,1} -> W row = (bj*2+k)*n + i = (bj*2+k)*2+i

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2   # 7 blocks of 2x2
    
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen coefficients for each of the 7 multiplications:
    # Each multiplication M_r = (sum of A terms) * (sum of B terms)
    # contributes to output C via linear combinations.
    #
    # M1 = (a+d)(e+h)  -> C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1 (with signs)
    # Let me write out the full Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # For each multiplication M_r, we have factors:
    # A-factor: linear combination of a,b,c,d
    # B-factor: linear combination of e,f,g,h
    #
    # The rank-1 term is: (A_factor) * (B_factor)
    # U[i*m+j, c] = coefficient of A[i,j] in A_factor
    # V[j*p + k, c] = coefficient of B[j,k] in B_factor (for the appropriate block)
    # W[k*n+i, c] = coefficient of C[i,k] in the output contribution
    
    # Define the 7 Strassen multiplications as (A_coeffs, B_coeffs)
    # A_coeffs: [coeff_a, coeff_b, coeff_c, coeff_d] for A[0,0], A[0,1], A[1,0], A[1,1]
    # B_coeffs: [coeff_e, coeff_f, coeff_g, coeff_h] for B[0,0], B[0,1], B[1,0], B[1,1]
    
    strassen_mults = [
        # (A_coeffs, B_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1 = (a+d)(e+h)
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2 = (c+d)(e)
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3 = (a)(f-h)
        ([0, 0, 0, 1], [0, 0, 1, -1]),  # M4 = (d)(g-e)
        ([1, 1, 0, 0], [0, 0, 0, 1]),   # M5 = (a+b)(h)
        ([1, 0, -1, 0], [1, 1, 0, 0]),  # M6 = (c-a)(e+f)  -- wait, (c-a) or (a-c)?
        ([0, 1, 0, -1], [0, 0, 1, 1]),  # M7 = (b-d)(g+h)
    ]
    
    # Let me re-verify M6: Standard Strassen has M6 = (c-a)(e+f) or (a-c)(e+f)?
    # Different conventions exist. Let me use the one that gives correct result.
    # C[1,1] = M1 - M2 + M3 + M6
    # If M6 = (c-a)(e+f), then:
    # M1 - M2 + M3 + M6 = (a+d)(e+h) - (c+d)(e) + a(f-h) + (c-a)(e+f)
    # = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    # = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    # Let me collect:
    # ae: 1 - 1 = 0... hmm that doesn't look right.
    # Let me redo: (a+d)(e+h) = ae + ah + de + dh
    # (c+d)(e) = ce + de
    # a(f-h) = af - ah
    # (c-a)(e+f) = ce + cf - ae - af
    # Sum: ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    # = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + (af-af) + cf
    # = dh + cf
    # That's C[1,1] = d*h + c*f. Correct!
    
    # Let me verify C[0,0] = M1 + M4 - M5 + M7
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)(h) = ah + bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    # Sum: ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    # = ae + bg
    # C[0,0] = a*e + b*g. Correct!
    
    # C[0,1] = M3 + M5
    # M3 = a(f-h) = af - ah
    # M5 = (a+b)(h) = ah + bh
    # Sum: af - ah + ah + bh = af + bh
    # C[0,1] = a*f + b*h. Correct!
    
    # C[1,0] = M2 + M4
    # M2 = (c+d)(e) = ce + de
    # M4 = d(g-e) = dg - de
    # Sum: ce + de + dg - de = ce + dg
    # C[1,0] = c*e + d*g. Correct!
    
    # Great, the Strassen formulas are verified.
    
    # Now build the decomposition:
    # For each block b (0..6) and each Strassen mult r (0..6):
    #   column = b*7 + r
    #   U[i*2+j, col] = A_coeffs[i*2+j]  (since i*2+j maps to a,b,c,d in order)
    #   V[j*14 + (b*2+k), col] = B_coeffs[j*2+k]
    #   W[(b*2+k)*2 + i, col] = output coefficient
    
    # Output coefficients for each M_r:
    # C[0,0]: M1:+1, M4:+1, M5:-1, M7:+1
    # C[0,1]: M3:+1, M5:+1
    # C[1,0]: M2:+1, M4:+1
    # C[1,1]: M1:+1, M2:-1, M3:+1, M6:+1
    
    # W coefficients: W[k*n+i, col] = coefficient of C[i,k] from M_r
    # C[i,k] where i is row of result, k is col of result (within 2x2 block)
    # W row index = (b*2+k)*2 + i
    
    # For M_r, the output contribution to C[i,k]:
    output_coeffs = [
        # M1: C[0,0]+=1, C[0,1]+=0, C[1,0]+=0, C[1,1]+=1
        {0: 1, 1: 0, 2: 0, 3: 1},   # (i,k) -> coeff: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
        # M2: C[1,0]+=1, C[1,1]+=-1
        {0: 0, 1: 0, 2: 1, 3: -1},
        # M3: C[0,1]+=1, C[1,1]+=1
        {0: 0, 1: 1, 2: 0, 3: 1},
        # M4: C[0,0]+=1, C[1,0]+=1
        {0: 1, 1: 0, 2: 1, 3: 0},
        # M5: C[0,0]+=-1, C[0,1]+=1
        {0: -1, 1: 1, 2: 0, 3: 0},
        # M6: C[1,1]+=1
        {0: 0, 1: 0, 2: 0, 3: 1},
        # M7: C[0,0]+=1
        {0: 1, 1: 0, 2: 0, 3: 0},
    ]
    
    for b in range(num_blocks):
        for r in range(7):
            col = b * 7 + r
            a_coeffs = strassen_mults[r][0]  # [a, b, c, d]
            b_coeffs = strassen_mults[r][1]  # [e, f, g, h]
            
            # Fill U: row = i*2+j, where (i,j) -> a_coeffs[i*2+j]
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = a_coeffs[i * m + j]
            
            # Fill V: row = j*14 + (b*2+k), where (j,k) -> b_coeffs[j*2+k]
            for j in range(m):
                for k in range(2):
                    V[j * p + b * 2 + k][col] = b_coeffs[j * 2 + k]
            
            # Fill W: row = (b*2+k)*2 + i, where (i,k) -> output_coeffs[r]
            for i in range(n):
                for k in range(2):
                    W[(b * 2 + k) * n + i][col] = output_coeffs[r][i * 2 + k]
    
    return {"U": U, "V": V, "W": W}
