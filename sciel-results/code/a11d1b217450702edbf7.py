"""Program for matrix multiplication 2x4 by 4x6 using Winograd/Strassen decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 2x4x6 multiplication tensor as a Kronecker product:
    # (2x2x2 Strassen) ⊗ (1x2x3 identity)
    # This gives rank 7 * 1 = 7.
    
    # Strassen 2x2x2 decomposition factors:
    # U_s: 2x7, V_s: 2x7, W_s: 2x7
    # Using the standard Strassen decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # The tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k (the identity tensor for matrix mult)
    
    # Strassen U (left factor, size 2x7):
    # U[i, r] corresponds to coefficient of A[i, j] in M_r
    U_s = [
        # M1  M2  M3  M4  M5  M6  M7
        [1,  0,  1,  0,  1, -1,  0],  # A11
        [1,  1,  0,  1,  0,  1, -1],  # A22? Wait, let me be more careful.
    ]
    
    # Let me use the standard Strassen formula properly.
    # For C = A * B where A, B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M3 - M4 - M7
    # C12 = M5 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M5 + M6
    
    # U[i, r]: coefficient of A[i,j] in M_r. But U is indexed by row of A.
    # Actually, the tensor is T[i, j, k] where i indexes rows of A, j indexes columns of A (=rows of B), k indexes columns of B.
    # So T[i, j, k] = 1 means: the product entry C[i,k] gets contribution 1 from A[i,j]*B[j,k].
    
    # For Strassen, we need U (2x7), V (2x7), W (2x7) such that:
    # sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k in {0,1}
    
    # Let me look up the standard Strassen tensor decomposition.
    # The rank-7 decomposition of the 2x2x2 matrix multiplication tensor:
    
    # U (rows of A): 
    #   r0: [1, 0]  -> a
    #   r1: [0, 1]  -> d
    #   r2: [1, 0]  -> a
    #   r3: [0, 1]  -> d
    #   r4: [1, 0]  -> a
    #   r5: [1, -1] -> a-c
    #   r6: [0, -1] -> -d
    # Wait, this isn't right either. Let me be very explicit.
    
    # M1 = (a+d)(e+h): A contributes [a+d], B contributes [e+h]
    # M2 = (c+d)e: A contributes [0, c+d]... no.
    
    # Let me define it differently. For each r:
    # U[i, r] = coefficient of A[i, *] in the r-th bilinear form.
    # But A is 2x2, so each A[i,*] is a row. The bilinear form is linear in each entry of A and B separately.
    
    # Standard Strassen:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - g)
    # m4 = (d)(e - g)
    # m5 = (a + b)(h)
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # U[i, r]: for row i of A, the coefficients in each mr:
    # i=0 (row [a,b]): 
    #   m1: a+0b -> coeff of a is 1, coeff of b is 0. But U is 2x7, so U[0,r] should be the coefficient for the whole row? 
    #   No, U[i,r] is a scalar. The tensor entry is T[i,j,k]. 
    #   For the tensor decomposition, U[i,r] V[j,r] W[k,r] summed over r should equal 1.
    #   Here i is row of A, j is col of A / row of B, k is col of B.
    
    # So for m1 = (a+d)(e+h):
    # This contributes to C[0,0], C[0,1], C[1,0], C[1,1] with specific coefficients.
    # Actually, m1 is used in C11 and C22.
    
    # Let me just use the known Strassen U, V, W matrices:
    # U (2x7) - rows are indexed by i (0,1), columns by r (0..6):
    U_s = [
        [1,  0,  1,  0,  1, -1,  0],  # i=0: a appears in m1, m3, m5, m6
        [1,  1,  0,  1,  0,  0, -1],  # i=1: d appears in m1, m2, m4, m7; c in m2, m6; b in m5, m7
    ]
    # Hmm, this is getting complicated. Let me use a verified Strassen decomposition.
    
    # Verified Strassen 2x2x2 rank-7 decomposition:
    # U[i, r] for i in {0,1}, r in {0,...,6}
    U_s = [
        [1,  0,  1,  0,  1, -1,  0],
        [1,  1,  0,  1,  0,  1, -1],
    ]
    # V[j, r] for j in {0,1}, r in {0,...,6}
    V_s = [
        [1,  1,  0,  1,  0,  1,  0],
        [1,  0,  1, -1, 1,  0,  1],
    ]
    # W[k, r] for k in {0,1}, r in {0,...,6}
    W_s = [
        [1,  0, -1, -1, 0,  0, -1],
        [1, -1,  1,  1, 1,  1,  1],
    ]
    
    # Let me verify this quickly: T[0,0,0] = sum_r U_s[0,r]*V_s[0,r]*W_s[0,r]
    # = 1*1*1 + 0*1*0 + 1*0*(-1) + 0*1*(-1) + 1*0*0 + (-1)*1*0 + 0*0*(-1)
    # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1. Good.
    
    # T[0,0,1] = sum_r U_s[0,r]*V_s[0,r]*W_s[1,r]
    # = 1*1*1 + 0*1*(-1) + 1*0*1 + 0*1*1 + 1*0*1 + (-1)*1*1 + 0*0*1
    # = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0. Bad! Should be 1.
    
    # My Strassen matrices are wrong. Let me use a correct one.
    
    # Correct Strassen decomposition (from standard references):
    # Let me use the formulation where:
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-g), m4 = d(e-g), m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    
    # C[0,0] = m1 + m3 - m4 - m7
    # C[0,1] = m5 + m7
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m5 + m6
    
    # U[i, r]: coefficient of A[i, j] (for any j) in mr. But since U is 2x7, each entry U[i,r] is the coefficient for the entire row i of A in mr.
    # But mr is a bilinear form, linear in each A[i,j] and B[j,k] separately.
    
    # For m1 = (a+d)(e+h): 
    #   In A: a has coeff 1, b has coeff 0, c has coeff 0, d has coeff 1
    #   In B: e has coeff 1, f has coeff 0, g has coeff 0, h has coeff 1
    #   So U[0,0]=1 (for a), U[1,0]=1 (for d)... but U is 2x7, not 4x7.
    
    # I think the issue is that the tensor T[i,j,k] has i indexing rows of A (2 values), j indexing columns of A/rows of B (2 values), k indexing columns of B (2 values).
    # So U should be 2x7 where U[i,r] is the coefficient for row i of A. But each row of A has 2 entries, so this doesn't work directly.
    
    # Actually, I think the standard tensor decomposition for matrix multiplication uses:
    # U of size (n*m) x rank, V of size (m*p) x rank, W of size (p*n) x rank.
    # For 2x2x2: U is 2x7? No, U is (n*m) x rank = 4x7. V is (m*p) x rank = 4x7. W is (p*n) x rank = 4x7.
    
    # Wait, the problem says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # For n=2, m=2, p=2: U is 4x7, V is 4x7, W is 4x7.
    
    # Let me redo this. For the 2x2x2 case:
    # U[i*m+j, r] = U[i*m+j, r]
    # V[j*p+k, r] = V[j*p+k, r]
    # W[k*n+i, r] = W[k*n+i, r]
    
    # For Strassen 2x2x2, the standard rank-7 decomposition:
    
    # Let me use the known correct Strassen U, V, W for 2x2x2:
    # U (4x7):
    # Row 0 (i=0,j=0, a): m1:1, m2:0, m3:1, m4:0, m5:1, m6:-1, m7:0
    # Row 1 (i=0,j=1, b): m1:0, m2:0, m3:0, m4:0, m5:1, m6:0, m7:1
    # Row 2 (i=1,j=0, c): m1:0, m2:1, m3:0, m4:0, m5:0, m6:1, m7:0
    # Row 3 (i=1,j=1, d): m1:1, m2:1, m3:0, m4:1, m5:0, m6:0, m7:-1
    
    U_s = [
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1],
    ]
    
    # V (4x7):
    # Row 0 (j=0,k=0, e): m1:1, m2:1, m3:0, m4:1, m5:0, m6:1, m7:0
    # Row 1 (j=0,k=1, f): m1:0, m2:0, m3:1, m4:0, m5:0, m6:1, m7:0
    # Row 2 (j=1,k=0, g): m1:0, m2:0, m3:-1, m4:-1, m5:0, m6:0, m7:1
    # Row 3 (j=1,k=1, h): m1:1, m2:0, m3:0, m4:0, m5:1, m6:0, m7:1
    
    V_s = [
        [1,  1,  0,  1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0, -1, -1, 0,  0, 1],
        [1,  0,  0,  0,  1,  0,  1],
    ]
    
    # W (4x7):
    # Row 0 (k=0,i=0): C[0,0] = m1+m3-m4-m7: m1:1, m2:0, m3:1, m4:-1, m5:0, m6:0, m7:-1
    # Row 1 (k=1,i=0): C[0,1] = m5+m7: m1:0, m2:0, m3:0, m4:0, m5:1, m6:0, m7:1
    # Row 2 (k=0,i=1): C[1,0] = m2+m4: m1:0, m2:1, m3:0, m4:1, m5:0, m6:0, m7:0
    # Row 3 (k=1,i=1): C[1,1] = m1-m2+m5+m6: m1:1, m2:-1, m3:0, m4:0, m5:1, m6:1, m7:0
    
    W_s = [
        [1,  0,  1, -1, 0,  0, -1],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [1, -1, 0,  0,  1,  1,  0],
    ]
    
    # Verify T[0,0,0] = U_s[0,0]*V_s[0,0]*W_s[0,0] + ... = 1*1*1 + 0*1*0 + 1*0*1 + 0*1*(-1) + 1*0*0 + (-1)*1*0 + 0*0*(-1) = 1
    # T[0,1,0] = U_s[0,0]*V_s[1,0]*W_s[0,0] + ... = 1*0*1 + 0*0*0 + 1*1*1 + 0*0*(-1) + 1*0*0 + (-1)*1*0 + 0*0*(-1) = 1. Good.
    
    # Now for the full 2x4x6 problem:
    # n=2, m=4, p=6
    # We decompose as (2x2x2) ⊗ (1x2x3)
    # The 1x2x3 identity tensor has rank 1: U_i[0,0]=1, V_i[j,0]=1 for j=0,1, W_i[k,0]=1 for k=0,1,2
    
    # Kronecker product:
    # U_full has size (n*m) x rank = 8 x 7
    # V_full has size (m*p) x rank = 24 x 7
    # W_full has size (p*n) x rank = 12 x 7
    
    # U_full[i*2+j1, r] where i in {0,1}, j1 in {0,1} (from the 2x2 block), and the 1x2x3 part contributes j2 in {0,1}
    # Actually, m=4 = 2*2, so j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}
    # p=6 = 2*3, so k = k1*3 + k2 where k1 in {0,1}, k2 in {0,1,2}
    # n=2 = 2*1, so i = i1*1 + i2 where i1 in {0,1}, i2=0
    
    # U_full[i, j, r] = U_s[i1*2+j1, r] * U_i[i2, 0] = U_s[i1*2+j1, r] * 1
    # But U is indexed by (i*m + j) = (i*4 + j)
    # i = i1 (since n=2, i in {0,1})
    # j = j1*2 + j2
    # So U_full[i*4 + j1*2 + j2, r] = U_s[i*2 + j1, r]
    
    rank = 7
    n, m, p = 2, 4, 6
    
    U_full = [[0]*rank for _ in range(n*m)]  # 8x7
    V_full = [[0]*rank for _ in range(m*p)]  # 24x7
    W_full = [[0]*rank for _ in range(p*n)]  # 12x7
    
    for i in range(n):
        for j1 in range(2):
            for j2 in range(2):
                j = j1 * 2 + j2
                for r in range(rank):
                    U_full[i*m + j][r] = U_s[i*2 + j1][r]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(3):
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    for r in range(rank):
                        V_full[j*p + k][r] = V_s[j1*2 + k1][r]
    
    for k1 in range(2):
        for k2 in range(3):
            for i in range(n):
                k = k1 * 3 + k2
                for r in range(rank):
                    W_full[k*n + i][r] = W_s[k1*2 + i][r]
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
