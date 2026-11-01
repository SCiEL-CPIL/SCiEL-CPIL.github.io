"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition based on 2x2 Strassen block + naive borders
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term: U[i, col] += u, V[j, col] += v, W[k, col] += w
    # i is index in A (n*m), j is index in B (m*p), k is index in C (p*n)
    # W index convention: W[k*n + i, r] where C[i, k] = sum U[i, r] V[k, r] W[i, r]? 
    # Task says: W[k*n+i, r]. 
    # Standard tensor: C_{ik} = sum_r U_{ir} V_{kr} W_{ir}? No.
    # Usually C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T_{ijk} = delta.
    # Decomposition: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}? No, indices must match.
    # Let's check the seed:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This implies C_{ik} = sum_r U_{i*m + j, r} V_{j*p + k, r} W_{k*n + i, r}?
    # No, the sum is over r.
    # The term is U_{row_U, r} * V_{row_V, r} * W_{row_W, r}.
    # For schoolbook:
    # U_{i*m+j, r} is 1 if r corresponds to (i,j,k).
    # V_{j*p+k, r} is 1 if r corresponds to (i,j,k).
    # W_{k*n+i, r} is 1 if r corresponds to (i,j,k).
    # So C_{ik} = sum_j sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # This matches C_{ik} = sum_j A_{ij} B_{jk}.
    
    # So for a term corresponding to product A_{ij} B_{jk} contributing to C_{ik}:
    # U[i*m+j, col] += coeff_A
    # V[j*p+k, col] += coeff_B
    # W[k*n+i, col] += coeff_C
    
    col = 0
    def add_term(i, j, k, u, v, w):
        nonlocal col
        U[i * m + j][col] += u
        V[j * p + k][col] += v
        W[k * n + i][col] += w
        col += 1

    # 1. Strassen 2x2 for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (a11 + a22)(b11 + b22)
    add_term(0, 0, 0, 1, 1, 1) # a11 b11 -> c11
    add_term(0, 1, 1, 1, 1, 1) # a11 b22 -> c12
    add_term(1, 0, 0, 1, 1, 1) # a22 b11 -> c21
    add_term(1, 1, 1, 1, 1, 1) # a22 b22 -> c22
    # Wait, Strassen M1 is (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22.
    # This single multiplication produces 4 terms.
    # In the tensor decomposition, each multiplication is a column.
    # The coefficients in U, V, W are the linear combination coefficients.
    # For M1:
    # U[0,0] += 1 (a11)
    # U[1,1] += 1 (a22)
    # V[0,0] += 1 (b11)
    # V[1,1] += 1 (b22)
    # W[0,0] += 1 (c11)
    # W[1,1] += 1 (c12)
    # W[0,1] += 1 (c21)
    # W[1,0] += 1 (c22)
    # Note: W index is k*n + i.
    # c11 is i=0, k=0 -> W[0]
    # c12 is i=0, k=1 -> W[1]
    # c21 is i=1, k=0 -> W[3] (since n=3, k*n+i = 0*3+1=1? No. k*n+i. c21 is i=1, k=0. 0*3+1=1. Wait.
    # C is 3x3.
    # C[0,0] -> W[0*3+0] = W[0]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[1,1] -> W[1*3+1] = W[4]
    
    # Let's restart the Strassen terms carefully.
    # M1: (a00+a11)(b00+b11)
    add_term(0, 0, 0, 1, 1, 1) # a00 b00 -> c00 (W[0])
    add_term(0, 1, 1, 1, 1, 1) # a00 b11 -> c01 (W[3])
    add_term(1, 0, 0, 1, 1, 1) # a11 b00 -> c10 (W[1])
    add_term(1, 1, 1, 1, 1, 1) # a11 b11 -> c11 (W[4])
    # This is 4 columns? No, Strassen M1 is ONE column with multiple non-zeros in U, V, W.
    # My add_term function adds to a specific column.
    # I need to define one column for M1.
    
    # Let's rewrite using direct column assignment.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_term(col, i, j, k, u, v, w):
        U[i * m + j][col] += u
        V[j * p + k][col] += v
        W[k * n + i][col] += w

    # M1: (a00+a11)(b00+b11)
    set_term(0, 0, 0, 0, 1, 1, 1) # a00 b00 -> c00
    set_term(0, 0, 1, 1, 1, 1, 1) # a00 b11 -> c01
    set_term(0, 1, 0, 0, 1, 1, 1) # a11 b00 -> c10
    set_term(0, 1, 1, 1, 1, 1, 1) # a11 b11 -> c11
    
    # M2: (a10+a11)b00
    set_term(1, 1, 0, 0, 1, 1, 1) # a10 b00 -> c10
    set_term(1, 1, 1, 0, 1, 1, 1) # a11 b00 -> c11
    
    # M3: a00(b11-b01)
    set_term(2, 0, 0, 1, 1, -1, 1) # a00 b01 -> c01
    set_term(2, 0, 1, 1, 1, 1, 1)  # a00 b11 -> c02? No.
    # Wait. a00 b11 contributes to c01.
    # a00 b01 contributes to c00? No.
    # A[0,0] B[0,1] -> C[0,1].
    # A[0,0] B[1,1] -> C[0,2].
    # M3 = a00(b11 - b01).
    # b11 is B[1,1]. b01 is B[0,1].
    # a00 b11 -> C[0,2] (i=0, k=2). W[2*3+0] = W[6].
    # a00 b01 -> C[0,1] (i=0, k=1). W[1*3+0] = W[3].
    # So:
    set_term(2, 0, 0, 1, 1, -1, 1) # a00 b01 -> c01 (W[3])
    set_term(2, 0, 1, 1, 1, 1, 1)  # a00 b11 -> c02 (W[6])
    
    # M4: a11(b10-b00)
    # a11 b10 -> C[1,2] (i=1, k=2). W[2*3+1] = W[7].
    # a11 b00 -> C[1,1] (i=1, k=1). W[1*3+1] = W[4].
    set_term(3, 1, 0, 0, 1, -1, 1) # a11 b00 -> c11 (W[4])
    set_term(3, 1, 1, 1, 1, 1, 1)  # a11 b10 -> c12 (W[7])
    
    # M5: (a00+a01)b11
    # a00 b11 -> C[0,2] (W[6])
    # a01 b11 -> C[0,3]? No, k=1. C[0,1] (W[3]).
    # Wait. B[1,1] is at j=1, k=1.
    # A[0,0] B[1,1] -> C[0,1].
    # A[0,1] B[1,1] -> C[0,2].
    set_term(4, 0, 0, 1, 1, 1, 1) # a00 b11 -> c01 (W[3])
    set_term(4, 0, 1, 1, 1, 1, 1) # a01 b11 -> c02 (W[6])
    
    # M6: (a10-a00)(b00+b01)
    # a10 b00 -> C[1,0] (W[1])
    # a10 b01 -> C[1,1] (W[4])
    # a00 b00 -> C[0,0] (W[0])
    # a00 b01 -> C[0,1] (W[3])
    set_term(5, 1, 0, 0, 1, 1, 1) # a10 b00 -> c10 (W[1])
    set_term(5, 1, 0, 1, 1, 1, 1) # a10 b01 -> c11 (W[4])
    set_term(5, 0, 0, 0, -1, 1, 1) # a00 b00 -> c00 (W[0])
    set_term(5, 0, 0, 1, -1, 1, 1) # a00 b01 -> c01 (W[3])
    
    # M7: (a01-a11)(b10+b11)
    # a01 b10 -> C[0,2] (W[6])
    # a01 b11 -> C[0,3]? No. B[1,1] -> k=1. C[0,1] (W[3]).
    # a11 b10 -> C[1,2] (W[7])
    # a11 b11 -> C[1,1] (W[4]).
    set_term(6, 0, 1, 0, 1, 1, 1) # a01 b10 -> c02 (W[6])
    set_term(6, 0, 1, 1, 1, 1, 1) # a01 b11 -> c01 (W[3])
    set_term(6, 1, 1, 0, -1, 1, 1) # a11 b10 -> c12 (W[7])
    set_term(6, 1, 1, 1, -1, 1, 1) # a11 b11 -> c11 (W[4])

    # 8-11: A[0:2, 0:2] * b (col 2 of B)
    # b = [b02, b12]^T
    # M8 = a00 b02 -> C[0,2] (W[6])
    set_term(7, 0, 0, 2, 1, 1, 1)
    # M9 = a01 b12 -> C[0,2] (W[6])
    set_term(8, 0, 1, 2, 1, 1, 1)
    # M10 = a10 b02 -> C[1,2] (W[7])
    set_term(9, 1, 0, 2, 1, 1, 1)
    # M11 = a11 b12 -> C[1,2] (W[7])
    set_term(10, 1, 1, 2, 1, 1, 1)

    # 12-15: a^T (row 2 of A) * B[0:2, 0:2]
    # a^T = [a20, a21]
    # M12 = a20 b00 -> C[2,0] (W[0*3+2]=W[2])
    set_term(11, 2, 0, 0, 1, 1, 1)
    # M13 = a21 b10 -> C[2,0] (W[2])
    set_term(12, 2, 1, 0, 1, 1, 1)
    # M14 = a20 b01 -> C[2,1] (W[1*3+2]=W[5])
    set_term(13, 2, 0, 1, 1, 1, 1)
    # M15 = a21 b11 -> C[2,1] (W[5])
    set_term(14, 2, 1, 1, 1, 1, 1)

    # 16-17: a (col 2 of A) * b22
    # a = [a02, a12]^T
    # M16 = a02 b22 -> C[0,2] (W[6])
    set_term(15, 0, 2, 2, 1, 1, 1)
    # M17 = a12 b22 -> C[1,2] (W[7])
    set_term(16, 1, 2, 2, 1, 1, 1)

    # 18-19: a22 * c^T (row 2 of B)
    # c^T = [b20, b21]
    # M18 = a22 b20 -> C[2,0] (W[2])
    set_term(17, 2, 2, 0, 1, 1, 1)
    # M19 = a22 b21 -> C[2,1] (W[5])
    set_term(18, 2, 2, 1, 1, 1, 1)

    # 20-21: a^T (row 2 of A) * b (col 2 of B, top 2)
    # M20 = a20 b02 -> C[2,2] (W[2*3+2]=W[8])
    set_term(19, 2, 0, 2, 1, 1, 1)
    # M21 = a21 b12 -> C[2,2] (W[8])
    set_term(20, 2, 1, 2, 1, 1, 1)

    # 22: a22 * b22
    # M22 = a22 b22 -> C[2,2] (W[8])
    set_term(21, 2, 2, 2, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
