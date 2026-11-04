"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26  # Hybrid Strassen + Direct block decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: A[i,j] * B[j,k] -> C[i,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    def add_term(i, j, k, sign=1):
        nonlocal col
        U[i * m + j][col] = sign
        V[j * p + k][col] = sign
        W[k * n + i][col] = sign
        col += 1

    # 1. A11 B11 (2x2 block) using Strassen (7 terms)
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    # This requires intermediate sums, which cannot be directly represented as single U,V,W entries 
    # unless we expand them. 
    # Strassen's algorithm is a circuit, not a direct tensor decomposition of the original inputs 
    # into a single set of products unless we account for the sums.
    # However, the tensor decomposition definition requires U, V, W such that 
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k].
    # If we use Strassen, we are essentially introducing auxiliary variables. 
    # This is allowed if the "rank" is the number of multiplications of linear forms.
    # But the standard definition of tensor rank R(M) is the minimum R such that M = sum_r u_r \otimes v_r \otimes w_r.
    # Strassen's 7 multiplications are of linear combinations of inputs.
    # This corresponds to a decomposition of rank 7 for the 2x2 tensor.
    # So, we can define the 7 terms by the linear forms.
    
    # Let's explicitly define the 7 Strassen terms for the 2x2 block (indices 0,1).
    
    # T1: (A00 + A11) * (B00 + B11)
    # U part: A00 and A11 have coeff 1. V part: B00 and B11 have coeff 1.
    # This is NOT a single term in the basis of original inputs.
    # The question asks for U, V, W matrices. 
    # If U[i,j,c] is a coefficient, and we sum U*V*W, does it support linear combinations?
    # Yes, if we allow multiple entries in U and V for the same column c.
    # Example: For term c, if U[0,0,c]=1 and U[1,1,c]=1, and V[0,0,c]=1 and V[1,1,c]=1,
    # then the product is (A00+A11)(B00+B11).
    # So we can implement Strassen by setting multiple non-zero entries in U and V for a single column.
    
    # Reset col to 0 and implement Strassen for A11 B11
    
    col = 0
    
    # Strassen Terms for A[0:2, 0:2] * B[0:2, 0:2]
    # Indices i,k in {0,1}. j in {0,1}.
    
    # 1. M1 = (A00 + A11)(B00 + B11)
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1; V[1*3+1][col] = 1
    # Result C00 += M1, C11 += M1
    # Wait, Strassen outputs are combinations.
    # C00 = M1 + M4 - M5 + M7
    # This means W matrix will have coefficients.
    # W[k*n+i, c] is the coefficient of C[k,i] (using trace convention W[k*n+i]).
    # Actually, the output is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # If we set W[k,i,c] to the coefficient from Strassen, it works.
    
    # Let's map Strassen results to W.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to assign coefficients to W[0,0], W[0,1], W[1,0], W[1,1] for each M_r.
    
    # M1 (col 0)
    U[0][0] = 1; U[3+1] = 1  # A00, A11 (indices 0, 1 in 2x2 block -> 0, 1 in 3x3? No, 0,1 and 1,1)
    # A indices: (0,0), (1,1). U[0*3+0]=U[0], U[1*3+1]=U[4].
    # B indices: (0,0), (1,1). V[0*3+0]=V[0], V[1*3+1]=V[4].
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    # W for M1:
    # C00 += 1*M1 -> W[0*3+0] = W[0]
    # C11 += 1*M1 -> W[1*3+1] = W[4]
    W[0][0] = 1; W[4][0] = 1
    col += 1
    
    # M2 (col 1) = (A10 + A11) B00
    # A: (1,0), (1,1) -> U[3], U[4]
    # B: (0,0) -> V[0]
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    # W for M2:
    # C10 += 1*M2 -> W[0*3+1] = W[3]
    # C11 += -1*M2 -> W[1*3+1] = W[4]
    W[3][1] = 1; W[4][1] = -1
    col += 1
    
    # M3 (col 2) = A00 (B01 - B11)
    # A: (0,0) -> U[0]
    # B: (0,1), (1,1) -> V[1], V[4]
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    # W for M3:
    # C01 += 1*M3 -> W[1*3+0] = W[3]
    # C11 += 1*M3 -> W[1*3+1] = W[4]
    W[3][2] = 1; W[4][2] = 1
    col += 1
    
    # M4 (col 3) = A11 (B10 - B00)
    # A: (1,1) -> U[4]
    # B: (1,0), (0,0) -> V[3], V[0]
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    # W for M4:
    # C00 += 1*M4 -> W[0]
    # C10 += 1*M4 -> W[3]
    W[0][3] = 1; W[3][3] = 1
    col += 1
    
    # M5 (col 4) = (A00 + A10) B11
    # A: (0,0), (1,0) -> U[0], U[3]
    # B: (1,1) -> V[4]
    U[0][4] = 1; U[3][4] = 1
    V[4][4] = 1
    # W for M5:
    # C00 += -1*M5 -> W[0]
    # C01 += 1*M5 -> W[3]
    W[0][4] = -1; W[3][4] = 1
    col += 1
    
    # M6 (col 5) = (A10 - A00) (B00 + B01)
    # A: (1,0), (0,0) -> U[3], U[0]
    # B: (0,0), (0,1) -> V[0], V[1]
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # W for M6:
    # C11 += 1*M6 -> W[4]
    W[4][5] = 1
    col += 1
    
    # M7 (col 6) = (A11 - A10) (B10 + B11)
    # A: (1,1), (1,0) -> U[4], U[3]
    # B: (1,0), (1,1) -> V[3], V[4]
    U[4][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    # W for M7:
    # C00 += 1*M7 -> W[0]
    W[0][6] = 1
    col += 1
    
    # Now direct terms for remaining blocks.
    # A12 (2x1) * B21 (1x2) -> C11 (2x2) part? No, C[0:2, 0:2] += A12 B21.
    # A12: i in {0,1}, j=2. B21: j=2, k in {0,1}.
    # Terms:
    # 7. A02 B20 -> C00
    # 8. A02 B21 -> C01
    # 9. A12 B20 -> C10
    # 10. A12 B21 -> C11
    
    add_term(0, 2, 0, 1) # A02 B20 -> C00 (W[0])
    add_term(0, 2, 1, 1) # A02 B21 -> C01 (W[3])
    add_term(1, 2, 0, 1) # A12 B20 -> C10 (W[3])
    add_term(1, 2, 1, 1) # A12 B21 -> C11 (W[4])
    
    # A11 (2x2) * B12 (2x1) -> C12 (2x1)
    # A11: i in {0,1}, j in {0,1}. B12: j in {0,1}, k=2.
    # Terms:
    # 11. A00 B02 -> C02
    # 12. A01 B12 -> C02
    # 13. A10 B02 -> C12
    # 14. A11 B12 -> C12
    
    add_term(0, 0, 2, 1) # A00 B02 -> C02 (W[2*3+0]=W[6])
    add_term(0, 1, 2, 1) # A01 B12 -> C02 (W[6])
    add_term(1, 0, 2, 1) # A10 B02 -> C12 (W[2*3+1]=W[7])
    add_term(1, 1, 2, 1) # A11 B12 -> C12 (W[7])
    
    # A12 (2x1) * B22 (1x1) -> C12 (2x1)
    # 15. A02 B22 -> C02
    # 16. A12 B22 -> C12
    
    add_term(0, 2, 2, 1)
    add_term(1, 2, 2, 1)
    
    # A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21: i=2, j in {0,1}. B11: j in {0,1}, k in {0,1}.
    # 17. A20 B00 -> C20
    # 18. A20 B10 -> C20
    # 19. A21 B01 -> C21
    # 20. A21 B11 -> C21
    
    add_term(2, 0, 0, 1)
    add_term(2, 1, 0, 1)
    add_term(2, 0, 1, 1)
    add_term(2, 1, 1, 1)
    
    # A22 (1x1) * B21 (1x2) -> C21 (1x2)
    # 21. A22 B20 -> C20
    # 22. A22 B21 -> C21
    
    add_term(2, 2, 0, 1)
    add_term(2, 2, 1, 1)
    
    # A21 (1x2) * B12 (2x1) -> C22 (1x1)
    # 23. A20 B02 -> C22
    # 24. A21 B12 -> C22
    
    add_term(2, 0, 2, 1)
    add_term(2, 1, 2, 1)
    
    # A22 (1x1) * B22 (1x1) -> C22
    # 25. A22 B22 -> C22
    
    add_term(2, 2, 2, 1)
    
    # Total rank 26
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
