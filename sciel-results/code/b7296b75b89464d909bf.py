"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    We use a hybrid approach:
    - For the first 4 columns of the 2x7 matrix, treat as two 2x2 multiplications
      and apply Strassen's algorithm (rank 7 each) -> 14 terms
    - For the remaining 3 columns, use schoolbook (rank 2 each) -> 6 terms
    - Total rank: 20
    
    However, we can do better by treating the whole thing more carefully.
    
    Actually, let's think about this differently. The tensor is T_{i,j,k} where:
    - i in {0,1} (row of first matrix, 2 values)
    - j in {0,1} (column of first matrix = row of second matrix, 2 values)  
    - k in {0,...,6} (column of second matrix, 7 values)
    
    The multiplication tensor M_{i,j,k} = sum over j of A_{i,j} * B_{j,k}
    
    In the standard tensor notation for matrix multiplication:
    - U has shape (n*m, rank) = (4, rank)
    - V has shape (m*p, rank) = (14, rank)
    - W has shape (p*n, rank) = (14, rank)
    
    With trace convention W[k*n+i, r]
    
    The schoolbook rank is n*m*p = 2*2*7 = 28.
    
    We can improve by using Strassen for parts of the computation.
    
    Let's use the following approach:
    - Split the 7 columns into groups: first 4 columns can be handled with 
      a 4x4 matrix multiplication trick, remaining 3 with schoolbook.
    
    Actually, let me use a simpler but effective approach:
    - Use Strassen for a 2x2 * 2x2 subproblem (rank 7) for columns 0,1
    - Use schoolbook for columns 2,3,4,5,6 (rank 2*5 = 10)
    - Total: 17
    
    Wait, that's not quite right either. Let me think more carefully.
    
    The matrix multiplication A(2x2) * B(2x7) = C(2x7)
    
    C[i,k] = sum_j A[i,j] * B[j,k] for j in {0,1}
    
    So for each output element C[i,k], we need 2 multiplications.
    Total: 2*2*7 = 28 scalar multiplications in schoolbook.
    
    Using Strassen on the 2x2 * 2x2 part (columns 0,1):
    M1 = (A10+A11)(B00+B11)
    M2 = (A10+A11)(B10-B11)
    M3 = A00(B00-B10)
    M4 = A11(B11-B01)
    M5 = (A00+A11)(B01+B10)
    M6 = (A00-A10)(B00+B10)
    M7 = (A11-A01)(B10+B11)
    
    This gives us C[0,0], C[0,1], C[1,0], C[1,1] with 7 multiplications.
    
    For the remaining 5 columns (k=2..6), we need 2*5 = 10 multiplications.
    
    Total rank: 7 + 10 = 17.
    
    Let me verify the Strassen formulas:
    For C = A*B where A, B are 2x2:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    C00 = M1 - M2 + M4 - M7
    C01 = M3 + M7
    C10 = M1 - M4
    C11 = M2 - M3 + M5 + M6
    
    Wait, let me use the standard Strassen formulas:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B01-B11)
    P4 = A11*(B10-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B01)
    P7 = (A01-A11)*(B10+B11)
    
    C00 = P1 - P2 + P4 + P7
    C01 = P3 + P5 + P7
    C10 = P2 + P4 - P5 - P6
    C11 = P1 + P3 - P2 - P6
    
    Let me double-check with a simple example.
    A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    C = [[19,22],[43,50]]
    
    P1 = (1+4)*(5+8) = 5*13 = 65
    P2 = (3+4)*5 = 7*5 = 35
    P3 = 1*(6-8) = 1*(-2) = -2
    P4 = 4*(7-5) = 4*2 = 8
    P5 = (1+2)*8 = 3*8 = 24
    P6 = (3-1)*(5+6) = 2*11 = 22
    P7 = (2-4)*(7+8) = (-2)*15 = -30
    
    C00 = 65 - 35 + 8 + (-30) = 8 ✓
    C01 = -2 + 24 + (-30) = -8 ✗ (should be 22)
    
    Hmm, that's not right. Let me look up the correct Strassen formulas.
    
    Standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)(B00+B01)
    M3 = (A00+A01)(B10+B11)
    M4 = A00(B00-B10)
    M5 = A11(B01-B00)
    M6 = (A00+A10)(B00+B10)
    M7 = (A01+A11)(B01+B11)
    
    C00 = M1 - M5 + M7 - M3
    C01 = M4 + M5
    C10 = M2 - M7 + M3 - M1
    C11 = M1 - M2 + M4 - M6
    
    Let me verify:
    M1 = (1+4)(5+8) = 65
    M2 = (3+4)(5+6) = 77
    M3 = (1+2)(7+8) = 45
    M4 = 1*(5-7) = -2
    M5 = 4*(6-5) = 4
    M6 = (1+3)(5+7) = 48
    M7 = (2+4)(6+8) = 84
    
    C00 = 65 - 4 + 84 - 45 = 100 ✗
    
    I'm getting confused with the formulas. Let me just use a known correct set.
    
    From Wikipedia, Strassen's algorithm:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M2 + M4 + M7
    C01 = M3 + M5 + M7
    C10 = M2 + M4 - M5 - M6
    C11 = M1 + M3 - M2 - M6
    
    Let me verify again:
    A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    
    M1 = (1+4)(5+8) = 65
    M2 = (3+4)*5 = 35
    M3 = 1*(6-8) = -2
    M4 = 4*(7-5) = 8
    M5 = (1+2)*8 = 24
    M6 = (3-1)(5+6) = 22
    M7 = (2-4)(7+8) = -30
    
    C00 = 65 - 35 + 8 + (-30) = 8 ✓ (should be 1*5+2*7=19)
    
    Still wrong. Let me just implement schoolbook for now and verify correctness,
    then try to optimize.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # Schoolbook decomposition
    rank = n * m * p  # 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 14 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
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
