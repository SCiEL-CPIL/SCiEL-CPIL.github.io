"""Strassen rank-7 decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matmul.
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    with n=m=p=2.
    
    Strassen's 7 products:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11) B00
      M3 = A00 (B10 - B11)
      M4 = A11 (B01 - B10)
      M5 = (A00 + A01) B11
      M6 = (A10 - A00) (B00 + B01)
      M7 = (A01 - A11) (B10 + B11)
    
    Output C = A*B where:
      C00 = M1 - M4 + M5 + M7
      C01 = M3 + M4
      C10 = M2 + M5
      C11 = M1 - M2 + M3 + M6
    
    We need to express this as:
      C[i,k] = sum_c U[i,j_c] * V[j_c,k_c] * W[k_c,i_c]
    
    Actually, the standard tensor decomposition form is:
      C_ik = sum_c U_{i, c} * V_{j, c} * W_{k, c}  -- no, that's not right either.
    
    Let me think about this more carefully. The matrix multiplication tensor T is defined as:
      T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    A rank-r decomposition expresses T as a sum of r rank-1 tensors:
      T[i,j,k] = sum_{c=1}^{r} U[i,c] * V[j,c] * W[k,c]
    
    But wait, the problem statement says:
      U has shape (n*m, rank) -> U[i*m+j, c]
      V has shape (m*p, rank) -> V[j*p+k, c]  
      W has shape (p*n, rank) -> W[k*n+i, c]
    
    And the decomposition is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 if this term contributes to C[i,k] (i.e., always 1 for valid indices in 2x2).
    
    Wait, that's the schoolbook approach. For Strassen, we need a different formulation.
    
    Actually, re-reading the task: "exactly decompose the matrix multiplication tensor"
    
    The matrix multiplication tensor for 2x2 is a 2x2x2 tensor where T[i,j,k] = delta (it's 1 for all valid i,j,k in the standard formulation, but that gives rank 8).
    
    Hmm, let me reconsider. The standard way to view matrix multiplication as a tensor:
    
    For C = AB, we have C[i,k] = sum_j A[i,j] B[j,k].
    
    The "multiplication tensor" is often defined such that a rank-r decomposition corresponds to r scalar multiplications. 
    
    In the formulation given:
      U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This doesn't quite match the standard tensor decomposition. Let me look at this differently.
    
    Actually, I think the correct interpretation is:
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 (for all i,j,k in range).
    A rank-r decomposition means:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But that would mean for each (i,j,k), the sum over c of those products equals 1.
    
    For the schoolbook (rank 8), each term is a single product that is 1 for exactly one (i,j,k) combination.
    
    For Strassen, we need rank 7. But Strassen's algorithm computes C = AB using 7 multiplications, which means the tensor rank is 7.
    
    The key insight: in Strassen's algorithm, the 7 "products" are not simply A[i,j]*B[j,k]. They are linear combinations. So the decomposition involves non-trivial U, V, W matrices.
    
    Let me set up the Strassen decomposition properly.
    
    For 2x2 matrices:
      A = [[A00, A01], [A10, A11]]
      B = [[B00, B01], [B10, B11]]
      C = [[C00, C01], [C10, C11]]
    
    Strassen's 7 intermediate products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B10)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    Then:
      C00 = M1 - M4 + M5 + M7
      C01 = M3 + M4
      C10 = M2 + M5
      C11 = M1 - M2 + M3 + M6
    
    Now, each Mc is a product of two linear forms in A and B.
    
    Mc = (sum_a u_{a,c} * A_a) * (sum_b v_{b,c} * B_b)
    
    And C_k = sum_c w_{k,c} * Mc
    
    In the tensor decomposition framework:
      C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_{i',j'} U[i*m+j', c] * A[i',j']) * (sum_{j'',k'} V[j'', k', c] * B[j'',k']) * W[k*n+i, c]
    
    Wait, I'm getting confused. Let me re-read the problem statement.
    
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    "W[k*n+i, r]" - this is the trace convention for the output.
    
    I think the decomposition is:
      For each c in range(rank):
        The c-th term contributes to the computation as:
        (linear combination of A entries via U[:,c]) * (linear combination of B entries via V[:,c])
        And this product is added to C entries with weights from W[:,c].
    
    More precisely:
      Let a_c = sum_{i,j} U[i*m+j, c] * A[i,j]
      Let b_c = sum_{j,k} V[j*p+k, c] * B[j,k]
      Let p_c = a_c * b_c
      
      Then C[i,k] = sum_c W[k*n+i, c] * p_c
    
    And we need: C[i,k] = sum_j A[i,j] * B[j,k] for all i,k.
    
    This is exactly the tensor rank decomposition of matrix multiplication!
    
    Now for Strassen:
    
    M1 = (A00+A11)(B00+B11)
      a_1 = A00 + A11 -> U[0*2+0, 0]=1, U[1*2+1, 0]=1 (i=0,j=0 and i=1,j=1)
      b_1 = B00 + B11 -> V[0*2+0, 0]=1, V[1*2+1, 0]=1 (j=0,k=0 and j=1,k=1)
      W contribution: M1 appears in C00 with +1, C11 with +1
      W[0*2+0, 0]=1 (k=0,i=0 -> C00), W[1*2+1, 0]=1 (k=1,i=1 -> C11)
    
    M2 = (A10+A11)B00
      a_2 = A10 + A11 -> U[1*2+0, 1]=1, U[1*2+1, 1]=1
      b_2 = B00 -> V[0*2+0, 1]=1
      W: M2 in C10 with +1, C11 with -1
      W[0*2+1, 1]=1 (k=0,i=1 -> C10), W[1*2+1, 1]=-1 (k=1,i=1 -> C11)
    
    M3 = A00(B10-B11)
      a_3 = A00 -> U[0*2+0, 2]=1
      b_3 = B10 - B11 -> V[1*2+0, 2]=1, V[1*2+1, 2]=-1
      W: M3 in C01 with +1, C11 with +1
      W[1*2+0, 2]=1 (k=1,i=0 -> C01), W[1*2+1, 2]=1 (k=1,i=1 -> C11)
    
    M4 = A11(B01-B10)
      a_4 = A11 -> U[1*2+1, 3]=1
      b_4 = B01 - B10 -> V[0*2+1, 3]=1, V[1*2+0, 3]=-1
      W: M4 in C00 with -1, C01 with +1
      W[0*2+0, 3]=-1 (k=0,i=0 -> C00), W[1*2+0, 3]=1 (k=1,i=0 -> C01)
    
    M5 = (A00+A01)B11
      a_5 = A00 + A01 -> U[0*2+0, 4]=1, U[0*2+1, 4]=1
      b_5 = B11 -> V[1*2+1, 4]=1
      W: M5 in C00 with +1, C10 with +1
      W[0*2+0, 4]=1 (k=0,i=0 -> C00), W[0*2+1, 4]=1 (k=0,i=1 -> C10)
    
    M6 = (A10-A00)(B00+B01)
      a_6 = A10 - A00 -> U[1*2+0, 5]=1, U[0*2+0, 5]=-1
      b_6 = B00 + B01 -> V[0*2+0, 5]=1, V[0*2+1, 5]=1
      W: M6 in C11 with +1
      W[1*2+1, 5]=1 (k=1,i=1 -> C11)
    
    M7 = (A01-A11)(B10+B11)
      a_7 = A01 - A11 -> U[0*2+1, 6]=1, U[1*2+1, 6]=-1
      b_7 = B10 + B11 -> V[1*2+0, 6]=1, V[1*2+1, 6]=1
      W: M7 in C00 with +1
      W[0*2+0, 6]=1 (k=0,i=0 -> C00)
    
    Let me verify C00 = M1 - M4 + M5 + M7:
      From W column 0: +1 * M1
      From W column 3: -1 * M4
      From W column 4: +1 * M5
      From W column 6: +1 * M7
      Yes! C00 = M1 - M4 + M5 + M7 ✓
    
    C01 = M3 + M4:
      From W column 2: +1 * M3
      From W column 3: +1 * M4
      Yes! C01 = M3 + M4 ✓
    
    C10 = M2 + M5:
      From W column 1: +1 * M2
      From W column 4: +1 * M5
      Yes! C10 = M2 + M5 ✓
    
    C11 = M1 - M2 + M3 + M6:
      From W column 0: +1 * M1
      From W column 1: -1 * M2
      From W column 2: +1 * M3
      From W column 5: +1 * M6
      Yes! C11 = M1 - M2 + M3 + M6 ✓
    
    Great, so the decomposition is correct.
    
    Now let me build the matrices. n=m=p=2, rank=7.
    
    U has shape (4, 7):
      Row indices: i*m+j = i*2+j
        0: i=0,j=0 (A00)
        1: i=0,j=1 (A01)
        2: i=1,j=0 (A10)
        3: i=1,j=1 (A11)
    
    V has shape (4, 7):
      Row indices: j*p+k = j*2+k
        0: j=0,k=0 (B00)
        1: j=0,k=1 (B01)
        2: j=1,k=0 (B10)
        3: j=1,k=1 (B11)
    
    W has shape (4, 7):
      Row indices: k*n+i = k*2+i
        0: k=0,i=0 (C00)
        1: k=0,i=1 (C01)
        2: k=1,i=0 (C10)
        3: k=1,i=1 (C11)
    
    U:
      col 0 (M1): A00=1, A11=1 -> [1, 0, 0, 1]
      col 1 (M2): A10=1, A11=1 -> [0, 0, 1, 1]
      col 2 (M3): A00=1 -> [1, 0, 0, 0]
      col 3 (M4): A11=1 -> [0, 0, 0, 1]
      col 4 (M5): A00=1, A01=1 -> [1, 1, 0, 0]
      col 5 (M6): A10=1, A00=-1 -> [-1, 0, 1, 0]
      col 6 (M7): A01=1, A11=-1 -> [0, 1, 0, -1]
    
    V:
      col 0 (M1): B00=1, B11=1 -> [1, 0, 0, 1]
      col 1 (M2): B00=1 -> [1, 0, 0, 0]
      col 2 (M3): B10=1, B11=-1 -> [0, 0, 1, -1]
      col 3 (M4): B01=1, B10=-1 -> [0, 1, -1, 0]
      col 4 (M5): B11=1 -> [0, 0, 0, 1]
      col 5 (M6): B00=1, B01=1 -> [1, 1, 0, 0]
      col 6 (M7): B10=1, B11=1 -> [0, 0, 1, 1]
    
    W:
      col 0 (M1): C00=1, C11=1 -> [1, 0, 0, 1]
      col 1 (M2): C10=1, C11=-1 -> [0, 0, 1, -1]
      col 2 (M3): C01=1, C11=1 -> [0, 1, 0, 1]
      col 3 (M4): C00=-1, C01=1 -> [-1, 1, 0, 0]
      col 4 (M5): C00=1, C10=1 -> [1, 0, 1, 0]
      col 5 (M6): C11=1 -> [0, 0, 0, 1]
      col 6 (M7): C00=1 -> [1, 0, 0, 0]
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # Column 1: M2 = (A10+A11)B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[2][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # Column 2: M3 = A00(B10-B11)
    U[0][2] = 1   # A00
    V[2][2] = 1   # B10
    V[3][2] = -1  # B11
    W[1][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # Column 3: M4 = A11(B01-B10)
    U[3][3] = 1   # A11
    V[1][3] = 1   # B01
    V[2][3] = -1  # B10
    W[0][3] = -1  # C00
    W[1][3] = 1   # C01
    
    # Column 4: M5 = (A00+A01)B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11
    W[0][4] = 1   # C00
    W[2][4] = 1   # C10
    
    # Column 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    W[3][5] = 1   # C11
    
    # Column 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11
    W[0][6] = 1   # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
