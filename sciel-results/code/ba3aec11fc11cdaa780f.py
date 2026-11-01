"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return Strassen's rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    The convention is:
      U[i*m + j, r]  (m=2, n=2)
      V[j*p + k, r]  (p=2, m=2)
      W[k*n + i, r]  (n=2, p=2)
    
    Strassen's 7 multiplications M_1..M_7:
      M1 = (A11 + A22) * (B11 + B22)
      M2 = (A21 + A22) * B11
      M3 = A11 * (B12 - B22)
      M4 = A22 * (B21 - B11)
      M5 = (A11 + A12) * B22
      M6 = (A21 - A11) * (B11 + B12)
      M7 = (A12 - A22) * (B21 + B22)
    
    Output:
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    # Deadline guard
    deadline = time.time() + 120.0
    
    # Strassen's rank-7 decomposition
    # Each column r corresponds to one of the 7 multiplications
    # U[row, r] = coefficient of A element in multiplication r
    # V[row, r] = coefficient of B element in multiplication r
    # W[row, r] = coefficient of C element in multiplication r
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=2
    rank = 7
    
    # U has shape (n*m, rank) = (4, 7)
    # Index: U[i*m + j, r]
    # V has shape (m*p, rank) = (4, 7)
    # Index: V[j*p + k, r]
    # W has shape (p*n, rank) = (4, 7)
    # Index: W[k*n + i, r]
    
    # Initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A indices: A[0,0]=U[0], A[1,1]=U[3]
    # B indices: B[0,0]=V[0], B[1,1]=V[3]
    # C contribution: C[0,0]=W[0], C[1,1]=W[3]
    r = 0
    U[0][r] = 1   # A11
    U[3][r] = 1   # A22
    V[0][r] = 1   # B11
    V[3][r] = 1   # B22
    W[0][r] = 1   # C11
    W[3][r] = 1   # C22
    
    # M2 = (A21 + A22) * B11
    # A indices: A[1,0]=U[2], A[1,1]=U[3]
    # B indices: B[0,0]=V[0]
    # C contribution: C[1,0]=W[2], C[1,1]=W[3]
    r = 1
    U[2][r] = 1   # A21
    U[3][r] = 1   # A22
    V[0][r] = 1   # B11
    W[2][r] = 1   # C21
    W[3][r] = -1  # C22 (with minus sign)
    
    # M3 = A11 * (B12 - B22)
    # A indices: A[0,0]=U[0]
    # B indices: B[0,1]=V[1], B[1,1]=V[3]
    # C contribution: C[0,1]=W[1], C[1,1]=W[3]
    r = 2
    U[0][r] = 1   # A11
    V[1][r] = 1   # B12
    V[3][r] = -1  # B22 (with minus sign)
    W[1][r] = 1   # C12
    W[3][r] = 1   # C22
    
    # M4 = A22 * (B21 - B11)
    # A indices: A[1,1]=U[3]
    # B indices: B[1,0]=V[2], B[0,0]=V[0]
    # C contribution: C[0,0]=W[0], C[1,0]=W[2]
    r = 3
    U[3][r] = 1   # A22
    V[2][r] = 1   # B21
    V[0][r] = -1  # B11 (with minus sign)
    W[0][r] = 1   # C11
    W[2][r] = 1   # C21
    
    # M5 = (A11 + A12) * B22
    # A indices: A[0,0]=U[0], A[0,1]=U[1]
    # B indices: B[1,1]=V[3]
    # C contribution: C[0,0]=W[0], C[0,1]=W[1]
    r = 4
    U[0][r] = 1   # A11
    U[1][r] = 1   # A12
    V[3][r] = 1   # B22
    W[0][r] = -1  # C11 (with minus sign)
    W[1][r] = 1   # C12
    
    # M6 = (A21 - A11) * (B11 + B12)
    # A indices: A[1,0]=U[2], A[0,0]=U[0]
    # B indices: B[0,0]=V[0], B[0,1]=V[1]
    # C contribution: C[1,1]=W[3]
    r = 5
    U[2][r] = 1   # A21
    U[0][r] = -1  # A11 (with minus sign)
    V[0][r] = 1   # B11
    V[1][r] = 1   # B12
    W[3][r] = 1   # C22
    
    # M7 = (A12 - A22) * (B21 + B22)
    # A indices: A[0,1]=U[1], A[1,1]=U[3]
    # B indices: B[1,0]=V[2], B[1,1]=V[3]
    # C contribution: C[0,0]=W[0], C[1,1]=W[3]
    r = 6
    U[1][r] = 1   # A12
    U[3][r] = -1  # A22 (with minus sign)
    V[2][r] = 1   # B21
    V[3][r] = 1   # B22
    W[0][r] = 1   # C11
    W[3][r] = 1   # C22
    
    # Verify before returning (with deadline guard)
    if time.time() < deadline - 5.0:
        # Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{k,k'}
        # For each (i,j,k) triple, the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] should be 1 if i'=i and k'=k, else 0
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and k == kp) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                            if abs(actual - expected) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
