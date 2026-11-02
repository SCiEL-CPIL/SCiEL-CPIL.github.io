import numpy as np
import time

def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Uses Strassen's rank-7 algorithm for the 2x2x2 block (rows 0-1, cols 0-1)
    and schoolbook for all other terms.
    Total rank: 72 - 1 = 71
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 3, 8
    
    # Use Strassen for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # and schoolbook for all other terms
    rank = 71
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 terms for 2x2x2 block
    # The 2x2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen algorithm:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor notation for our case:
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[k,i]
    
    # For the 2x2x2 block, we use indices:
    # i in {0,1} -> row index in A
    # j in {0,1} -> col index in A, row index in B
    # k in {0,1} -> col index in B
    
    # Strassen terms (0-indexed):
    # M1: A[0,0]+A[1,1] times B[0,0]+B[1,1]
    # M2: A[1,0]+A[1,1] times B[0,0]
    # M3: A[0,0] times B[1,0]-B[1,1]
    # M4: A[1,1] times B[0,1]-B[0,0]
    # M5: A[0,0]+A[0,1] times B[1,1]
    # M6: A[1,0]-A[0,0] times B[0,0]+B[0,1]
    # M7: A[0,1]-A[1,1] times B[1,0]+B[1,1]
    
    idx = 0
    
    # M1
    U[0*3 + 0, idx] = 1.0  # A[0,0]
    U[1*3 + 1, idx] = 1.0  # A[1,1]
    V[0*8 + 0, idx] = 1.0  # B[0,0]
    V[1*8 + 1, idx] = 1.0  # B[1,1]
    W[0*3 + 0, idx] = 1.0  # C[0,0]
    W[0*3 + 1, idx] = 1.0  # C[0,1] (for C11)
    W[1*3 + 0, idx] = 1.0  # C[1,0]
    W[1*3 + 1, idx] = 1.0  # C[1,1]
    idx += 1
    
    # M2
    U[1*3 + 0, idx] = 1.0  # A[1,0]
    U[1*3 + 1, idx] = 1.0  # A[1,1]
    V[0*8 + 0, idx] = 1.0  # B[0,0]
    W[1*3 + 0, idx] = 1.0  # C[1,0]
    W[1*3 + 1, idx] = -1.0  # C[1,1]
    idx += 1
    
    # M3
    U[0*3 + 0, idx] = 1.0  # A[0,0]
    V[1*8 + 0, idx] = 1.0  # B[1,0]
    V[1*8 + 1, idx] = -1.0  # B[1,1]
    W[0*3 + 1, idx] = 1.0  # C[0,1]
    W[1*3 + 1, idx] = 1.0  # C[1,1]
    idx += 1
    
    # M4
    U[1*3 + 1, idx] = 1.0  # A[1,1]
    V[0*8 + 1, idx] = 1.0  # B[0,1]
    V[0*8 + 0, idx] = -1.0  # B[0,0]
    W[0*3 + 0, idx] = 1.0  # C[0,0]
    W[1*3 + 0, idx] = 1.0  # C[1,0]
    idx += 1
    
    # M5
    U[0*3 + 0, idx] = 1.0  # A[0,0]
    U[0*3 + 1, idx] = 1.0  # A[0,1]
    V[1*8 + 1, idx] = 1.0  # B[1,1]
    W[0*3 + 0, idx] = -1.0  # C[0,0]
    W[0*3 + 1, idx] = 1.0  # C[0,1]
    idx += 1
    
    # M6
    U[1*3 + 0, idx] = 1.0  # A[1,0]
    U[0*3 + 0, idx] = -1.0  # A[0,0]
    V[0*8 + 0, idx] = 1.0  # B[0,0]
    V[0*8 + 1, idx] = 1.0  # B[0,1]
    W[1*3 + 1, idx] = 1.0  # C[1,1]
    idx += 1
    
    # M7
    U[0*3 + 1, idx] = 1.0  # A[0,1]
    U[1*3 + 1, idx] = -1.0  # A[1,1]
    V[1*8 + 0, idx] = 1.0  # B[1,0]
    V[1*8 + 1, idx] = 1.0  # B[1,1]
    W[0*3 + 0, idx] = 1.0  # C[0,0]
    idx += 1
    
    # Schoolbook terms for all other (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is part of the 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    
    # Verify idx == rank
    assert idx == rank, f"Expected rank {rank}, got {idx}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U_list]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V_list]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
