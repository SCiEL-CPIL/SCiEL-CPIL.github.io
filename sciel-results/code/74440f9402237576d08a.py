"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructive rank-minimal decomposition for 2x3x3 matrix multiplication.
    
    Strategy: Decompose the tensor into:
      1) A 2x2x2 sub-tensor (i in {0,1}, j in {0,1}, k in {0,1}) using Strassen (rank 7)
      2) Terms with j=2 (all i, k): 2x1x3 schoolbook (rank 6)
      3) Terms with j in {0,1}, k=2 (all i): 2x2x1 schoolbook (rank 4)
    Total rank: 7 + 6 + 4 = 17
    """
    import time
    t_start = time.time()
    deadline = t_start + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (9, rank)
    # W has shape (p*n, rank) = (6, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen decomposition for 2x2x2 sub-tensor
    # Sub-tensor: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen's 7 terms for 2x2x2 matrix multiplication:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # where:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*B[0,1]
    # M4 = A[1,1]*B[1,0]
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # In our tensor: A is 2x2 (sub), B is 2x2 (sub), C is 2x2 (sub)
    # A[i,j] for i in {0,1}, j in {0,1} -> U index: i*m + j = i*3 + j
    # B[j,k] for j in {0,1}, k in {0,1} -> V index: j*p + k = j*3 + k
    # C[i,k] for i in {0,1}, k in {0,1} -> W index: k*n + i = k*2 + i
    
    # Strassen M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U contribution: A[0,0]=1, A[1,1]=1
    U[0*3+0][col] = 1.0
    U[1*3+1][col] = 1.0
    # V contribution: B[0,0]=1, B[1,1]=1
    V[0*3+0][col] = 1.0
    V[1*3+1][col] = 1.0
    # W contribution: C[0,0]=1, C[0,1]=-1, C[1,0]=-1, C[1,1]=1
    W[0*2+0][col] = 1.0   # k=0,i=0
    W[0*2+1][col] = -1.0  # k=0,i=1
    W[1*2+0][col] = -1.0  # k=1,i=0
    W[1*2+1][col] = 1.0   # k=1,i=1
    col += 1
    
    # Strassen M2: (A[1,0]+A[1,1]) * B[0,0]
    U[1*3+0][col] = 1.0
    U[1*3+1][col] = 1.0
    V[0*3+0][col] = 1.0
    # W: C[1,0]=1, C[1,1]=1
    W[0*2+1][col] = 1.0  # k=0,i=1
    W[1*2+1][col] = 1.0  # k=1,i=1
    col += 1
    
    # Strassen M3: A[0,0] * B[0,1]
    U[0*3+0][col] = 1.0
    V[0*3+1][col] = 1.0
    # W: C[0,1]=1, C[1,1]=-1
    W[0*2+1][col] = 1.0  # k=0,i=1
    W[1*2+1][col] = -1.0 # k=1,i=1
    col += 1
    
    # Strassen M4: A[1,1] * B[1,0]
    U[1*3+1][col] = 1.0
    V[1*3+0][col] = 1.0
    # W: C[0,0]=1, C[1,0]=1
    W[0*2+0][col] = 1.0  # k=0,i=0
    W[0*2+1][col] = 1.0  # k=0,i=1
    col += 1
    
    # Strassen M5: (A[0,0]+A[0,1]) * B[1,1]
    U[0*3+0][col] = 1.0
    U[0*3+1][col] = 1.0
    V[1*3+1][col] = 1.0
    # W: C[0,1]=1, C[1,1]=1
    W[0*2+1][col] = 1.0  # k=0,i=1
    W[1*2+1][col] = 1.0  # k=1,i=1
    col += 1
    
    # Strassen M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*3+0][col] = 1.0
    U[0*3+0][col] = -1.0
    V[0*3+0][col] = 1.0
    V[0*3+1][col] = 1.0
    # W: C[1,1]=1
    W[1*2+1][col] = 1.0  # k=1,i=1
    col += 1
    
    # Strassen M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*3+1][col] = 1.0
    U[1*3+1][col] = -1.0
    V[1*3+0][col] = 1.0
    V[1*3+1][col] = 1.0
    # W: C[0,0]=-1, C[1,1]=1
    W[0*2+0][col] = -1.0 # k=0,i=0
    W[1*2+1][col] = 1.0  # k=1,i=1
    col += 1
    
    # Part 2: j=2 terms (i in {0,1}, k in {0,1,2})
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(3):
            U[i*3+2][col] = 1.0
            V[2*3+k][col] = 1.0
            W[k*2+i][col] = 1.0
            col += 1
    
    # Part 3: j in {0,1}, k=2 terms (i in {0,1})
    # C[i,2] += A[i,j] * B[j,2]
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1.0
            V[j*3+2][col] = 1.0
            W[2*2+i][col] = 1.0
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition using numpy
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 
    # if (i,j,k) corresponds to a valid term, else 0.
    # Actually, the tensor entry T[i,j,k] = 1 for all valid (i,j,k) in 2x3x3.
    # The decomposition should satisfy: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    if error > 1e-10:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
