import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # If we ever run out of time, return trivial decomposition
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # n=2, m=2, p=9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # The multiplication tensor M_{2,2,9} can be decomposed by splitting the columns.
    # Rank of M_{2,2,k} is known for small k.
    # M_{2,2,1} has rank 4.
    # M_{2,2,2} has rank 7 (Strassen).
    # M_{2,2,3} has rank 11 (Strassen 2,2,1 -> 7+4).
    # M_{2,2,4} has rank 14 (2 * Strassen).
    # M_{2,2,5} has rank 18 (Strassen 2,2,3 -> 7+11).
    
    # We need M_{2,2,9}.
    # Split 9 into 4 + 5.
    # Rank = 14 + 18 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a Strassen block for 2x2 matrix multiplication
    # Maps A[i,j] (i,j in 0..1), B[j,k] (j in 0..1, k in k0..k0+1) to C[i,k]
    def add_strassen(k0, k1):
        nonlocal col
        # 1. (a00+a11)(b00+b11) -> C00, C11
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # 2. (a10+a11)b00 -> C10, C11 (with -1 on C11)
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        
        # 3. a00(b01-b11) -> C01, C11
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # 4. a11(b10-b00) -> C00, C10
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        
        # 5. (a00+a01)b11 -> C00, C01 (with -1 on C00)
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01) -> C11
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11) -> C00
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    # Helper: add a trivial block for 2x1 matrix multiplication
    # Maps A[i,j] (i,j in 0..1), B[j,k] (j in 0..1, k=k) to C[i,k]
    def add_trivial(k):
        nonlocal col
        # 1. a00*b0k -> C0k
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 2. a01*b1k -> C0k
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 3. a10*b0k -> C1k
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # 4. a11*b1k -> C1k
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Split p=9 into 4 and 5
    # Block 1: k in 0..3 (2 Strassen blocks)
    add_strassen(0, 1)
    add_strassen(2, 3)
    
    # Block 2: k in 4..8 (Strassen for 4,5 and Trivial for 6, Strassen for 7,8)
    # M_{2,2,3} (cols 4,5,6) = M_{2,2,2} (cols 4,5) + M_{2,2,1} (col 6)
    add_strassen(4, 5)
    add_trivial(6)
    # M_{2,2,2} (cols 7,8)
    add_strassen(7, 8)
    
    # Full verification
    if time.time() > deadline:
        return trivial_decomposition()
        
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k] should be 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return trivial_decomposition()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
