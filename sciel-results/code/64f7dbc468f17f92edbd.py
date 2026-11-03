"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,2) k-pairs."""
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Split j into {0,1} (use Strassen per k-pair) and j=2 (schoolbook)
    # For j in {0,1}: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x10 (rows j=0,1; cols k=0..9)
    # Group k into 5 pairs: (0,1),(2,3),(4,5),(6,7),(8,9)
    # Each pair gives a (2,2,2) subproblem solvable with rank 7 (Strassen)
    # Total for j in {0,1}: 5 * 7 = 35
    # For j=2: schoolbook, 2*10 = 20
    # Total rank: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # For this (2,2,2) block:
        # A indices: A[i,j] for i in {0,1}, j in {0,1} -> U row = i*m+j = i*3+j
        # B indices: B[j,k] for j in {0,1}, k in {k0,k1} -> V row = j*p+k = j*10+k
        # C indices: C[i,k] for i in {0,1}, k in {k0,k1} -> W row = k*n+i = k*2+i
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0*3+0][col] = 1   # A[0,0]
        U[1*3+1][col] = 1   # A[1,1]
        V[0*10+k0][col] = 1  # B[0,k0]
        V[1*10+k1][col] = 1  # B[1,k1]
        W[k0*2+0][col] = 1   # C[0,k0]
        W[k1*2+1][col] = 1   # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[1*3+0][col] = 1   # A[1,0]
        U[1*3+1][col] = 1   # A[1,1]
        V[0*10+k0][col] = 1  # B[0,k0]
        W[k0*2+1][col] = 1   # C[1,k0]
        W[k1*2+1][col] = -1  # C[1,k1]
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0*3+0][col] = 1   # A[0,0]
        V[0*10+k1][col] = 1   # B[0,k1]
        V[1*10+k1][col] = -1  # B[1,k1]
        W[k1*2+0][col] = 1   # C[0,k1]
        W[k1*2+1][col] = 1   # C[1,k1]
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[1*3+1][col] = 1   # A[1,1]
        V[1*10+k0][col] = 1   # B[1,k0]
        V[0*10+k0][col] = -1  # B[0,k0]
        W[k0*2+0][col] = 1   # C[0,k0]
        W[k0*2+1][col] = 1   # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0*3+0][col] = 1   # A[0,0]
        U[0*3+1][col] = 1   # A[0,1]
        V[1*10+k1][col] = 1  # B[1,k1]
        W[k0*2+0][col] = -1  # C[0,k0]
        W[k1*2+0][col] = 1   # C[0,k1]
        col += 1
        
        # M6 = (A[1,1]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[1*3+1][col] = 1   # A[1,1]
        U[0*3+0][col] = -1  # A[0,0]
        V[0*10+k0][col] = 1   # B[0,k0]
        V[0*10+k1][col] = 1   # B[0,k1]
        W[k1*2+1][col] = 1   # C[1,k1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[0*3+1][col] = 1   # A[0,1]
        U[1*3+1][col] = -1  # A[1,1]
        V[1*10+k0][col] = 1  # B[1,k0]
        V[1*10+k1][col] = 1  # B[1,k1]
        W[k0*2+0][col] = 1   # C[0,k0]
        W[k1*2+1][col] = -1  # C[1,k1]
        col += 1
    
    # Part 2: Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1   # A[i,2]
            V[2*10+k][col] = 1   # B[2,k]
            W[k*2+i][col] = 1    # C[i,k]
            col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify correctness
    def verify(U, V, W, n, m, p, max_checks=20):
        rank = len(U[0])
        for _ in range(max_checks):
            A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
            B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
            C = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            Cc = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                        vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                        acc += W[k*n+i][r] * ua * vb
                    Cc[i][k] = acc
            for i in range(n):
                for k in range(p):
                    if abs(C[i][k] - Cc[i][k]) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
