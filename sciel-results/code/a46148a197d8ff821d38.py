import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 decomposition:
    # 5 blocks of Strassen (2x2x2) for j in {0,1}, k pairs (0,1)..(8,9) -> 5 * 7 = 35
    # 1 block of Schoolbook (2x1x10) for j=2 -> 20
    # Total 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    
    # Strassen for 2x2x2
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
        # M3
        U[0*3+0][col] = 1
        V[0*10+k1][col] = 1
        V[1*10+k1][col] = -1
        W[k1*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M4
        U[1*3+1][col] = 1
        V[1*10+k0][col] = 1
        V[0*10+k0][col] = -1
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M5
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # M6
        U[1*3+1][col] = 1
        U[0*3+0][col] = -1
        V[0*10+k0][col] = 1
        V[0*10+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M7
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*10+k0][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verification function with multiple random tests
    def verify(U, V, W, n, m, p, num_tests=3):
        rank = len(U[0])
        for _ in range(num_tests):
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

    if not verify(U, V, W, n, m, p, num_tests=3):
        # Fallback to schoolbook (rank 60)
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
