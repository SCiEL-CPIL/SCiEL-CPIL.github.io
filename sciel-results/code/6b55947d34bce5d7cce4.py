"""Tensor decomposition for 2x2x14 matrix multiplication using Strassen per 2x2 block."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49 total
    
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 28 rows
    
    for q in range(num_blocks):
        base = q * 7
        # U indices: A[i][j] -> U[i*2+j], i,j in {0,1}
        # V indices: B[j][k] -> V[j*14 + (2q+k)], j in {0,1}, k in {0,1}
        # W indices: C[k][i] -> W[(2q+k)*2 + i], k in {0,1}, i in {0,1}
        
        # M1=(A00+A11)(B00+B11)
        c = base + 0
        U[0][c] += 1; U[3][c] += 1
        V[2*q][c] += 1; V[14+2*q+1][c] += 1
        W[2*q][c] += 1; W[(2*q+1)*2+1][c] += 1
        
        # M2=(A10+A11)(B00)
        c = base + 1
        U[2][c] += 1; U[3][c] += 1
        V[2*q][c] += 1
        W[(2*q+1)*2][c] += 1
        
        # M3=A00(B01-B11)
        c = base + 2
        U[0][c] += 1
        V[2*q+1][c] += 1; V[14+2*q+1][c] -= 1
        W[(2*q+1)*2][c] += 1; W[(2*q+1)*2+1][c] += 1
        
        # M4=A11(B10-B00)
        c = base + 3
        U[3][c] += 1
        V[14+2*q][c] += 1; V[2*q][c] -= 1
        W[2*q][c] += 1; W[(2*q+1)*2][c] += 1
        
        # M5=(A00+A01)(B11)
        c = base + 4
        U[0][c] += 1; U[1][c] += 1
        V[14+2*q+1][c] += 1
        W[2*q][c] -= 1; W[(2*q+1)*2][c] += 1
        
        # M6=(A10-A00)(B00+B10)
        c = base + 5
        U[2][c] += 1; U[0][c] -= 1
        V[2*q][c] += 1; V[14+2*q][c] += 1
        W[(2*q+1)*2+1][c] += 1
        
        # M7=(A01-A11)(B01+B11)
        c = base + 6
        U[1][c] += 1; U[3][c] -= 1
        V[2*q+1][c] += 1; V[14+2*q+1][c] += 1
        W[2*q][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
