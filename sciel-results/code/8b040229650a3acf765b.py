import time
import numpy as np

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target rank 28: Strassen(7) + schoolbook(4) for 2x2x5 block + schoolbook(10) for 2x1x5 block
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a single schoolbook term a_ij * b_jk -> c_ik
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    # Helper: add a Strassen 2x2x2 block at offsets (i0, j0, k0)
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        U[i0*m + j0][col] = 1.0
        U[(i0+1)*m + (j0+1)][col] = 1.0
        V[j0*p + k0][col] = 1.0
        V[(j0+1)*p + (k0+1)][col] = 1.0
        W[k0*n + i0][col] = 1.0
        W[(k0+1)*n + (i0+1)][col] = 1.0
        col += 1
        
        # M2: (a10+a11)b00 -> c10 += M2, c11 -= M2
        U[(i0+1)*m + j0][col] = 1.0
        U[(i0+1)*m + (j0+1)][col] = 1.0
        V[j0*p + k0][col] = 1.0
        W[k0*n + (i0+1)][col] = 1.0
        W[(k0+1)*n + (i0+1)][col] = -1.0
        col += 1
        
        # M3: a00(b01-b11) -> c01 += M3, c11 += M3
        U[i0*m + j0][col] = 1.0
        V[j0*p + (k0+1)][col] = 1.0
        V[(j0+1)*p + (k0+1)][col] = -1.0
        W[(k0+1)*n + i0][col] = 1.0
        W[(k0+1)*n + (i0+1)][col] = 1.0
        col += 1
        
        # M4: a11(b10-b00) -> c00 += M4, c10 += M4
        U[(i0+1)*m + (j0+1)][col] = 1.0
        V[(j0+1)*p + k0][col] = 1.0
        V[j0*p + k0][col] = -1.0
        W[k0*n + i0][col] = 1.0
        W[k0*n + (i0+1)][col] = 1.0
        col += 1
        
        # M5: (a00+a01)b11 -> c00 -= M5, c01 += M5
        U[i0*m + j0][col] = 1.0
        U[i0*m + (j0+1)][col] = 1.0
        V[(j0+1)*p + (k0+1)][col] = 1.0
        W[k0*n + i0][col] = -1.0
        W[(k0+1)*n + i0][col] = 1.0
        col += 1
        
        # M6: (a10-a00)(b00+b01) -> c11 += M6
        U[(i0+1)*m + j0][col] = 1.0
        U[i0*m + j0][col] = -1.0
        V[j0*p + k0][col] = 1.0
        V[j0*p + (k0+1)][col] = 1.0
        W[(k0+1)*n + (i0+1)][col] = 1.0
        col += 1
        
        # M7: (a01-a11)(b10+b11) -> c00 += M7
        U[i0*m + (j0+1)][col] = 1.0
        U[(i0+1)*m + (j0+1)][col] = -1.0
        V[(j0+1)*p + k0][col] = 1.0
        V[(j0+1)*p + (k0+1)][col] = 1.0
        W[k0*n + i0][col] = 1.0
        col += 1
    
    # Block 1: j=0, all i,k -> schoolbook 2x1x5 = 10 terms
    for i in range(n):
        for k in range(p):
            add_schoolbook(i, 0, k)
    
    # Block 2: j=1,2; k=0,1 -> Strassen 2x2x2 = 7 terms
    add_strassen(0, 1, 0)
    
    # Block 3: j=1,2; k=2 -> schoolbook 2x2x1 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            add_schoolbook(i, j, 2)
    
    # Block 4: j=1,2; k=3,4 -> Strassen 2x2x2 = 7 terms
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Fast verification using NumPy
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i*m + j] * v[j*p + k] * w[k*n + i]
    
    if not np.allclose(tensor, np.ones((n, m, p)), atol=1e-9):
        # Fallback: pure schoolbook rank 30
        rank_school = n * m * p
        U2 = [[0.0] * rank_school for _ in range(n * m)]
        V2 = [[0.0] * rank_school for _ in range(m * p)]
        W2 = [[0.0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m + j][c2] = 1.0
                    V2[j*p + k][c2] = 1.0
                    W2[k*n + i][c2] = 1.0
                    c2 += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
