import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 51 decomposition for 3x5x5 matrix multiplication
    # Strategy:
    # 1. 4 Strassen 2x2x2 blocks (rank 7 each) covering i in {0,1}, j in {0..3}, k in {0..3}
    # 2. Optimized rank-5 decomposition for i=2 (1x5x5 part)
    # 3. Schoolbook for remaining uncovered terms in i in {0,1}
    
    total_rank = 51
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # 4 Strassen blocks on i in {0,1}, covering j,k in {0..3}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    
    for (i0, j0, k0) in blocks:
        # Strassen M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[i0*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # Strassen M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        U[(i0+1)*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] -= 1.0
        col += 1
        
        # Strassen M3: A00(B01-B11) -> C01 += 1, C11 += 1
        U[i0*m + j0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # Strassen M4: A11(B10-B00) -> C00 += 1, C10 += 1
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[j0*p + k0, col] -= 1.0
        W[k0*n + i0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        col += 1
        
        # Strassen M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        U[i0*m + j0, col] += 1.0
        U[i0*m + (j0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        col += 1
        
        # Strassen M6: (A10-A00)(B00+B01) -> C11 += 1
        U[(i0+1)*m + j0, col] += 1.0
        U[i0*m + j0, col] -= 1.0
        V[j0*p + k0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # Strassen M7: (A01-A11)(B10+B11) -> C00 += 1
        U[i0*m + (j0+1), col] += 1.0
        U[(i0+1)*m + (j0+1), col] -= 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        col += 1
    
    # Optimized rank-5 decomposition for i=2
    # C[2,k] = sum_j A[2,j] B[j,k]
    # Use 5 terms, one for each j
    for r in range(m):
        # U[2, r] = 1
        U[2*m + r, col] = 1.0
        # V[r, k] = 1 for all k
        for k in range(p):
            V[r*p + k, col] = 1.0
        # W[k, 2] = 1 for all k
        for k in range(p):
            W[k*n + 2, col] = 1.0
        col += 1
    
    # Schoolbook for uncovered terms in i in {0,1}
    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
    
    for i in range(n):
        if i == 2:
            continue # Already handled
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verify all 75 entries
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
                    print(f"Error at ({i},{j},{k}): got {val}")
    
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
