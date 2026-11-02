import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 73 solution:
    # Two Strassen blocks (rank 7 each) + Schoolbook for the rest (59 terms)
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {3,4}
    # Note: k indices are disjoint, so ranks add.
    
    total_rank = 7 + 7 + (75 - 8 - 8)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[(i_off)*m + (j_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off)][col] += -1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off)][col] += -1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[(i_off)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        col += 1

    # Add Block 1: i=0, j=0, k=0
    add_strassen(0, 0, 0)
    
    # Add Block 2: i=0, j=0, k=3
    add_strassen(0, 0, 3)
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip Block 1
                if i < 2 and j < 2 and k < 2:
                    continue
                # Skip Block 2
                if i < 2 and j < 2 and k >= 3 and k < 5:
                    continue
                U[i*m+j][col] += 1
                V[j*p+k][col] += 1
                W[k*n+i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_reconstructed[i, k] += U[i * m + j, :] @ (V[j * p + k, :] * W[k * n + i, :])
    
    error = np.max(np.abs(C_reconstructed - C_expected))
    print(f"Max reconstruction error: {error}")
    assert error < 1e-10, "Decomposition is incorrect!"
    print("Decomposition verified successfully.")
