"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank 50 decomposition:
    # 4 blocks of 2x2x2 (Strassen, rank 7 each) -> 28
    # 2 blocks of 2x2x2 (Strassen, rank 7 each) -> 14
    # 8 rank-1 terms for 2x4x1 -> 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A block: rows i0, i0+1; cols j0, j0+1
    # B block: rows j0, j0+1; cols k0, k0+1
    # C block: rows i0, i0+1; cols k0, k0+1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen mults M1..M7
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2 (k=4..5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1 (k=6)
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify decomposition
    if time.time() - start_time < DEADLINE - 10:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Check: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = 0.0
                        for r in range(rank):
                            lhs += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(lhs - 1.0) > 1e-9:
                            print(f"Verification failed at i={i},j={j},k={k}: lhs={lhs}")
                            return {"U": U, "V": V, "W": W}
        except Exception as e:
            print(f"Verification error: {e}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
