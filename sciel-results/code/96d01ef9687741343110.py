"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    # 5 Strassen blocks (7 terms each) + 20 schoolbook terms for A[:,2]*B[2,:]
    rank = 5 * 7 + 20  # 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A row/col indices: A[i,j] -> U[i*m + j]
    # B row/col indices: B[i,j] -> V[i*p + j]
    # C row/col indices: C[i,j] -> W[j*n + i] (trace convention W[k*n+i, r] where k=col, i=row)
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # B column indices for this 2x2 block
        B00_idx = 0 * p + k0   # row 0, col k0
        B01_idx = 0 * p + k1   # row 0, col k1
        B10_idx = 1 * p + k0   # row 1, col k0
        B11_idx = 1 * p + k1   # row 1, col k1
        
        # C (output) indices: W[col * n + row]
        # C[0,k0] -> W[k0*2+0], C[0,k1] -> W[k1*2+0]
        # C[1,k0] -> W[k0*2+1], C[1,k1] -> W[k1*2+1]
        C00_idx = k0 * n + 0
        C01_idx = k1 * n + 0
        C10_idx = k0 * n + 1
        C11_idx = k1 * n + 1
        
        # A indices: U[row*m + col]
        A00_idx = 0 * m + 0  # U[0]
        A01_idx = 0 * m + 1  # U[1]
        A10_idx = 1 * m + 0  # U[3]
        A11_idx = 1 * m + 1  # U[4]
        
        # P1 = (A00 + A11) * (B00 + B11)
        U[A00_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1
        
        # P2 = A11 * (B10 - B00)
        U[A11_idx][col] = 1
        V[B10_idx][col] = 1
        V[B00_idx][col] = -1
        W[C10_idx][col] = 1
        W[C11_idx][col] = -1
        col += 1
        
        # P3 = A00 * (B01 - B11)
        U[A00_idx][col] = 1
        V[B01_idx][col] = 1
        V[B11_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1
        
        # P4 = (A00 + A01) * B11
        U[A00_idx][col] = 1
        U[A01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        W[C10_idx][col] = 1
        col += 1
        
        # P5 = (A10 - A00) * (B00 + B01)
        U[A10_idx][col] = 1
        U[A00_idx][col] = -1
        V[B00_idx][col] = 1
        V[B01_idx][col] = 1
        W[C00_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1
        
        # P6 = (A01 - A11) * (B01 + B11)
        U[A01_idx][col] = 1
        U[A11_idx][col] = -1
        V[B01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C01_idx][col] = 1
        col += 1
        
        # P7 = (A10 + A11) * B00
        U[A10_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        W[C10_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1
    
    # Schoolbook: A[i,2] * B[2,k] -> C[i,k] for i in {0,1}, k in 0..9
    for k in range(10):
        # A[0,2] * B[2,k] -> C[0,k]
        U[0 * m + 2][col] = 1  # U[2]
        V[2 * p + k][col] = 1  # V[20+k]
        W[k * n + 0][col] = 1  # W[2k]
        col += 1
        
        # A[1,2] * B[2,k] -> C[1,k]
        U[1 * m + 2][col] = 1  # U[5]
        V[2 * p + k][col] = 1  # V[20+k]
        W[k * n + 1][col] = 1  # W[2k+1]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Quick verification with random inputs (time-guarded)
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(3):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute via tensor decomposition
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u_r = U_arr[:, r]
                v_r = V_arr[:, r]
                w_r = W_arr[:, r]
                prod = np.dot(u_r, A_test.flatten()) * np.dot(v_r, B_test.flatten())
                for i in range(n):
                    for j in range(p):
                        C_decomp[i, j] += w_r[j * n + i] * prod
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                # Fallback: return schoolbook rank-60 if verification fails
                U_sb = [[0] * (n * m * p) for _ in range(n * m)]
                V_sb = [[0] * (n * m * p) for _ in range(m * p)]
                W_sb = [[0] * (n * m * p) for _ in range(p * n)]
                c = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i * m + j][c] = 1
                            V_sb[j * p + k][c] = 1
                            W_sb[k * n + i][c] = 1
                            c += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
