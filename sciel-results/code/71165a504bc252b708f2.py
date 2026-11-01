"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-7 Strassen decomposition for 2x2 matrix multiplication.

    The matrix multiplication tensor T_{i,j,k} = delta_{i,k} delta_{j,l} is
    decomposed as sum_{r=0}^{6} U_{i,j,r} V_{j,k,r} W_{k,i,r} with the
    official trace convention W[k*n + i, r] (n=2, so W[k*2+i, r]).

    Strassen's 7-product scheme:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11) B00
      M3 = A00 (B10 - B11)
      M4 = A11 (B01 - B00)
      M5 = (A00 + A01) B11
      M6 = (A10 - A00)(B00 + B01)
      M7 = (A01 - A11)(B10 + B11)

    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # U[i*m + j, r] extracts element A[i,j] from matrix A (row i, col j)
    # U has shape (n*m, rank) = (4, 7)
    # V[j*p + k, r] extracts element B[j,k] from matrix B (row j, col k)
    # V has shape (m*p, rank) = (4, 7)
    # W[k*n + i, r] contributes to C[i,k] (row i, col k) per trace convention
    # W has shape (p*n, rank) = (4, 7)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Indexing: A[i,j] -> U[i*m+j] = U[i*2+j]
    #            B[j,k] -> V[j*p+k] = V[j*2+k]
    #            C[i,k] -> W[k*n+i] = W[k*2+i]

    # M1 = (A00 + A11)(B00 + B11)
    # U: A00=1, A11=1 ; V: B00=1, B11=1
    # C00 += M1, C11 += M1
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1  # C[0,0]
    W[1*2+1][0] = 1  # C[1,1]

    # M2 = (A10 + A11) B00
    # U: A10=1, A11=1 ; V: B00=1
    # C10 += M2, C11 -= M2
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1  # C[1,0]
    W[1*2+1][1] = -1 # C[1,1]

    # M3 = A00 (B10 - B11)
    # U: A00=1 ; V: B10=1, B11=-1
    # C01 += M3, C11 += M3
    U[0*2+0][2] = 1
    V[1*2+0][2] = 1; V[1*2+1][2] = -1
    W[0*2+0][2] = 1  # C[0,1]
    W[1*2+1][2] = 1  # C[1,1]

    # M4 = A11 (B01 - B00)
    # U: A11=1 ; V: B01=1, B00=-1
    # C00 += M4, C10 += M4
    U[1*2+1][3] = 1
    V[0*2+1][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = 1  # C[0,0]
    W[0*2+1][3] = 1  # C[1,0]

    # M5 = (A00 + A01) B11
    # U: A00=1, A01=1 ; V: B11=1
    # C00 -= M5, C01 += M5
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1 # C[0,0]
    W[0*2+0][4] = -1 # C[0,1]  -- wait, W[0*2+0] is C[0,0], W[1*2+0] is C[0,1]
    # Let me re-check: W[k*n+i] where k is col of C, i is row of C.
    # C[0,1] means row=0, col=1, so k=1, i=0 -> W[1*2+0] = W[2]
    # C[0,0] means row=0, col=0, so k=0, i=0 -> W[0*2+0] = W[0]
    W[0*2+0][4] = -1 # C[0,0]
    W[1*2+0][4] = 1  # C[0,1]

    # M6 = (A10 - A00)(B00 + B01)
    # U: A10=1, A00=-1 ; V: B00=1, B01=1
    # C11 += M6
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1  # C[1,1]

    # M7 = (A01 - A11)(B10 + B11)
    # U: A01=1, A11=-1 ; V: B10=1, B11=1
    # C00 -= M7
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = -1 # C[0,0]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
