import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Rank 19 construction:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] (2x2x2) -> Rank 7 (Strassen)
    # 2. A[0:2, 0:2] * B[0:2, 2:3] (2x2x1) -> Rank 4 (Schoolbook)
    # 3. A[0:2, 0:2] * B[0:2, 3:5] (2x2x2) -> Rank 7 (Strassen)
    # 4. A[0:2, 2:3] * B[2:3, 0:5] (2x1x5) -> Rank 1 (Outer Product)
    # Total = 19
    
    rank = 19
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # --- Part 1: Strassen 2x2x2 for A[0:2, 0:2] and B[0:2, 0:2] ---
    # Indices: A(i,j) for i in 0,1; j in 0,1
    # B(j,k) for j in 0,1; k in 0,1
    # Output C(i,k) for i in 0,1; k in 0,1
    
    # Strassen form:
    # M1 = (A00 + A11)(B01 - B10)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 + B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M5 + M4 - M3 + M7
    # C01 = M1 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M6
    
    # We will add 7 columns for this part.
    
    # Helper to add a term to U, V, W
    def add_term(i_u, j_u, i_v, j_v, i_w, j_w, val=1.0):
        nonlocal col
        U[i_u * m + j_u, col] = val
        V[i_v * p + j_v, col] = val
        W[i_w * n + j_w, col] = val
        col += 1
        
    # M1
    # U: A00, A11 (1)
    # V: B01, B10 (-1)
    # W: C01, C11 (1)
    U[0*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+1, col] = 1.0; V[1*p+0, col] = -1.0
    W[1*n+0, col] = 1.0; W[1*n+1, col] = 1.0
    col += 1
    
    # M2
    # U: A10, A11 (1)
    # V: B00 (1)
    # W: C10, C11 (-1)
    U[1*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+0, col] = 1.0
    W[0*n+1, col] = 1.0; W[1*n+1, col] = -1.0
    col += 1
    
    # M3
    # U: A00 (1)
    # V: B10, B11 (1)
    # W: C00 (-1), C01 (1)
    U[0*m+0, col] = 1.0
    V[1*p+0, col] = 1.0; V[1*p+1, col] = 1.0
    W[0*n+0, col] = -1.0; W[1*n+0, col] = 1.0
    col += 1
    
    # M4
    # U: A11 (1)
    # V: B01, B00 (-1)
    # W: C00 (1), C10 (1)
    U[1*m+1, col] = 1.0
    V[0*p+1, col] = 1.0; V[0*p+0, col] = -1.0
    W[0*n+0, col] = 1.0; W[0*n+1, col] = 1.0
    col += 1
    
    # M5
    # U: A00, A01 (1)
    # V: B11 (1)
    # W: C00 (1), C11 (1)
    U[0*m+0, col] = 1.0; U[0*m+1, col] = 1.0
    V[1*p+1, col] = 1.0
    W[0*n+0, col] = 1.0; W[1*n+1, col] = 1.0
    col += 1
    
    # M6
    # U: A10, A00 (-1)
    # V: B00, B01 (1)
    # W: C11 (1)
    U[1*m+0, col] = 1.0; U[0*m+0, col] = -1.0
    V[0*p+0, col] = 1.0; V[0*p+1, col] = 1.0
    W[1*n+1, col] = 1.0
    col += 1
    
    # M7
    # U: A01, A11 (-1)
    # V: B10, B11 (1)
    # W: C00 (1)
    U[0*m+1, col] = 1.0; U[1*m+1, col] = -1.0
    V[1*p+0, col] = 1.0; V[1*p+1, col] = 1.0
    W[0*n+0, col] = 1.0
    col += 1
    
    # --- Part 2: Schoolbook 2x2x1 for A[0:2, 0:2] and B[0:2, 2] ---
    # A(i,j) i in 0,1; j in 0,1
    # B(j, 2) j in 0,1
    # Output C(i, 2)
    
    # C0_2 = A00 B0_2 + A01 B1_2
    # C1_2 = A10 B0_2 + A11 B1_2
    
    # Term 2a: A00 * B0_2
    U[0*m+0, col] = 1.0
    V[0*p+2, col] = 1.0
    W[2*n+0, col] = 1.0
    col += 1
    
    # Term 2b: A01 * B1_2
    U[0*m+1, col] = 1.0
    V[1*p+2, col] = 1.0
    W[2*n+0, col] = 1.0
    col += 1
    
    # Term 2c: A10 * B0_2
    U[1*m+0, col] = 1.0
    V[0*p+2, col] = 1.0
    W[2*n+1, col] = 1.0
    col += 1
    
    # Term 2d: A11 * B1_2
    U[1*m+1, col] = 1.0
    V[1*p+2, col] = 1.0
    W[2*n+1, col] = 1.0
    col += 1
    
    # --- Part 3: Strassen 2x2x2 for A[0:2, 0:2] and B[0:2, 3:5] ---
    # B columns 3, 4.
    # Output C(i, 3), C(i, 4).
    
    # M1
    U[0*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+4, col] = 1.0; V[1*p+3, col] = -1.0
    W[4*n+0, col] = 1.0; W[4*n+1, col] = 1.0
    col += 1
    
    # M2
    U[1*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+3, col] = 1.0
    W[3*n+1, col] = 1.0; W[4*n+1, col] = -1.0
    col += 1
    
    # M3
    U[0*m+0, col] = 1.0
    V[1*p+3, col] = 1.0; V[1*p+4, col] = 1.0
    W[3*n+0, col] = -1.0; W[4*n+0, col] = 1.0
    col += 1
    
    # M4
    U[1*m+1, col] = 1.0
    V[0*p+4, col] = 1.0; V[0*p+3, col] = -1.0
    W[3*n+0, col] = 1.0; W[3*n+1, col] = 1.0
    col += 1
    
    # M5
    U[0*m+0, col] = 1.0; U[0*m+1, col] = 1.0
    V[1*p+4, col] = 1.0
    W[3*n+0, col] = 1.0; W[4*n+1, col] = 1.0
    col += 1
    
    # M6
    U[1*m+0, col] = 1.0; U[0*m+0, col] = -1.0
    V[0*p+3, col] = 1.0; V[0*p+4, col] = 1.0
    W[4*n+1, col] = 1.0
    col += 1
    
    # M7
    U[0*m+1, col] = 1.0; U[1*m+1, col] = -1.0
    V[1*p+3, col] = 1.0; V[1*p+4, col] = 1.0
    W[3*n+0, col] = 1.0
    col += 1
    
    # --- Part 4: Outer Product 2x1x5 for A[0:2, 2] and B[2, 0:5] ---
    # A(i, 2)
    # B(2, k)
    # C(i, k) = A(i,2) B(2,k)
    
    # Rank 1
    U[0*m+2, col] = 1.0; U[1*m+2, col] = 1.0
    V[2*p+0, col] = 1.0; V[2*p+1, col] = 1.0; V[2*p+2, col] = 1.0; V[2*p+3, col] = 1.0; V[2*p+4, col] = 1.0
    W[0*n+0, col] = 1.0; W[1*n+0, col] = 1.0; W[2*n+0, col] = 1.0; W[3*n+0, col] = 1.0; W[4*n+0, col] = 1.0
    W[0*n+1, col] = 1.0; W[1*n+1, col] = 1.0; W[2*n+1, col] = 1.0; W[3*n+1, col] = 1.0; W[4*n+1, col] = 1.0
    col += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
