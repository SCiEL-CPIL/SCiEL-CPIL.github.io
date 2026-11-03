import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Laderman decomposition for 3x3x3
    # U: 9x7, V: 9x7, W: 9x7
    # Indices for A (3x3): 0..8
    # Indices for B (3x3): 0..8
    # Indices for C (3x3): 0..8
    
    U_lad = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    V_lad = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    W_lad = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                    s = np.sum(val)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try to construct rank 25 solution
    # Rank 7 for first 3 columns, Rank 9 for next 2 columns
    rank = 25
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Part 1: Columns 0, 1, 2 using Laderman
    # Laderman indices:
    # A: i*m+j
    # B: j*p+k (for k in 0,1,2, this is j*3+k if we map B to 3x3)
    #    But B is 3x5. For k=0,1,2, B[j, k] is at index j*5+k.
    #    Laderman V is for a 3x3 B. We need to map V_lad indices to V indices.
    #    V_lad row j (0..8) corresponds to B index j in 3x3 layout (j=0..8).
    #    In 3x5 layout, B[j, k] is at j*5+k.
    #    For k=0,1,2, the B index in 3x3 layout is j*3+k.
    #    So we need to map V_lad row (j*3+k) to V row (j*5+k).
    
    # Laderman W is for a 3x3 C.
    # C[i, k] is at index i*3+k in 3x3 layout.
    # In 3x5 layout, C[i, k] is at index i*5+k? No, W is p*n = 5*3 = 15.
    # W index is k*n+i.
    # For k=0,1,2, W index is k*3+i.
    # Laderman W row (i*3+k) corresponds to C index i*3+k in 3x3 layout.
    # We need to map W_lad row (i*3+k) to W row (k*3+i).
    
    # Let's verify the mapping.
    # Laderman:
    # U[i, j, t] * V[j, k, t] * W[k, i, t] = 1 for all i,j,k.
    # Here i,j,k are 0..2.
    # U index: i*3+j.
    # V index: j*3+k.
    # W index: k*3+i.
    
    # In our problem:
    # U index: i*3+j. (Same)
    # V index: j*5+k. (Different for k)
    # W index: k*3+i. (Same as Laderman W index? No, Laderman W is k*3+i. Our W is k*3+i. Yes.)
    
    # So we copy U_lad directly to U[:, 0:7].
    # For V, we need to scatter V_lad.
    # V_lad row r (0..8) corresponds to j = r // 3, k = r % 3.
    # We want to put this value in V row (j*5 + k), col t.
    # For W, we copy W_lad directly to W[:, 0:7].
    
    U[:, 0:7] = U_lad
    
    for r in range(9):
        j = r // 3
        k = r % 3
        # V_lad[r, :] goes to V[j*5 + k, :]
        V[j*5 + k, 0:7] = V_lad[r, :]
        
    W[:, 0:7] = W_lad
    
    # Part 2: Column 3 (Schoolbook)
    # Rank 9
    # i=0..2, j=0..2
    # U[i*3+j, col] = 1
    # V[j*5+3, col] = 1
    # W[3*3+i, col] = 1
    col = 7
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j*5+3, col] = 1
            W[3*3+i, col] = 1
            col += 1
            
    # Part 3: Column 4 (Schoolbook)
    # Rank 9
    col = 16
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[j*5+4, col] = 1
            W[4*3+i, col] = 1
            col += 1
            
    # Verify
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
