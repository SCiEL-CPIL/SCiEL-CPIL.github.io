import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 120

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    This implementation constructs Strassen's algorithm decomposition programmatically.
    Strassen's algorithm achieves rank 7, which is the known minimal rank for 2x2 matrix
    multiplication over fields of characteristic != 2.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    U has shape (n*m, rank) = (4, 7)
    V has shape (m*p, rank) = (4, 7)  
    W has shape (p*n, rank) = (4, 7)
    
    The function returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        raise TimeoutError("Time limit exceeded")
        
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices as lists of lists of zeros
    # U: rows correspond to A[i, j] with index i*m + j
    # V: rows correspond to B[j, k] with index j*p + k
    # W: rows correspond to C[i, k] with index k*n + i (as per task description W[k*n+i, r])
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define Strassen's 7 products
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # Indices for A (U rows): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (V rows): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Product 1: (a + d) * (e + h)
    # U contributions: a (idx 0), d (idx 3)
    # V contributions: e (idx 0), h (idx 3)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    
    # Product 2: (c + d) * e
    # U contributions: c (idx 2), d (idx 3)
    # V contributions: e (idx 0)
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    
    # Product 3: a * (f - h)
    # U contributions: a (idx 0)
    # V contributions: f (idx 1), -h (idx 3)
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    
    # Product 4: d * (g - e)
    # U contributions: d (idx 3)
    # V contributions: g (idx 2), -e (idx 0)
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    
    # Product 5: (a + b) * h
    # U contributions: a (idx 0), b (idx 1)
    # V contributions: h (idx 3)
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    
    # Product 6: (c - a) * (e + f)
    # U contributions: c (idx 2), -a (idx 0)
    # V contributions: e (idx 0), f (idx 1)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    
    # Product 7: (b - d) * (g + h)
    # U contributions: b (idx 1), -d (idx 3)
    # V contributions: g (idx 2), h (idx 3)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    
    # Now, determine the contributions to the output matrix C
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W row indices: k*n + i
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]  (Wait, let's re-verify the convention)
    # Task says: W[k*n+i, r].
    # Usually C[i, k] is the element at row i, col k.
    # If index is k*n + i:
    # idx 0: k=0, i=0 -> C[0,0]
    # idx 1: k=0, i=1 -> C[1,0]
    # idx 2: k=1, i=0 -> C[0,1]
    # idx 3: k=1, i=1 -> C[1,1]
    
    # Let's map the Strassen results to these indices.
    
    # C[0,0] (idx 0) = M1 + M4 - M5 + M7
    W[0][0] += 1  # M1
    W[0][3] += 1  # M4
    W[0][4] -= 1  # M5
    W[0][6] += 1  # M7
    
    # C[1,0] (idx 1) = M2 + M4
    W[1][1] += 1  # M2
    W[1][3] += 1  # M4
    
    # C[0,1] (idx 2) = M3 + M5
    W[2][2] += 1  # M3
    W[2][4] += 1  # M5
    
    # C[1,1] (idx 3) = M1 - M2 + M3 + M6
    W[3][0] += 1  # M1
    W[3][1] -= 1  # M2
    W[3][2] += 1  # M3
    W[3][5] += 1  # M6

    # Convert to integer types if they are integers to ensure "integer/half-integer" requirement is met cleanly
    # The entries are all -1, 0, or 1.
    U = [[int(val) for val in row] for row in U]
    V = [[int(val) for val in row] for row in V]
    W = [[int(val) for val in row] for row in W]

    return {"U": U, "V": V, "W": W}
