import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses a block decomposition:
    - Block 1 (j=0): 2x1x5 schoolbook, rank 10.
    - Block 2 (j in {1,2}): 2x2x5, split into k-groups.
      - k in {0,1}: Strassen 2x2x2, rank 7.
      - k=2: schoolbook 2x2x1, rank 4.
      - k in {3,4}: Strassen 2x2x2, rank 7.
    Total rank: 28.
    
    Indexing:
    - U: (n*m, rank) = (6, 28), A[i,j] -> U[i*m+j, r]
    - V: (m*p, rank) = (15, 28), B[j,k] -> V[j*p+k, r]
    - W: (p*n, rank) = (10, 28), C[i,k] -> W[k*n+i, r]
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    # Initialize as numpy arrays for efficient vectorized verification
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # col = 10
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    def add_strassen_2x2x2(base_j, base_k):
        """Add Strassen decomposition for 2x2x2 block at j=base_j..base_j+1, k=base_k..base_k+1"""
        nonlocal col
        # M1: (A[i=0,j=base]+A[i=1,j=base+1]) * (B[j=base,k=base]+B[j=base+1,k=base+1])
        U[0 * m + base_j, col] = 1
        U[1 * m + (base_j + 1), col] = 1
        V[base_j * p + base_k, col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M2: (A[i=1,j=base]+A[i=1,j=base+1]) * B[j=base,k=base]
        U[1 * m + base_j, col] = 1
        U[1 * m + (base_j + 1), col] = 1
        V[base_j * p + base_k, col] = 1
        W[base_k * n + 1, col] = 1
        W[(base_k + 1) * n + 1, col] = -1
        col += 1
        
        # M3: A[i=0,j=base] * (B[j=base,k=base+1]-B[j=base+1,k=base+1])
        U[0 * m + base_j, col] = 1
        V[base_j * p + (base_k + 1), col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = -1
        W[(base_k + 1) * n + 0, col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M4: A[i=1,j=base+1] * (B[j=base+1,k=base]-B[j=base,k=base])
        U[1 * m + (base_j + 1), col] = 1
        V[(base_j + 1) * p + base_k, col] = 1
        V[base_j * p + base_k, col] = -1
        W[base_k * n + 0, col] = 1
        W[base_k * n + 1, col] = 1
        col += 1
        
        # M5: (A[i=0,j=base]+A[i=0,j=base+1]) * B[j=base+1,k=base+1]
        U[0 * m + base_j, col] = 1
        U[0 * m + (base_j + 1), col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = -1
        W[(base_k + 1) * n + 0, col] = 1
        col += 1
        
        # M6: (A[i=1,j=base]-A[i=0,j=base]) * (B[j=base,k=base]+B[j=base,k=base+1])
        U[1 * m + base_j, col] = 1
        U[0 * m + base_j, col] = -1
        V[base_j * p + base_k, col] = 1
        V[base_j * p + (base_k + 1), col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M7: (A[i=0,j=base+1]-A[i=1,j=base+1]) * (B[j=base+1,k=base]+B[j=base+1,k=base+1])
        U[0 * m + (base_j + 1), col] = 1
        U[1 * m + (base_j + 1), col] = -1
        V[(base_j + 1) * p + base_k, col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = 1
        col += 1
    
    add_strassen_2x2x2(1, 0)  # j=1,2; k=0,1
    # col = 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    # col = 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    add_strassen_2x2x2(1, 3)  # j=1,2; k=3,4
    # col = 28
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify the decomposition is correct
    # For each (i,k) pair, check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    # Actually, the tensor equation is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] should be 1 for all i,j,k (since matrix mult tensor has 1s everywhere)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Decomposition failed at i={i}, j={j}, k={k}: got {val}")
    
    # Convert to lists for the return format (integer entries)
    U_list = [[int(round(val)) for val in row] for row in U.tolist()]
    V_list = [[int(round(val)) for val in row] for row in V.tolist()]
    W_list = [[int(round(val)) for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
