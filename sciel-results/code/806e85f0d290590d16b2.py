import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145  # seconds
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize factor matrices as numpy arrays for easier manipulation
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                # Index in V: row j*p + k
                # Index in W: row k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    # If we broke early due to time, we return a partial/invalid result?
    # The problem requires exact decomposition. If we don't finish, we should return the full one if possible.
    # But the loop is very fast (96 iterations), so this check is just formal.
    
    # Convert to lists of lists as per the interface requirement: 
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    # The example used lists of lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are integers (they are 0.0 or 1.0)
    # Converting to int is safe.
    U_list = [[int(x) for x in row] for row in U_list]
    V_list = [[int(x) for x in row] for row in V_list]
    W_list = [[int(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
