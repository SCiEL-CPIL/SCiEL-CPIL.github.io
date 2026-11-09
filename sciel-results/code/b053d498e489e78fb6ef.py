import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as numpy arrays of zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return current partial result (which is empty if we just started, 
                    # but we are inside the loop so at least some columns are filled).
                    # However, partial result is invalid. 
                    # Given the loop is very fast (63 iterations), this will never trigger.
                    # We return the full schoolbook decomposition which is fast.
                    pass
                    
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Convert to lists to match the expected interface strictly if needed, 
    # though numpy arrays are usually accepted. 
    # The prompt example used lists. Let's provide lists to be safe.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
