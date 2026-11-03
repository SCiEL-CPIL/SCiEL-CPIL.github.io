import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of matrix multiplication M_{3,4,5}.
    Uses schoolbook algorithm with rank 60.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 110 seconds, leaving 40s for evaluator overhead
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column]
                U[i * m + j, column] = 1
                # V[j*p + k, column]
                V[j * p + k, column] = 1
                # W[k*n + i, column]
                W[k * n + i, column] = 1
                column += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to lists as required by the interface example (though numpy arrays might be accepted, 
    # the prompt says "integer/half-integer entries", usually lists of lists are safer for serialization 
    # if the evaluator expects JSON-like structures, but the seed used lists. 
    # The seed returned lists. I will return lists to match the seed interface exactly.)
    
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
