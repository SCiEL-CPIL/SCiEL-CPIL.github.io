import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Uses Schoolbook multiplication with rank 72.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
                
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U_list]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V_list]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Constructed rank {len(result['U'][0])}")
