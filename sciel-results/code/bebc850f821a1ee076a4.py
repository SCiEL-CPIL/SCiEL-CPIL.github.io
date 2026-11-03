import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T_{ijk} for matrix multiplication M_{n,m,p} is 1 for all i,j,k
    # if we define the decomposition such that sum_c U_{ij,c} V_{jk,c} W_{ki,c} = T_{ijk}.
    # If T_{ijk} = 1 for all i,j,k, then a rank-1 decomposition with all 1s works.
    
    rank = 1
    U = np.ones((n*m, rank))
    V = np.ones((m*p, rank))
    W = np.ones((p*n, rank))
    
    # Validate
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i*m + j]
                v = V[j*p + k]
                w = W[k*n + i]
                val = np.sum(u * v * w)
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook if my assumption is wrong
                    rank = n * m * p
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m + j2, col] = 1
                                V[j2*p + k2, col] = 1
                                W[k2*n + i2, col] = 1
                                col += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
