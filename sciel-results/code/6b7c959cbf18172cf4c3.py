"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _error(params, rank, n=3, m=3, p=5):
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    U = params[:num_u].reshape(n*m, rank)
    V = params[num_u:num_u+num_v].reshape(m*p, rank)
    W = params[num_u+num_v:].reshape(p*n, rank)
    
    err = 0.0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U[i*m+l], V[l*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    for i in range(n):
        for l in range(m):
            for lp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i==ip and l==lp and k==kp: continue
                            s = np.dot(U[i*m+l], V[lp*p+k] * W[kp*n+ip])
                            err += s*s
    return err

def construct():
    n, m, p = DIMENSIONS
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
    return {"U":U,"V":V,"W":W}
# EVOLVE-BLOCK-END
