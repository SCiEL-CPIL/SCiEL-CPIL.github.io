"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # Use Winograd F(3) for 3x3x3 which has rank 15.
    # For 3x3x7, split into 3+3+1 blocks.
    # Two 3x3x3 blocks (rank 15 each) + one 3x3x1 block (rank 9) = 39.
    # But we can do better by sharing structure.
    
    # Actually, let's use the known Winograd rank-15 coefficients for 3x3x3.
    # The Winograd F(3) algorithm:
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # C = AB
    
    # The 15 bilinear forms for Winograd F(3):
    # t1 = (a+d+g)(j+m+p)
    # t2 = (a+d-g)(j+m-p)
    # t3 = (a-d+g)(j-m+p)
    # t4 = (a-d-g)(j-m-p)
    # t5 = (b+e+h)(k+n+q)
    # t6 = (b+e-h)(k+n-q)
    # t7 = (b-e+h)(k-n+q)
    # t8 = (b-e-h)(k-n-q)
    # t9 = (c+f+i)(l+o+r)
    # t10 = (c+f-i)(l+o-r)
    # t11 = (c-f+i)(l-o+r)
    # t12 = (c-f-i)(l-o-r)
    # t13 = (a+e+i)(j+n+r)
    # t14 = (b+f+g)(k+o+p)
    # t15 = (c+d+h)(l+m+q)
    
    # Wait, that's not quite right. Let me use the standard Winograd F(3) scheme.
    # Actually, the standard Winograd F(3) has rank 15, but the coefficients are specific.
    
    # Let me just use schoolbook for now since getting the exact Winograd coefficients
    # right is error-prone, and focus on the structure.
    
    # Schoolbook: rank = n*m*p = 63
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
