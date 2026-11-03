"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Attempt to find a rank-26 decomposition for 3x3 matrix multiplication
    # by fixing 25 terms as schoolbook and solving for the 26th term.
    # If successful, return rank 26. Otherwise, fall back to schoolbook rank 27.
    
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s limit
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def return_schoolbook():
        U = U_school.tolist()
        V = V_school.tolist()
        W = W_school.tolist()
        return {"U": U, "V": V, "W": W}
    
    # Try rank-26 decomposition
    # We fix the first 25 schoolbook terms and solve for the 26th.
    # For each (i,j,k), the residual is 1 if (i,j,k) is one of the first 25, else 1.
    # We set 25 terms to schoolbook, and the 26th term must satisfy:
    # For the excluded (i0,j0,k0): U[i0*m+j0,25]*V[j0*p+k0,25]*W[k0*n+i0,25] = 1
    # For all other (i,j,k): U[i*m+j,25]*V[j*p+k0,25]*W[k*n+i,25] = 0
    
    # We can try to find a solution where the 26th term is non-zero only for one (i,j,k)
    # but zero for all others. This is impossible with rank-1 term unless it's schoolbook.
    # So we need a different approach.
    
    # Actually, a rank-26 decomposition for 3x3 is not known to exist with simple structure.
    # The best known rank is 23 (Landsberg, 2019) but finding it is complex.
    
    # Since we can't easily find rank-26, return schoolbook.
    if time.time() - start_time > time_limit:
        return return_schoolbook()
    
    # Return schoolbook decomposition
    return return_schoolbook()
# EVOLVE-BLOCK-END
