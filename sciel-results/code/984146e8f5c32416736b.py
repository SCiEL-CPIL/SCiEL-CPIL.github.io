"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 140.0

    # Search for lower rank
    # Known bounds suggest 43 is possible.
    for R in range(44, 39, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts
        for restart in range(500):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            
            # Generate sparse random integer matrices U and V
            # U: (n*m, R), V: (m*p, R)
            # Use small integers to keep numbers small
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            
            # Randomly set some entries to +/- 1
            # Density around 0.2 - 0.3
            for idx in range(n * m):
                for r in range(R):
                    if np.random.rand() < 0.2:
                        U[idx, r] = np.random.choice([-1, 1])
                        
            for idx in range(m * p):
                for r in range(R):
                    if np.random.rand() < 0.2:
                        V[idx, r] = np.random.choice([-1, 1])

            # Check if this U, V allows a valid W
            valid = True
            W_mat = np.zeros((p * n, R), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    # Construct M_ik: (m, R)
                    # M_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M_ik = np.zeros((m, R))
                    for j in range(m):
                        M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M_ik @ W_ik = ones(m)
                    # Use least squares
                    try:
                        W_ik, residuals, rank, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if W_ik entries are close to half-integers
                    W_ik_check = np.round(W_ik * 2) / 2
                    if not np.allclose(W_ik, W_ik_check, atol=1e-6):
                        valid = False
                        break
                        
                    W_mat[k * n + i, :] = W_ik_check
            
            if valid:
                # Final verification
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W_mat[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-6:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_mat.tolist()
                    }

    return schoolbook()

# EVOLVE-BLOCK-END
