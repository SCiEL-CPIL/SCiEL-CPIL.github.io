"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try to find a low-rank decomposition
    # Use the fact that we can fix U and V, then solve for W
    
    best_result = None
    
    for rank in range(28, 15, -1):
        for attempt in range(50):
            random.seed(attempt * 1000 + rank * 7)
            np.random.seed(attempt * 1000 + rank * 7)
            
            # Random integer initialization in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, rank))
            V = np.random.randint(-1, 2, size=(mp, rank))
            
            # Compute S[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            S = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        S[i,k,r] = s
            
            # Solve for W
            W = np.zeros((pn, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    s_row = S[i,k,:]
                    norm_sq = np.dot(s_row, s_row)
                    if abs(norm_sq) < 1e-12:
                        valid = False
                        break
                    # W[k*n+i, :] = s_row / norm_sq
                    # Check if this gives integer/half-integer
                    w_candidate = s_row / norm_sq
                    # For now, just use it and check later
                    W[k*n+i, :] = w_candidate
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for r in range(rank):
                            total += W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
                        err = abs(total - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                # Check if entries are close to integer/half-integer
                all_half_int = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        if abs(val * 2 - round(val * 2)) > 1e-6:
                            all_half_int = False
                            break
                    if not all_half_int:
                        break
                
                if all_half_int:
                    # Quantize to half-integers
                    U_q = np.round(U * 2) / 2
                    V_q = np.round(V * 2) / 2
                    W_q = np.round(W * 2) / 2
                    
                    # Re-verify after quantization
                    max_err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for r in range(rank):
                                    total += W_q[k*n+i, r] * U_q[i*m+j, r] * V_q[j*p+k, r]
                                err = abs(total - 1.0)
                                if err > max_err2:
                                    max_err2 = err
                    
                    if max_err2 < 1e-8:
                        return {
                            "U": U_q.tolist(),
                            "V": V_q.tolist(),
                            "W": W_q.tolist()
                        }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
