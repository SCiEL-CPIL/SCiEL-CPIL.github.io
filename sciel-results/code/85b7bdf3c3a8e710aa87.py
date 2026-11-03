"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[d, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Try to find better rank
    for r in range(12, best_rank):
        if time.time() > deadline:
            break
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            
            # UV[a,b,c] = U[a,c]*V[b,c]
            UV = np.einsum('ac,bc->abc', U, V)
            
            W_sol = np.zeros((p*n, r))
            max_err = 0.0
            
            for d in range(p*n):
                B_d = UV.reshape(n*m * m*p, r)
                rhs = T[:, :, d].flatten()
                W_d, _, _, _ = np.linalg.lstsq(B_d, rhs, rcond=None)
                W_sol[d, :] = W_d
                pred = B_d @ W_d
                max_err = max(max_err, np.max(np.abs(pred - rhs)))
            
            if max_err < 1e-10:
                # Quick verify
                err = 0.0
                for a in range(n*m):
                    for b in range(m*p):
                        for d in range(p*n):
                            pred = np.dot(UV[a, b, :], W_sol[d, :])
                            err = max(err, abs(pred - T[a, b, d]))
                if err < 1e-10:
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W_sol
                    break
        
        if best_rank == r:
            break
    
    def to_list(mat):
        mat2 = mat * 2
        if np.allclose(mat2, np.round(mat2), atol=1e-8):
            return [[int(round(x)) for x in row] for row in mat2]
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
