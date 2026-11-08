"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    num_constraints = n * p * m * m  # 126
    
    # Try ranks from lower bound up
    for r in range(21, rank_school):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + r * 1000 + trial)
            # Use half-integer values: multiples of 0.5
            U_rand = 0.5 * np.random.randint(-4, 5, size=(nm, r))
            V_rand = 0.5 * np.random.randint(-4, 5, size=(mp, r))
            
            # Build linear system for W
            A_mat = np.zeros((num_constraints, pn * r))
            b_vec = np.zeros(num_constraints)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            for c in range(r):
                                A_mat[idx, (k*n+i)*r + c] = U_rand[i*m+j, c] * V_rand[jp*p+k, c]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
            
            W_sol, res, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
            W_sol = W_sol.reshape(pn, r)
            
            # Check exactness
            if np.allclose(A_mat @ W_sol.flatten(), b_vec, atol=1e-8):
                best_rank = r
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_sol.copy()
                found = True
                break
        
        if found:
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
