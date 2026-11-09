"""Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 30
    
    # Build the schoolbook solution as a fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    for c in range(rank_school):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_school[u_idx, c] = 1.0
        V_school[v_idx, c] = 1.0
        W_school[w_idx, c] = 1.0
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    for target_rank in range(24, rank_school):
        if time.time() > deadline:
            break
            
        found = False
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with structured random matrices
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                if np.random.random() < 0.8:
                    c_school = np.random.randint(0, rank_school)
                    i = c_school // (m * p)
                    rem = c_school % (m * p)
                    j = rem // p
                    k = rem % p
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    weight = np.random.choice([-1.0, 1.0])
                    U[u_idx, c] = weight
                    V[v_idx, c] = weight
                    W[w_idx, c] = weight
                else:
                    for _ in range(2):
                        u_rand = np.random.randint(0, n * m)
                        v_rand = np.random.randint(0, m * p)
                        w_rand = np.random.randint(0, p * n)
                        U[u_rand, c] += np.random.choice([-1.0, 1.0])
                        V[v_rand, c] += np.random.choice([-1.0, 1.0])
                        W[w_rand, c] += np.random.choice([-1.0, 1.0])
            
            # Verify the decomposition
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
