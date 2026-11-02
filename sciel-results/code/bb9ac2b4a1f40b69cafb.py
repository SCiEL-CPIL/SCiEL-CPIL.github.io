import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36
    
    def check_solution(U, V, W, r):
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jpp in range(m):
                            for kpp in range(p):
                                target = 1 if (ip == i and kpp == k and jp == jpp) else 0
                                val = 0
                                for c in range(r):
                                    val += W_r[k, i, c] * U_r[ip, jp, c] * V_r[jpp, kpp, c]
                                if val != target:
                                    return False
        return True

    deadline = time.time() + 110
    
    # Try to find a solution with r terms
    # We will try r=30, 31, 32, 33, 34, 35
    for r in range(30, 36):
        r_sub = r - 9 # Terms for k=0,1,2
        if r_sub < 0:
            continue
            
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            W = np.zeros((p * n, r), dtype=int)
            
            # Fix last 9 terms for k=3 (4th column)
            # C[i, 3] = sum_j A[i, j] B[j, 3]
            # Terms c = r-9 to r-1
            # Map (i, j) to term index
            idx = 0
            for i in range(n):
                for j in range(m):
                    c = r - 9 + idx
                    U[i * m + j, c] = 1
                    V[j * p + 3, c] = 1
                    W[3 * n + i, c] = 1
                    idx += 1
            
            # Initialize first r_sub terms randomly
            if r_sub > 0:
                U[:, :r_sub] = np.random.randint(-1, 2, (n * m, r_sub))
                V[:, :r_sub] = np.random.randint(-1, 2, (m * p, r_sub))
                W[:, :r_sub] = np.random.randint(-1, 2, (p * n, r_sub))
                
                # Ensure V and W are zero for k=3 in the first r_sub terms?
                # To keep them orthogonal to the fixed terms, we should ensure
                # that for c < r_sub, V[j, 3, c] = 0 and W[3, i, c] = 0.
                # This allows the first r_sub terms to only affect k=0,1,2.
                V[:, r_sub-1 if False else 3, :r_sub] = 0 # V[j, 3, c]
                W[3 * n : 3 * n + n, :r_sub] = 0 # W[3, i, c]
                
                # Also U can be non-zero, but if V and W are 0 for k=3, 
                # the term contributes 0 to k=3. So it's safe.
                
            # Local search on the first r_sub terms
            for step in range(10000):
                if time.time() > deadline:
                    break
                
                if check_solution(U, V, W, r):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Only modify the first r_sub terms
                if r_sub == 0:
                    break
                    
                mat = random.choice([0, 1, 2])
                c = random.randint(0, r_sub - 1)
                
                if mat == 0:
                    i = random.randint(0, n*m-1)
                    if random.random() < 0.5:
                        U[i, c] = -U[i, c] if U[i, c] != 0 else 1
                    else:
                        U[i, c] = 0
                elif mat == 1:
                    # V[j, k, c]. k in 0..2
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-2) # 0..2
                    idx_v = j * p + k
                    if random.random() < 0.5:
                        V[idx_v, c] = -V[idx_v, c] if V[idx_v, c] != 0 else 1
                    else:
                        V[idx_v, c] = 0
                else:
                    # W[k, i, c]. k in 0..2
                    k = random.randint(0, p-2)
                    i = random.randint(0, n-1)
                    idx_w = k * n + i
                    if random.random() < 0.5:
                        W[idx_w, c] = -W[idx_w, c] if W[idx_w, c] != 0 else 1
                    else:
                        W[idx_w, c] = 0

    # Fallback: Schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
