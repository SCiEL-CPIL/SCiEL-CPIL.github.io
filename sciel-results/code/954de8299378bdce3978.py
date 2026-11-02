import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    # Build schoolbook U, V, W
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
    
    # Target tensor condition:
    # sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c] = delta(i',i) * delta(k'',k) * delta(j',j'')
    # We want to find U, V, W of rank r < rank_schoolbook
    # We try r = rank_schoolbook - 1 = 35
    # We use a simple random restart with local search?
    # Or just random sampling of {-1, 0, 1} for a few iterations?
    
    deadline = time.time() + 100  # 100s limit for search
    
    def check_solution(U, V, W, r):
        # Compute the tensor product sum
        # T[i', j', j'', k''] = sum_c W[k*n+i, c] U[i'*m+j', c] V[j''*p+k'', c]
        # We only need to check if it equals the target
        # Target: 1 if i'=i, k''=k, j'=j'' else 0
        # Actually, the equation is for all i, k, i', j', j'', k''
        # Let's just compute the full tensor and compare
        # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # The condition is:
        # sum_c W[k, i, c] U[i', j', c] V[j'', k'', c] = delta(i', i) delta(k'', k) delta(j', j'')
        # Note: W index is k*n+i, so W_r[k, i, c]
        # U index is i*m+j, so U_r[i', j', c]
        # V index is j*p+k, so V_r[j'', k'', c]
        
        # We can compute this using einsum or loops
        # Let's use loops for simplicity and early exit
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

    # Try to find rank 35
    r_target = rank_schoolbook - 1
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random initialization with small integers
        U = np.random.randint(-1, 2, (n * m, r_target))
        V = np.random.randint(-1, 2, (m * p, r_target))
        W = np.random.randint(-1, 2, (p * n, r_target))
        
        # Simple local search: flip a random entry
        for _ in range(1000):
            if time.time() > deadline:
                break
            if check_solution(U, V, W, r_target):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Pick a random matrix and entry
            mat = random.choice([0, 1, 2])
            if mat == 0:
                idx = (random.randint(0, n*m-1), random.randint(0, r_target-1))
                U[idx] = -U[idx] if U[idx] != 0 else 1
            elif mat == 1:
                idx = (random.randint(0, m*p-1), random.randint(0, r_target-1))
                V[idx] = -V[idx] if V[idx] != 0 else 1
            else:
                idx = (random.randint(0, p*n-1), random.randint(0, r_target-1))
                W[idx] = -W[idx] if W[idx] != 0 else 1

    # If not found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
