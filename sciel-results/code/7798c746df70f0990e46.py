import time
import math
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 115

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try ranks 24, 25, 26
    # We know rank(3x3x4) <= rank(3x3x3) + 1
    # If rank(3x3x3) is 23, then 24 is possible.
    # If rank(3x3x3) is 24, then 25 is possible.
    
    for target_rank in [24, 25, 26]:
        if _time_remaining() < 10:
            break
        if best_result and len(best_result["U"][0]) <= target_rank:
            continue
            
        max_attempts = 100
        for attempt in range(max_attempts):
            if _time_remaining() < 2:
                break
            try:
                result = _try_rank_search(n, m, p, target_rank, attempt * 12345 + 6789)
                if result is not None:
                    best_result = result
                    break
            except Exception:
                continue
                
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_rank_search(n, m, p, target_rank, seed):
    np.random.seed(seed)
    random.seed(seed)
    r = target_rank
    U = np.random.choice([-1, 1], size=(n*m, r))
    V = np.random.choice([-1, 1], size=(m*p, r))
    
    def get_badness(U, V):
        badness = 0
        M = np.zeros((n*p, r))
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U[i*m+j, c] * V[j*p+k, c]
                    M[i*p+k, c] = s
                # Check if row M[i*p+k] has a half-integer solution
                # M_row dot W_row = 1
                # W_row must be half-integers.
                # This is a linear Diophantine equation.
                # For simplicity, check if min |M| is small and solvable.
                # If we assume W is sparse (1 non-zero), then M[c] * W[c] = 1 => W[c] = 1/M[c].
                # W[c] must be half-integer => 1/M[c] = k/2 => M[c] = 2/k.
                # So M[c] must be 2, 1, 0.5, etc.
                # Since M is integer, M[c] must be 1 or 2 (or -1, -2).
                # If M[c] == 1, W[c] = 1.
                # If M[c] == 2, W[c] = 0.5.
                solvable = False
                for c in range(r):
                    if M[i*p+k, c] == 1:
                        solvable = True
                        break
                    if M[i*p+k, c] == 2:
                        solvable = True
                        break
                    if M[i*p+k, c] == -1:
                        solvable = True
                        break
                    if M[i*p+k, c] == -2:
                        solvable = True
                        break
                if not solvable:
                    badness += 1
        return badness, M
    
    badness, M = get_badness(U, V)
    
    max_iters = 1000
    for it in range(max_iters):
        if badness == 0:
            break
        if _time_remaining() < 2:
            break
            
        flip_U = random.random() < 0.5
        if flip_U:
            idx = random.randint(n*m)
            col = random.randint(r)
            U[idx, col] *= -1
        else:
            idx = random.randint(m*p)
            col = random.randint(r)
            V[idx, col] *= -1
            
        new_badness, new_M = get_badness(U, V)
        if new_badness < badness:
            badness = new_badness
            M = new_M
        else:
            if flip_U:
                U[idx, col] *= -1
            else:
                V[idx, col] *= -1
                
    if badness == 0:
        W = np.zeros((p*n, r))
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    if M[i*p+k, c] == 1:
                        W[k*n+i, c] = 1.0
                        break
                    elif M[i*p+k, c] == 2:
                        W[k*n+i, c] = 0.5
                        break
                    elif M[i*p+k, c] == -1:
                        W[k*n+i, c] = -1.0
                        break
                    elif M[i*p+k, c] == -2:
                        W[k*n+i, c] = -0.5
                        break
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = []
        for row in W:
            W_list.append([round(x*2)/2.0 for x in row])
            
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
