import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: random init + local search on integer/half-integer grid
    
    def residual(U, V, W, rank):
        """Compute max absolute residual of the tensor equation."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    for c in range(rank):
                        s += ui[c] * vj[c] * wk[c]
                    res = abs(s - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res
    
    def try_rank(rank, max_iters=50000, n_restarts=5):
        nonlocal best_result, best_rank
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                return False
            # Random initialization with small values
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(pn)]
            
            # Local search: coordinate descent
            cur_res = residual(U, V, W, rank)
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    return False
                if cur_res < 1e-12:
                    break
                # Pick a random matrix and entry to perturb
                mat_choice = random.random()
                if mat_choice < 0.33:
                    mat, rows = U, nm
                elif mat_choice < 0.66:
                    mat, rows = V, mp
                else:
                    mat, rows = W, pn
                row = random.randrange(rows)
                col = random.randrange(rank)
                old_val = mat[row][col]
                # Try nearby values
                candidates = [old_val + d for d in [-1, -0.5, 0.5, 1] if abs(old_val + d) <= 3]
                # Also try 0
                if 0 not in candidates:
                    candidates.append(0)
                best_new_val = old_val
                best_new_res = cur_res
                for cv in candidates:
                    if cv == old_val:
                        continue
                    mat[row][col] = cv
                    new_res = residual(U, V, W, rank)
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_new_val = cv
                    mat[row][col] = old_val
                if best_new_val != old_val:
                    mat[row][col] = best_new_val
                    cur_res = best_new_res
                else:
                    # Stagnation: try a larger jump
                    if random.random() < 0.1:
                        mat[row][col] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                        new_res = residual(U, V, W, rank)
                        if new_res < cur_res:
                            cur_res = new_res
                        else:
                            mat[row][col] = old_val
            
            if cur_res < 1e-12:
                # Found exact decomposition!
                result = {"U": U, "V": V, "W": W}
                if rank < best_rank:
                    best_result = result
                    best_rank = rank
                return True
        return False
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... actually rank >= max(n, m, p)
    # For (3,4,6), lower bound is at least 6
    # Known: rank(2,2,2)=7, so for (3,4,6) it's likely > 10
    # Let's try from 15 up to 71
    for rank in range(15, 72):
        if time.time() > DEADLINE:
            break
        found = try_rank(rank, max_iters=20000, n_restarts=3)
        if found:
            break
    
    return best_result
