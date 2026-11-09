import numpy as np
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
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
    
    best_result = schoolbook()
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def is_half_int(x, tol=1e-9):
        x2 = 2 * x
        return abs(x2 - round(x2)) < tol
    
    def try_rank(r, num_trials):
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random U, V with entries from {-1, 0, 1}
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            for i in range(n * m):
                for c in range(r):
                    U[i, c] = random.choice([-1, 0, 1])
            for i in range(m * p):
                for c in range(r):
                    V[i, c] = random.choice([-1, 0, 1])
            
            W = np.zeros((p * n, r), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M: m x r
                    M = np.zeros((m, r), dtype=float)
                    for j in range(m):
                        for c in range(r):
                            M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Find a half-integer solution w such that M @ w = ones(m)
                    # Try: pick a subset of m columns, invert, check half-integer
                    found = False
                    
                    # First try: if r == m, just invert
                    if r == m:
                        try:
                            w = np.linalg.solve(M, np.ones(m))
                            if all(is_half_int(w[c]) for c in range(r)):
                                W[k * n + i, :] = w
                                found = True
                        except:
                            pass
                    
                    if not found:
                        # Try random subsets of size m
                        cols = list(range(r))
                        max_subsets = min(200, len(list(itertools.combinations(range(r), m))))
                        random.shuffle(cols)
                        for _ in range(max_subsets):
                            sub = random.sample(range(r), m)
                            Ms = M[:, sub]
                            det = np.linalg.det(Ms)
                            if abs(det) < 1e-10:
                                continue
                            try:
                                w_sub = np.linalg.solve(Ms, np.ones(m))
                                if all(is_half_int(w_sub[c]) for c in range(m)):
                                    # Set W for these columns
                                    for idx, col in enumerate(sub):
                                        W[k * n + i, col] = w_sub[idx]
                                    found = True
                                    break
                            except:
                                continue
                    
                    if not found:
                        # Try least squares as fallback
                        w, res, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        if all(is_half_int(w[c]) for c in range(r)):
                            W[k * n + i, :] = w
                            found = True
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Convert to half-integer lists
            U_list = [[int(round(2 * U[i, c])) / 2 for c in range(r)] for i in range(n * m)]
            V_list = [[int(round(2 * V[i, c])) / 2 for c in range(r)] for i in range(m * p)]
            W_list = [[int(round(2 * W[i, c])) / 2 for c in range(r)] for i in range(p * n)]
            
            if verify(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for r in range(12, 36):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        trials = min(3000, max(50, int(remaining * 30)))
        result = try_rank(r, trials)
        if result is not None:
            best_result = result
            break
    
    return best_result
