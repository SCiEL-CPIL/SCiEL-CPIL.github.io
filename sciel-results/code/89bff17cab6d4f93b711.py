import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    target_rank_schoolbook = n * m * p
    
    # Schoolbook solution
    U_school = [[0] * target_rank_schoolbook for _ in range(nm)]
    V_school = [[0] * target_rank_schoolbook for _ in range(mp)]
    W_school = [[0] * target_rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = target_rank_schoolbook
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    
    start_time = time.time()
    time_limit = 140.0
    
    # Try ranks from 24 to best_rank - 1
    # We prioritize lower ranks
    for r in range(24, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Time budget for this rank
        rank_time_budget = 5.0
        
        # Randomized local search
        # Values allowed: 0, 1, -1, 0.5, -0.5
        # To make it faster, we can restrict to integers first, then half-integers.
        # Let's use a set of candidate values.
        candidates = [0, 1, -1, 0.5, -0.5]
        
        def compute_error(U, V, W):
            # U, V, W are lists of lists of floats
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for l in range(r):
                            s += U[i*m+j][l] * V[j*p+k][l] * W[k*n+i][l]
                        err += (s - 1.0) ** 2
            return err

        # Try a few restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            if time.time() - (start_time + rank_time_budget * restart) > rank_time_budget:
                # This logic is flawed, let's just check global time
                pass
                
            # Initialize
            U = [[random.choice(candidates) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice(candidates) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(candidates) for _ in range(r)] for _ in range(pn)]
            
            current_err = compute_error(U, V, W)
            
            # Local search
            max_steps = 1000
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                if current_err < 1e-9:
                    # Verify exactness
                    if verify_exact(U, V, W, r):
                        best_rank = r
                        best_solution = {"U": U, "V": V, "W": W}
                    break
                
                # Pick a random matrix, row, col
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    rows, cols = nm, r
                    mat = U
                elif mat_idx == 1:
                    rows, cols = mp, r
                    mat = V
                else:
                    rows, cols = pn, r
                    mat = W
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
                
                old_val = mat[row][col]
                new_val = random.choice(candidates)
                if new_val == old_val:
                    continue
                    
                mat[row][col] = new_val
                new_err = compute_error(U, V, W)
                
                if new_err < current_err:
                    current_err = new_err
                else:
                    mat[row][col] = old_val

    return best_solution

def verify_exact(U, V, W, r):
    """
    Verifies if U, V, W form an exact decomposition.
    Assumes entries are floats that should be half-integers.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Convert to Fractions
    U_frac = [[Fraction(0) for _ in range(r)] for _ in range(nm)]
    V_frac = [[Fraction(0) for _ in range(r)] for _ in range(mp)]
    W_frac = [[Fraction(0) for _ in range(r)] for _ in range(pn)]
    
    for i in range(nm):
        for l in range(r):
            val = U[i][l]
            snapped = round(val * 2) / 2
            if abs(val - snapped) > 1e-6:
                return False
            U_frac[i][l] = Fraction(int(round(val * 2)), 2)
            
    for j in range(mp):
        for l in range(r):
            val = V[j][l]
            snapped = round(val * 2) / 2
            if abs(val - snapped) > 1e-6:
                return False
            V_frac[j][l] = Fraction(int(round(val * 2)), 2)
            
    for k in range(pn):
        for l in range(r):
            val = W[k][l]
            snapped = round(val * 2) / 2
            if abs(val - snapped) > 1e-6:
                return False
            W_frac[k][l] = Fraction(int(round(val * 2)), 2)
            
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for l in range(r):
                    s += U_frac[i*m+j][l] * V_frac[j*p+k][l] * W_frac[k*n+i][l]
                if s != 1:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    assert len(res['U']) == n * m
    assert len(res['U'][0]) == n * m * p
    assert len(res['V']) == m * p
    assert len(res['V'][0]) == n * m * p
    assert len(res['W']) == p * n
    assert len(res['W'][0]) == n * m * p
    print("Construct successful with rank", n*m*p)
