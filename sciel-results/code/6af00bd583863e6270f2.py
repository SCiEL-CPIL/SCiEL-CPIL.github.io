import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 110.0
    
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
    
    # Attempt to find a rank 22 decomposition using a heuristic search
    # We use small integers and half-integers.
    
    def compute_error(U, V, W):
        """Compute the max absolute error of the decomposition."""
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    err = abs(total - 1)
                    if err > max_err:
                        max_err = err
        return max_err

    def verify(U, V, W):
        return compute_error(U, V, W) == 0

    # Search for Rank 22
    # We try a few random initializations and then do a local search.
    # Local search: pick a random entry in U, V, or W, change it by +/- 0.5 or +/- 1, 
    # if error decreases, keep it.
    
    def local_search(U, V, W, max_steps=500):
        current_err = compute_error(U, V, W)
        if current_err == 0:
            return U, V, W
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, len(U[0])-1)
                old_val = U[row][col]
                # Try small changes
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    U[row][col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    elif new_err == current_err and random.random() < 0.1:
                        # Accept equal error to escape plateaus
                        current_err = new_err
                        break
                    else:
                        U[row][col] = old_val
            elif mat_idx == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, len(V[0])-1)
                old_val = V[row][col]
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    V[row][col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    elif new_err == current_err and random.random() < 0.1:
                        current_err = new_err
                        break
                    else:
                        V[row][col] = old_val
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, len(W[0])-1)
                old_val = W[row][col]
                for delta in [-1, 1, -0.5, 0.5]:
                    new_val = old_val + delta
                    W[row][col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    elif new_err == current_err and random.random() < 0.1:
                        current_err = new_err
                        break
                    else:
                        W[row][col] = old_val
                        
            if current_err == 0:
                return U, V, W
                
        return None if current_err > 0 else (U, V, W)

    # Try Rank 22
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        U = [[random.choice([-1, 0, 1]) for _ in range(22)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(22)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(22)] for _ in range(p*n)]
        
        result = local_search(U, V, W, max_steps=200)
        if result:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Try Rank 23
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        U = [[random.choice([-1, 0, 1]) for _ in range(23)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(23)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(23)] for _ in range(p*n)]
        
        result = local_search(U, V, W, max_steps=200)
        if result:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

    return schoolbook()
