import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses vectorized error calculation and simulated annealing/local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    rows_u = n * m  # 9
    rows_v = m * p  # 12
    rows_w = p * n  # 12
    
    # Target tensor: 1 at active triples, 0 elsewhere
    # Active triple (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((rows_u, rows_v, rows_w), dtype=np.int64)
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
                active_triples.append((a, b, c))
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(rows_u)]
        V = [[0] * rank for _ in range(rows_v)]
        W = [[0] * rank for _ in range(rows_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def calc_error(U, V, W, r):
        """Calculate the number of mismatches between computed tensor and target."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        S = np.zeros((rows_u, rows_v, rows_w), dtype=np.int64)
        for k in range(r):
            # Broadcasting: U[:, k] is (9,), V[:, k] is (12,), W[:, k] is (12,)
            S += U_np[:, k][:, None, None] * V_np[:, k][None, :, None] * W_np[:, k][None, None, :]
        
        return int(np.sum(S != target))
    
    def local_search(r, max_steps=5000):
        """
        Simulated annealing to find a rank-r decomposition.
        Returns (U, V, W) if found, else None.
        """
        # Initialize with random sparse matrices
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(rows_u)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(rows_v)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(rows_w)]
        
        current_error = calc_error(U, V, W, r)
        if current_error == 0:
            return U, V, W
            
        T = 1.0
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            
            # Choose a random matrix and entry to flip
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, rows_u - 1)
                col = random.randint(0, r - 1)
                old_val = U[row][col]
                # Try a new value from {-1, 0, 1} different from old
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                U[row][col] = new_val
            elif mat_idx == 1:
                row = random.randint(0, rows_v - 1)
                col = random.randint(0, r - 1)
                old_val = V[row][col]
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                V[row][col] = new_val
            else:
                row = random.randint(0, rows_w - 1)
                col = random.randint(0, r - 1)
                old_val = W[row][col]
                new_val = random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = random.choice([-1, 0, 1])
                W[row][col] = new_val
            
            new_error = calc_error(U, V, W, r)
            
            if new_error == 0:
                return U, V, W
            
            delta = new_error - current_error
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_error = new_error
            else:
                # Revert
                if mat_idx == 0:
                    U[row][col] = old_val
                elif mat_idx == 1:
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
            
            T *= cooling
            if T < 0.01:
                # Reset temperature if stuck? Or just break and restart
                # For now, let it run, but maybe restart if no improvement for long
                pass
        
        return None
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Search for lower ranks
    # Start from a reasonable lower bound. Rank 24 is a good guess for 3x3x4.
    for r in range(24, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for each rank
        found = False
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
            result = local_search(r, max_steps=2000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                found = True
                break
        
        if found:
            # If we found one, we might try to go lower? 
            # The loop goes up, so we found the first one starting from 24.
            # If 24 works, we return 24.
            break
            
    return best_result
