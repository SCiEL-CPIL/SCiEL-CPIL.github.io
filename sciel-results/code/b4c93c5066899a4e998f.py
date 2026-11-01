import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    # Indices for U: (n*m, rank) -> (4, rank)
    # Indices for V: (m*p, rank) -> (8, rank)
    # Indices for W: (p*n, rank) -> (8, rank)
    
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def residual_sq(U, V, W):
        rank = U.shape[1]
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    res += (val - 1.0)**2
        return res

    def try_rank13():
        rank = 13
        # Initialize with random integers {-1, 0, 1}
        # To help convergence, we can start with a known rank-14 solution 
        # and try to reduce it, but that's complex. 
        # Random restarts with local search is more robust.
        
        best_val = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randint(-1, 2, (4, rank))
            V = np.random.randint(-1, 2, (8, rank))
            W = np.random.randint(-1, 2, (8, rank))
            
            # Local search
            current_val = residual_sq(U, V, W)
            
            # If already perfect, return
            if current_val < 1e-9:
                if verify(U, V, W):
                    return U, V, W
            
            # Perform local search steps
            steps = 100
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random matrix and entry to flip
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    row = np.random.randint(0, 4)
                    col = np.random.randint(0, rank)
                    old_val = U[row, col]
                    new_val = -old_val if old_val != 0 else 1
                    U[row, col] = new_val
                    new_val_res = residual_sq(U, V, W)
                    if new_val_res < current_val:
                        current_val = new_val_res
                    else:
                        U[row, col] = old_val
                elif mat_idx == 1:
                    row = np.random.randint(0, 8)
                    col = np.random.randint(0, rank)
                    old_val = V[row, col]
                    new_val = -old_val if old_val != 0 else 1
                    V[row, col] = new_val
                    new_val_res = residual_sq(U, V, W)
                    if new_val_res < current_val:
                        current_val = new_val_res
                    else:
                        V[row, col] = old_val
                else:
                    row = np.random.randint(0, 8)
                    col = np.random.randint(0, rank)
                    old_val = W[row, col]
                    new_val = -old_val if old_val != 0 else 1
                    W[row, col] = new_val
                    new_val_res = residual_sq(U, V, W)
                    if new_val_res < current_val:
                        current_val = new_val_res
                    else:
                        W[row, col] = old_val
            
            if current_val < best_val:
                best_val = current_val
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_val < 1e-9:
                if verify(best_U, best_V, best_W):
                    return best_U, best_V, best_W

        return None

    # Try rank 13
    result = try_rank13()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 14 (known valid solution)
    rank = 14
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
            
    # Strassen-based rank 14
    # Left block (k=0,1)
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Right block (k=2,3)
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}
