import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # We will try ranks from 24 to rank_schoolbook - 1
    # Use a heuristic search with a time limit
    
    start_time = time.time()
    time_limit = 100  # Leave some margin for 120s limit
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W, R):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Helper to generate schoolbook decomposition
    def get_schoolbook():
        R = rank_schoolbook
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    # Try to find lower rank
    # We'll use a simple random search with snapping to half-integers
    # Focus on R=24 first, as it's the known minimum
    
    for target_rank in range(24, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V with random small integers
            # Use {-1, 0, 1} for simplicity
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            
            # Solve for W
            # For each w_idx, we have a system of m equations
            # sum_r U[i*m+j, r] V[j*p+k, r] W[w_idx, r] = 1
            # Let A_w be the matrix where row j is U[i*m+j, :] * V[j*p+k, :]
            # We want A_w @ W[w_idx, :] = 1
            # This is a linear system. We can use least squares.
            W = np.zeros((p * n, target_rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Construct system for this w_idx
                    # Variables are W[w_idx, 0..R-1]
                    # Equations for j=0..m-1
                    A_sys = np.zeros((m, target_rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        # Row j of A_sys is element-wise product of U[u_idx] and V[v_idx]
                        A_sys[j, :] = U[u_idx, :] * V[v_idx, :]
                    
                    # Solve A_sys @ W[w_idx] = 1
                    # Use least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Snap to half-integers
                    W[w_idx, :] = np.round(sol * 2) / 2.0
                    
                    # Check if the snapped solution is close to the original
                    # If not, it might not be an exact solution
                    # We'll check the full decomposition later
            if not valid:
                continue
            
            # Check if the full decomposition is valid
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank_schoolbook:
            break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W, R = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
