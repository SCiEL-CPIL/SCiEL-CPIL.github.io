import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for t in range(rank):
                    s += U[a][t] * V[b][t] * W[c][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a] * V[b] * W[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, time_limit=100.0):
    """
    Attempt to find a rank-`target_rank` decomposition by starting from 
    schoolbook and randomly merging columns, then optimizing.
    
    Strategy:
    1. Start with schoolbook (rank = n*m*p).
    2. Randomly select pairs of columns to merge.
    3. Optimize the merged decomposition to satisfy the tensor equations.
    4. Verify the result.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    nmp = n * m * p
    
    # We need to reduce from nmp to target_rank, so we merge (nmp - target_rank) pairs
    num_merges = nmp - target_rank
    
    # Try multiple random merge patterns
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Create a random merge pattern
        # We'll create a list of column indices, then randomly merge some
        cols = list(range(nmp))
        
        # Randomly select pairs to merge
        merged_cols = []
        available = list(range(nmp))
        for _ in range(num_merges):
            if len(available) < 2:
                break
            # Pick two random columns to merge
            i = np.random.randint(len(available))
            j = np.random.randint(len(available))
            if i == j:
                continue
            a = available.pop(i)
            # Adjust index after pop
            if j > i:
                j -= 1
            b = available.pop(j)
            merged_cols.append((a, b))
        
        # Build the initial merged decomposition
        # Each merged pair becomes one column with the sum of the original columns' contributions
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # First, fill with schoolbook columns for non-merged ones
        merged_set = set()
        for a, b in merged_cols:
            merged_set.add(a)
            merged_set.add(b)
        
        col_idx = 0
        for c in range(nmp):
            if c in merged_set:
                continue
            # Find i, j, k for this column
            # c = i*m*p + j*p + k
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            U[u_idx, col_idx] = 1
            V[v_idx, col_idx] = 1
            W[w_idx, col_idx] = 1
            col_idx += 1
        
        # Now fill merged columns
        for a, b in merged_cols:
            # Find i, j, k for a and b
            i_a = a // (m * p)
            rem_a = a % (m * p)
            j_a = rem_a // p
            k_a = rem_a % p
            
            i_b = b // (m * p)
            rem_b = b % (m * p)
            j_b = rem_b // p
            k_b = rem_b % p
            
            u_idx_a = i_a * m + j_a
            v_idx_a = j_a * p + k_a
            w_idx_a = k_a * n + i_a
            
            u_idx_b = i_b * m + j_b
            v_idx_b = j_b * p + k_b
            w_idx_b = k_b * n + i_b
            
            # Initialize with the sum (or some combination)
            U[u_idx_a, col_idx] = 1
            V[v_idx_a, col_idx] = 1
            W[w_idx_a, col_idx] = 1
            
            if u_idx_b != u_idx_a:
                U[u_idx_b, col_idx] = 1
            if v_idx_b != v_idx_a:
                V[v_idx_b, col_idx] = 1
            if w_idx_b != w_idx_a:
                W[w_idx_b, col_idx] = 1
            
            col_idx += 1
        
        # Now optimize the decomposition
        # Flatten U, V, W into a single vector
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() > deadline:
                return float('inf')
            Ux = x[:n*m*target_rank].reshape(n*m, target_rank)
            Vx = x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            Wx = x[2*n*m*target_rank:].reshape(p*n, target_rank)
            
            # Compute the tensor entries
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = np.sum(Ux[a] * Vx[b] * Wx[c])
                        error += (s - 1.0) ** 2
            return error
        
        # Try to optimize
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < 1e-6:
                # Verify the solution
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
                
                # Convert to lists
                U_list = U_sol.tolist()
                V_list = V_sol.tolist()
                W_list = W_sol.tolist()
                
                if _verify_numpy(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 48) as the guaranteed fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 2x3x8, the rank is at least 46 (Lader)
    # Try ranks from best_rank-1 down to 46
    for target_rank in range(best_rank - 1, 45, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, time_limit=30.0)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END

BET = {"mechanism": "Attempting rank reduction from 48 to 46 via random column merging and Nelder-Mead optimization, which may find a valid lower-rank decomposition", "q": 0.15, "delta": [0.0, 0.02]}
