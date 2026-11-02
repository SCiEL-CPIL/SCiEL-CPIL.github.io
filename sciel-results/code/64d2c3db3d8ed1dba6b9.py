import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_over_f2_fast(U, V, n, m, p, r):
    """
    Solve for W over F2 using bitmask Gaussian elimination.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            eqs = []
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                prod = u_row & v_row
                mask = 0
                for t in range(r):
                    if prod[t]:
                        mask |= (1 << t)
                eqs.append((mask, 1))
            
            rank = 0
            pivot_cols = []
            for col in range(r):
                pivot = -1
                for rr in range(rank, m):
                    if (eqs[rr][0] >> col) & 1:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                
                eqs[rank], eqs[pivot] = eqs[pivot], eqs[rank]
                pivot_cols.append(col)
                
                for rr in range(m):
                    if rr != rank and ((eqs[rr][0] >> col) & 1):
                        eqs[rr] = (eqs[rr][0] ^ eqs[rank][0], eqs[rr][1] ^ eqs[rank][1])
                
                rank += 1
                if rank == m:
                    break
            
            for rr in range(rank, m):
                if eqs[rr][0] == 0 and eqs[rr][1] == 1:
                    return None
            
            x = [0] * r
            for rr in range(rank):
                col = pivot_cols[rr]
                x[col] = eqs[rr][1]
            
            W[k * n + i, :] = x
    
    # Verify over integers
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return None
    
    return W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    random.seed(42)
    
    # Try ranks from 24 up to 35
    # Rank 23 is known to be impossible for 3x3x4 (likely)
    # Rank 24 is the next candidate
    
    min_rank_to_try = 24
    max_rank_to_try = 35
    
    for target_rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank
        # If we have lots of time, try more attempts
        # Simple heuristic: try a fixed number of attempts per rank, or until time is up
        # Since lower ranks are harder, maybe give them more time?
        # But if 24 is impossible, we waste time.
        # Let's try a moderate number of attempts.
        
        max_attempts = 50000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            
            # Random sparse patterns
            # Tune sparsity
            for i in range(n * m):
                num_nonzero = random.randint(2, 4)
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    U[i, c] = 1
            
            for i in range(m * p):
                num_nonzero = random.randint(2, 4)
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    V[i, c] = 1
            
            W = _solve_w_over_f2_fast(U, V, n, m, p, target_rank)
            if W is not None:
                best_result = (target_rank, U, V, W)
                # If we found a solution, we can return immediately if it's low rank
                # Or we can break to try even lower? No, we are iterating upwards.
                # So this is the minimal rank found so far.
                # We can break out of the while loop and the for loop.
                # Since we are going 24, 25, ..., if we find 24, it's the best possible (assuming 23 is impossible).
                # If we find 25, we might still find 24? No, we already tried 24.
                # So we can return.
                return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
        
        # If we exhausted attempts for this rank without success, try next rank
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
