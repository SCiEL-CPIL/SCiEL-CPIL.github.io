import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_over_f2_fast(U, V, n, m, p, r):
    """
    Solve for W over F2 using bitmask Gaussian elimination.
    Returns W if a valid solution exists, else None.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            # Build system: for each j, sum over t of U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
            # Over F2: for each j, dot(U[i*m+j,:], V[j*p+k,:] & W[k*n+i,:]) = 1
            # Let A_j = U[i*m+j,:] & V[j*p+k,:] (bitmask), need A_j · W_row = 1
            
            eqs = []
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                prod = u_row & v_row
                mask = 0
                for t in range(r):
                    if prod[t]:
                        mask |= (1 << t)
                eqs.append([mask, 1])
            
            # Gaussian elimination over F2
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
                        eqs[rr][0] ^= eqs[rank][0]
                        eqs[rr][1] ^= eqs[rank][1]
                
                rank += 1
                if rank == m:
                    break
            
            # Check consistency
            for rr in range(rank, m):
                if eqs[rr][0] == 0 and eqs[rr][1] == 1:
                    return None
            
            # Extract solution (free variables = 0)
            x = [0] * r
            for rr in range(rank):
                col = pivot_cols[rr]
                x[col] = eqs[rr][1]
            
            W[k * n + i, :] = x
    
    # Verify over integers (entries are 0 or 1, so integer check = F2 check)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return None
    
    return W

def _try_rank(r, n, m, p, deadline, random_state, max_attempts=500000):
    """Try to find a valid decomposition at given rank."""
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random sparse patterns with varying density
        # Use different sparsity levels to explore the space
        if attempt % 3 == 0:
            u_density = random_state.randint(2, 3)
            v_density = random_state.randint(2, 3)
        elif attempt % 3 == 1:
            u_density = random_state.randint(3, 4)
            v_density = random_state.randint(3, 4)
        else:
            u_density = random_state.randint(2, 4)
            v_density = random_state.randint(2, 4)
        
        for i in range(n * m):
            num_nz = random_state.randint(max(1, u_density - 1), u_density + 1)
            num_nz = min(num_nz, r)
            if num_nz > 0:
                cols = random_state.sample(range(r), num_nz)
                U[i, :] = 0
                for c in cols:
                    U[i, c] = 1
        
        for i in range(m * p):
            num_nz = random_state.randint(max(1, v_density - 1), v_density + 1)
            num_nz = min(num_nz, r)
            if num_nz > 0:
                cols = random_state.sample(range(r), num_nz)
                V[i, :] = 0
                for c in cols:
                    V[i, c] = 1
        
        W = _solve_w_over_f2_fast(U, V, n, m, p, r)
        if W is not None:
            return (U, V, W)
    
    return None

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
    rng = random.Random(42)
    
    # Try ranks from low to high
    for target_rank in [22, 23, 24, 25]:
        if time.time() > deadline:
            break
        
        result = _try_rank(target_rank, n, m, p, deadline, rng, max_attempts=1000000)
        if result is not None:
            U, V, W = result
            best_result = (target_rank, U, V, W)
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
