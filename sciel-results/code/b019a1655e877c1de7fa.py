import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def full_verify(U, V, W):
        """Verify ALL entries of the tensor decomposition."""
        R = U.shape[1]
        # Check on-diagonal: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal: all other (u,v,w) must give 0
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    # Determine if this is an on-diagonal entry
                    i_from_u = u // m
                    j_from_u = u % m
                    j_from_v = v // p
                    k_from_v = v % p
                    k_from_w = w // n
                    i_from_w = w % n
                    if i_from_u == i_from_w and j_from_u == j_from_v and k_from_v == k_from_w:
                        continue  # Already checked
                    val = 0.0
                    for r in range(R):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if abs(val) > 1e-9:
                        return False
        return True
    
    def try_find_rank(R, time_budget=10.0):
        start_time = time.time()
        max_attempts = 5000
        values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        
        # Precompute indices
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((e, u_idx, v_idx, w_idx))
        
        # Precompute column positions for W
        # For each equation e and rank r, the column in C is r*pn + w_idx
        col_positions = []
        for e, u_idx, v_idx, w_idx in indices:
            cols = [r * pn + w_idx for r in range(R)]
            col_positions.append(cols)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with structured sparsity
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Try different sparsity patterns
            sparsity_level = random.choice([3, 4, 5, 6])
            for r in range(R):
                n_u_nonzero = random.randint(2, sparsity_level)
                n_v_nonzero = random.randint(2, sparsity_level)
                
                u_indices = random.sample(range(nm), min(n_u_nonzero, nm))
                v_indices = random.sample(range(mp), min(n_v_nonzero, mp))
                
                for idx in u_indices:
                    U[idx, r] = random.choice(values)
                for idx in v_indices:
                    V[idx, r] = random.choice(values)
            
            # Build coefficient matrix C for W
            C = np.zeros((n * m * p, R * pn))
            
            for idx, (e, u_idx, v_idx, w_idx) in enumerate(indices):
                cols = col_positions[idx]
                for r in range(R):
                    u_val = U[u_idx, r]
                    v_val = V[v_idx, r]
                    if u_val != 0.0 and v_val != 0.0:
                        C[e, cols[r]] = u_val * v_val
            
            b = np.ones(n * m * p)
            
            # Solve linear system
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            # Full verification
            if full_verify(U, V, W):
                # Polish: snap to nearest half-integer
                U_polished = np.rint(U * 2.0) / 2.0
                V_polished = np.rint(V * 2.0) / 2.0
                W_polished = np.rint(W * 2.0) / 2.0
                
                # Re-verify after polishing
                if full_verify(U_polished, V_polished, W_polished):
                    return U_polished.tolist(), V_polished.tolist(), W_polished.tolist()
                else:
                    return U.tolist(), V.tolist(), W.tolist()
                
        return None
    
    def try_structured_rank(R, time_budget=15.0):
        """Try structured patterns based on known algebraic decompositions."""
        start_time = time.time()
        
        # Pattern 1: Block-diagonal structure
        # Pattern 2: Symmetric patterns
        # Pattern 3: Based on Laderman's 7-multiplication algorithm adapted
        
        for attempt in range(2000):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Structured initialization: use block patterns
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Assign rows to blocks
            # For 2x3 * 3x3, we have 6 U-rows, 9 V-rows, 6 W-rows
            # Try to create structure where certain rows share patterns
            
            for r in range(R):
                # Randomly choose a "type" for this rank component
                r_type = random.randint(0, 3)
                
                if r_type == 0:
                    # Type 0: affects first block of i=0
                    for j in range(m):
                        U[j, r] = random.choice([1.0, -1.0, 0.5, -0.5])
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r] = random.choice([1.0, -1.0, 0.5, -0.5])
                elif r_type == 1:
                    # Type 1: affects second block of i=1
                    for j in range(m):
                        U[m + j, r] = random.choice([1.0, -1.0, 0.5, -0.5])
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r] = random.choice([1.0, -1.0, 0.5, -0.5])
                elif r_type == 2:
                    # Type 2: mixed
                    U[random.randint(0, nm-1), r] = random.choice([1.0, -1.0])
                    V[random.randint(0, mp-1), r] = random.choice([1.0, -1.0])
                    if random.random() > 0.5:
                        U[random.randint(0, nm-1), r] = random.choice([0.5, -0.5])
                    if random.random() > 0.5:
                        V[random.randint(0, mp-1), r] = random.choice([0.5, -0.5])
                else:
                    # Type 3: sparse random
                    U[random.randint(0, nm-1), r] = random.choice([1.0, -1.0, 0.5, -0.5])
                    V[random.randint(0, mp-1), r] = random.choice([1.0, -1.0, 0.5, -0.5])
            
            # Build C and solve for W
            indices = []
            col_positions = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i * m * p + j * p + k
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        indices.append((e, u_idx, v_idx, w_idx))
                        col_positions.append([r * pn + w_idx for r in range(R)])
            
            C = np.zeros((n * m * p, R * pn))
            for idx, (e, u_idx, v_idx, w_idx) in enumerate(indices):
                cols = col_positions[idx]
                for r in range(R):
                    u_val = U[u_idx, r]
                    v_val = V[v_idx, r]
                    if u_val != 0.0 and v_val != 0.0:
                        C[e, cols[r]] = u_val * v_val
            
            b = np.ones(n * m * p)
            
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            if full_verify(U, V, W):
                U_polished = np.rint(U * 2.0) / 2.0
                V_polished = np.rint(V * 2.0) / 2.0
                W_polished = np.rint(W * 2.0) / 2.0
                if full_verify(U_polished, V_polished, W_polished):
                    return U_polished.tolist(), V_polished.tolist(), W_polished.tolist()
                else:
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 17 down to 13
    ranks_to_try = [17, 16, 15, 14, 13]
    
    for R in ranks_to_try:
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 10.0:
            break
        
        # Try structured first, then random
        if R == 17:
            time_budget_structured = min(20.0, remaining - 10.0)
            time_budget_random = min(30.0, remaining - 10.0)
        elif R == 16:
            time_budget_structured = min(15.0, remaining - 10.0)
            time_budget_random = min(15.0, remaining - 10.0)
        else:
            time_budget_structured = min(10.0, remaining - 10.0)
            time_budget_random = min(10.0, remaining - 10.0)
        
        # Try structured search
        result = try_structured_rank(R, time_budget=time_budget_structured)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try random search
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining > 10.0:
            result = try_find_rank(R, time_budget=time_budget_random)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
        # Check time
        if time.time() - GLOBAL_START_TIME > TIME_LIMIT - 5.0:
            break
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
