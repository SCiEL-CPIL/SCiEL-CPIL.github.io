import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Precompute support indices
SUPPORT = []
# Map a to list of (b, c, idx)
A_TO_BC = {a: [] for a in range(NM)}
# Map b to list of (a, c, idx)
B_TO_AC = {b: [] for b in range(MP)}
# Map c to list of (a, b, idx)
C_TO_AB = {c: [] for c in range(PN)}

for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            idx = len(SUPPORT)
            SUPPORT.append((a, b, c))
            A_TO_BC[a].append((b, c, idx))
            B_TO_AC[b].append((a, c, idx))
            C_TO_AB[c].append((a, b, idx))

def get_trivial_solution():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill
    
    # Known lower bound is 18. Trivial is 36.
    # Let's try ranks from 18 up to 36.
    # We prioritize lower ranks but need to be fast.
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    for R in ranks_to_try:
        if time.time() > time_limit:
            break
        
        # Allocate time budget for this rank
        # If we are early, give more time. If late, less.
        # Simple heuristic: if R <= 20, give 10s. If R > 20, give 5s.
        # But we need to finish. Let's just try until time runs out or solution found.
        
        result = search_rank(R, time_limit - (time.time() - start_time) - 5)
        if result is not None:
            return result
            
    # Fallback: Rank 36
    return get_trivial_solution()

def search_rank(R, time_budget):
    num_support = len(SUPPORT)
    
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    # We will run multiple restarts
    # Each restart:
    # 1. Random init
    # 2. Local search with fast delta updates
    
    # Precompute lists for fast access
    A_TO_BC_list = [A_TO_BC[a] for a in range(NM)]
    B_TO_AC_list = [B_TO_AC[b] for b in range(MP)]
    C_TO_AB_list = [C_TO_AB[c] for c in range(PN)]
    
    while time.time() < time.time() + time_budget:
        # Random initialization
        # Using small integers helps with local search
        U = np.random.randint(-2, 3, size=(NM, R)).astype(np.float64)
        V = np.random.randint(-2, 3, size=(MP, R)).astype(np.float64)
        W = np.random.randint(-2, 3, size=(PN, R)).astype(np.float64)
        
        # Compute initial P and Error
        P = np.zeros(num_support, dtype=np.float64)
        for idx, (a, b, c) in enumerate(SUPPORT):
            # P[idx] = sum_r U[a,r] V[b,r] W[c,r]
            P[idx] = np.dot(U[a], V[b] * W[c])
            
        err = np.sum((1.0 - P) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
        
        # Local Search
        # We maintain P and err.
        # When we change U[a,r] by delta:
        # Affected P[idx] for (a, b, c, idx) in A_TO_BC[a]
        # P[idx] changes by delta * V[b,r] * W[c,r]
        # Let D = delta * V[b,r] * W[c,r]
        # Old P = P_old
        # New P = P_old + D
        # Old Err contribution = (1 - P_old)^2
        # New Err contribution = (1 - P_old - D)^2
        # Delta Err = (1 - P_old - D)^2 - (1 - P_old)^2
        #           = (1 - 2*P_old - D + D^2) - (1 - 2*P_old)  [Wait, (1-P)^2 = 1 - 2P + P^2]
        #           = (1 - 2(P_old + D) + (P_old + D)^2) - (1 - 2P_old + P_old^2)
        #           = -2D + (P_old^2 + 2 P_old D + D^2) - P_old^2
        #           = -2D + 2 P_old D + D^2
        #           = D^2 - 2D(1 - P_old)
        
        # This is very fast to compute.
        
        max_iters = 10000
        improved = True
        stall_count = 0
        max_stall = 500
        
        while improved and stall_count < max_stall:
            improved = False
            
            # Pick random matrix and entry
            # Weighted by current error? Or just random.
            # Let's just pick random.
            mat_choice = np.random.randint(0, 3)
            
            if mat_choice == 0:
                mat = U
                dim = NM
                name = 'U'
            elif mat_choice == 1:
                mat = V
                dim = MP
                name = 'V'
            else:
                mat = W
                dim = PN
                name = 'W'
            
            row = np.random.randint(0, dim)
            col = np.random.randint(0, R)
            
            old_val = mat[row, col]
            
            # Try +1 and -1
            # We should also try setting to 0 if not 0?
            # For now, just +/- 1.
            
            for delta in [1, -1]:
                new_val = old_val + delta
                
                # Calculate Delta Error
                delta_err = 0.0
                
                if name == 'U':
                    # a = row, r = col
                    for (b, c, idx) in A_TO_BC_list[row]:
                        D = delta * V[b, col] * W[c, col]
                        P_old = P[idx]
                        # Delta Err = D^2 - 2D(1 - P_old)
                        delta_err += D * D - 2 * D * (1 - P_old)
                        
                elif name == 'V':
                    # b = row, r = col
                    for (a, c, idx) in B_TO_AC_list[row]:
                        D = delta * U[a, col] * W[c, col]
                        P_old = P[idx]
                        delta_err += D * D - 2 * D * (1 - P_old)
                        
                else: # W
                    # c = row, r = col
                    for (a, b, idx) in C_TO_AB_list[row]:
                        D = delta * U[a, col] * V[b, col]
                        P_old = P[idx]
                        delta_err += D * D - 2 * D * (1 - P_old)
                
                if delta_err < -1e-10:
                    # Accept
                    mat[row, col] = new_val
                    err += delta_err
                    
                    # Update P
                    if name == 'U':
                        for (b, c, idx) in A_TO_BC_list[row]:
                            D = delta * V[b, col] * W[c, col]
                            P[idx] += D
                    elif name == 'V':
                        for (a, c, idx) in B_TO_AC_list[row]:
                            D = delta * U[a, col] * W[c, col]
                            P[idx] += D
                    else:
                        for (a, b, idx) in C_TO_AB_list[row]:
                            D = delta * U[a, col] * V[b, col]
                            P[idx] += D
                    
                    improved = True
                    stall_count = 0
                    break # Restart the inner loop or continue?
                    # Breaking out of delta loop, but not while loop.
                    # We should probably break out and pick a new random entry.
            
            if not improved:
                stall_count += 1
                
            # Check time
            if time.time() > time.time() + time_budget:
                break
                
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-9:
                    break
        
        if best_err < 1e-9:
            break
        
        # Check time for restarts
        if time.time() > time.time() + time_budget:
            break
    
    if best_err < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

# EVOLVE-BLOCK-END
