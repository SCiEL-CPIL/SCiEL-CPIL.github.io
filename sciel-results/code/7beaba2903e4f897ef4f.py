"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank matching
    # Heavy flows to short distances
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Ensure p is a valid permutation (it is by construction)
    # Calculate initial cost
    def calc_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Using einsum or matrix multiplication
        # b_perm = b[np.ix_(p, p)]
        # cost = np.sum(a * b_perm)
        # Faster:
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for faster delta?
    # We will use the loop method.
    
    tabu_size = 30
    tabu_list = {} # (i, j) -> expiry step
    current_step = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    # Main Loop
    while time.time() < deadline:
        # Check for restart
        if no_improve_count > max_no_improve:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_list.clear()
            continue
            
        # Find best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Vectorized-ish approach for speed?
        # For each i, compute deltas with all j?
        # Let's stick to simple loop first, n=150 is small.
        
        # Optimization: Precompute b[:, p] once per step?
        # b[:, p] is n x n.
        # Let's create Bp = b[:, p]
        Bp = b[:, p]
        
        for i in range(n):
            u = p[i]
            # We can skip if i is tabu with any j? Tabu is symmetric.
            # Check tabu for (min(i,j), max(i,j))
            
            # Compute diff_A_i = a[i, :]
            ai = a[i]
            
            for j in range(i + 1, n):
                # Check tabu
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                v = p[j]
                # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                # Note: b[v, p[k]] is Bp[v, k]
                # b[u, p[k]] is Bp[u, k]
                
                # Vectorized dot product
                # diff_A = ai - a[j]
                # diff_B = Bp[v] - Bp[u]
                
                # We can compute this as:
                # (ai - a[j]) dot (Bp[v] - Bp[u])
                # = ai dot Bp[v] - ai dot Bp[u] - a[j] dot Bp[v] + a[j] dot Bp[u]
                
                # Let's compute dots
                # ai_dot_Bp_v = np.dot(ai, Bp[v])
                # ai_dot_Bp_u = np.dot(ai, Bp[u])
                # aj_dot_Bp_v = np.dot(a[j], Bp[v])
                # aj_dot_Bp_u = np.dot(a[j], Bp[u])
                
                # This is 4 dot products per pair.
                # 11000 pairs * 4 dots = 44000 dots.
                # Each dot is length 150.
                # Might be slow in Python loop.
                
                # Alternative:
                # diff_A = ai - a[j]
                # diff_B = Bp[v] - Bp[u]
                # delta = np.dot(diff_A, diff_B)
                # 1 dot product per pair.
                
                delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                
                # Aspiration: if delta < 0 and (i,j) is tabu, allow if cost improves best
                # Actually, we just check if it's better than current best step delta
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        # If no valid move found (all tabu), clear tabu list?
        if best_i == -1:
            tabu_list.clear()
            continue
            
        # Perform swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu
        tabu_list[(best_i, best_j)] = current_step + tabu_size
        tabu_list[(best_j, best_i)] = current_step + tabu_size
        
        current_step += 1
        
        # Periodic cleanup of tabu list
        if current_step % 100 == 0:
            # Remove old entries
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
