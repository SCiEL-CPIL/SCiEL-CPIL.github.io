import numpy as np
import time

def solve(a, b):
    n = len(a)
    # 115 seconds safety margin
    deadline = time.time() + 115
    
    # Initial Solution: Rank Matching
    # Heavy flows (high sum in A) to short distances (low sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    heavy = np.argsort(-a_sums, kind='stable')
    # Sort locations by increasing distance sum
    central = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Function to compute cost
    def get_cost(p_curr):
        # Cost = sum a[i, j] * b[p[i], p[j]]
        # = np.sum(a * b[np.ix_(p_curr, p_curr)])
        return np.sum(a * b[np.ix_(p_curr, p_curr)])
    
    current_cost = get_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu Search with Vectorized Best Move Selection
    
    # Q[i, k] = sum_j a[i, j] * b[k, p[j]]
    # Q = a @ b[:, p]
    Q = a @ b[:, p]
    
    # C[i] = Q[i, p[i]]
    C = Q[np.arange(n), p]
    
    # Tabu List
    # We store (u, v, expiry_iteration)
    tabu_list = []
    tabu_tenure = 40
    current_iter = 0
    
    # Mask for tabu moves
    tabu_mask = np.zeros((n, n), dtype=bool)
    
    # Stagnation handling
    stagnation_limit = 200
    last_improvement_iter = 0
    
    while time.time() < deadline:
        current_iter += 1
        
        # Update Tabu Mask
        # Remove expired
        while tabu_list and tabu_list[0][2] <= current_iter:
            u, v, _ = tabu_list.pop(0)
            tabu_mask[u, v] = False
            tabu_mask[v, u] = False
            
        # Compute Delta for all pairs
        # D[u, v] = Q[u, p[v]] + Q[v, p[u]]
        Q_p = Q[:, p]
        D = Q_p + Q_p.T
        
        # Delta[u, v] = D[u, v] - C[u] - C[v]
        Delta = D - C[:, None] - C[None, :]
        
        # Apply Tabu Mask
        # We want to minimize Delta, so set tabu moves to inf
        Delta[tabu_mask] = np.inf
        # Diagonal is inf (no self-swap)
        np.fill_diagonal(Delta, np.inf)
        
        # Find best move
        min_val = np.min(Delta)
        
        if min_val == np.inf:
            # All moves are tabu or invalid, break or increase tenure
            tabu_tenure += 10
            continue
            
        u, v = np.unravel_index(np.argmin(Delta), Delta.shape)
        if u > v:
            u, v = v, u
            
        # Check Aspiration
        new_cost = current_cost + min_val
        if new_cost < best_cost:
            # Accept even if tabu
            pass
        elif tabu_mask[u, v]:
            # Tabu and not improving global best, reject
            # We need to find the next best non-tabu move?
            # For simplicity, if best is tabu, we might just break or skip.
            # Better: Mask the best move and find next best?
            # This is expensive. Let's just set the best move to inf and retry once?
            # Or just accept the risk of cycling?
            # Standard TS: if best is tabu, choose best non-tabu.
            # Since we vectorized, we can just mask out the best and re-min?
            # Let's just set Delta[u, v] = inf and re-min if it was tabu and not aspiration.
            if new_cost >= best_cost:
                Delta[u, v] = np.inf
                Delta[v, u] = np.inf
                min_val = np.min(Delta)
                if min_val == np.inf:
                    tabu_tenure += 10
                    continue
                u, v = np.unravel_index(np.argmin(Delta), Delta.shape)
                if u > v:
                    u, v = v, u
                new_cost = current_cost + min_val
        
        # Accept Move
        if new_cost < current_cost or new_cost < best_cost:
            # Update P
            p[u], p[v] = p[v], p[u]
            current_cost = new_cost
            
            # Update Best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                last_improvement_iter = current_iter
                
            # Update Q
            # Q[i, k] += (A[i, u] - A[i, v]) * (B[k, L_v] - B[k, L_u])
            # L_u, L_v are old locations
            L_u, L_v = p[v], p[u] # Note: p is already updated?
            # Wait, p[u] is now old p[v].
            # We need old L_u and L_v.
            # Let's save before update.
            
            # Re-do update logic with saved locations
            # Actually, I updated p above.
            # Let's trace:
            # old_p_u = p[u], old_p_v = p[v]
            # p[u], p[v] = p[v], p[u]
            # So L_u = old_p_u, L_v = old_p_v.
            # In the code above:
            # p[u], p[v] = p[v], p[u]
            # L_u, L_v = p[v], p[u] -> This is correct. p[v] is now old p[u]. p[u] is now old p[v].
            
            # Update Q
            # diff_A = a[:, u] - a[:, v]
            # diff_B = b[:, L_v] - b[:, L_u]
            # Q += np.outer(diff_A, diff_B)
            diff_A = a[:, u] - a[:, v]
            diff_B = b[:, L_v] - b[:, L_u]
            Q += np.outer(diff_A, diff_B)
            
            # Update C
            C[u] = Q[u, p[u]]
            C[v] = Q[v, p[v]]
            
            # Add to Tabu
            tabu_list.append((u, v, current_iter + tabu_tenure))
            tabu_mask[u, v] = True
            tabu_mask[v, u] = True
            
        else:
            # No improvement found in this iteration
            # Check stagnation
            if current_iter - last_improvement_iter > stagnation_limit:
                # Perturbation
                # Randomly swap 5-10 pairs
                num_swaps = np.random.randint(5, 15)
                for _ in range(num_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
                    # Update Q and C for each swap
                    L_i, L_j = p[j], p[i]
                    diff_A = a[:, i] - a[:, j]
                    diff_B = b[:, L_j] - b[:, L_i]
                    Q += np.outer(diff_A, diff_B)
                    C[i] = Q[i, p[i]]
                    C[j] = Q[j, p[j]]
                    
                # Recalculate cost
                current_cost = get_cost(p)
                
                # Reset tabu
                tabu_list = []
                tabu_mask[:] = False
                last_improvement_iter = current_iter
                tabu_tenure = 40 # Reset tenure

    return best_p.tolist()
