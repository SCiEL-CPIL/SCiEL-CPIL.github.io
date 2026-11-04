import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    # Heuristic Initialization: Rank Matching
    # Heavy flows (high sum of row) to Central locations (low sum of row)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow strength (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance centrality (ascending)
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Initial Cost
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    deadline = time.time() + 110  # Safety margin
    
    # Tabu List: store (i, j) swaps, i < j
    tabu = {}
    tabu_tenure = 20
    
    while time.time() < deadline:
        improved = False
        # First Fit Descent
        for i in range(n):
            if time.time() > deadline: break
            for j in range(i + 1, n):
                if time.time() > deadline: break
                
                key = (i, j)
                if key in tabu and tabu[key] > 0:
                    # Check Aspiration: if this swap improves best_cost, do it
                    # We need to calculate delta first to check aspiration
                    pass 
                
                x = p[i]
                y = p[j]
                
                # Calculate Delta
                # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[y,p[k]] - b[x,p[k]]) 
                #       + (a[i,j] + a[j,i]) * (b[y,x] - b[x,y])
                
                # Vectorized calculation
                diff_a = a[i] - a[j]
                diff_b = b[y, p] - b[x, p]
                
                total_dot = np.dot(diff_a, diff_b)
                
                # Subtract terms for k=i and k=j
                # k=i term: (a[i,i] - a[j,i]) * (b[y, p[i]] - b[x, p[i]]) = (a[i,i] - a[j,i]) * (b[y,x] - b[x,x])
                term_i = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
                # k=j term: (a[i,j] - a[j,j]) * (b[y, p[j]] - b[x, p[j]]) = (a[i,j] - a[j,j]) * (b[y,y] - b[x,y])
                term_j = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
                
                delta_sum = total_dot - term_i - term_j
                
                # Add direct interaction term
                # The sum excluded i and j. The original cost has a[i,j]b[x,y] and a[j,i]b[y,x].
                # New cost has a[i,j]b[y,x] and a[j,i]b[x,y].
                # Change: (a[i,j]b[y,x] + a[j,i]b[x,y]) - (a[i,j]b[x,y] + a[j,i]b[y,x])
                # = (a[i,j] - a[j,i]) * (b[y,x] - b[x,y]) ... wait
                # Let's re-verify the direct term.
                # Old: a_ij b_xy + a_ji b_yx
                # New: a_ij b_yx + a_ji b_xy
                # Diff: a_ij (b_yx - b_xy) + a_ji (b_xy - b_yx) = (a_ij - a_ji)(b_yx - b_xy)
                # My previous formula: (a[i,j] + a[j,i]) * (b[y,x] - b[x,y]) was wrong?
                # Let's check:
                # (a_ij - a_ji)(b_yx - b_xy)
                # If matrices are symmetric, a_ij = a_ji and b_xy = b_yx, so this term is 0.
                # For tho150, matrices are symmetric. So this term is 0.
                # But for general case, use the derived one.
                delta_direct = (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
                
                delta = delta_sum + delta_direct
                
                if delta < 0:
                    # Check Tabu Aspiration
                    if key in tabu and tabu[key] > 0:
                        if current_cost + delta >= best_cost:
                            continue # Reject if not improving global best
                    
                    # Apply Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    tabu[key] = tabu_tenure
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
                    improved = True
                    break # Restart j loop for this i? Or break i loop?
                    # Standard First Fit: break inner, continue outer? 
                    # Usually break inner and restart from j=0 or i+1? 
                    # Breaking inner and letting outer continue to i+1 is fine.
        
        # Decrement Tabu counters
        for key in list(tabu.keys()):
            tabu[key] -= 1
            if tabu[key] <= 0:
                del tabu[key]
        
        if not improved:
            # Local Minimum. Perturb.
            # Perform 10-20 random swaps
            for _ in range(15):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = compute_cost(p)
            # Clear tabu to allow free movement?
            tabu.clear()

    return best_p.tolist()
