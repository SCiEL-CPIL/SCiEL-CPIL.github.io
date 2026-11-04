"""
Solves the Quadratic Assignment Problem (QAP) for the tho150 instance.
Uses Simulated Annealing with efficient O(N) cost delta updates.
"""
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Safety margin: 110 seconds out of 120 (or 150)
    deadline = start_time + 110.0
    
    # Precompute some constants if needed, though a and b are passed directly.
    # Ensure a and b are float64 for numerical stability in SA, 
    # but the cost is integer. We can use int64 for exact cost if we are careful, 
    # but float64 is safer for SA temperature calculations.
    # The problem statement says a and b are int64.
    
    def get_cost(p):
        # p is a numpy array of ints
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This is O(N^2) but done rarely (for initialization and final check)
        # Vectorized: sum(a * b[p[:, None], p[None, :]])
        return np.sum(a * b[np.ix_(p, p)])
    
    def delta_swap(p, k, l, a, b):
        """
        Calculate change in cost if we swap positions k and l in permutation p.
        p[k] = i, p[l] = j.
        After swap: p'[k] = j, p'[l] = i.
        """
        i = p[k]
        j = p[l]
        
        # Vector part: sum_{m != k, l} (a[i,m] - a[j,m]) * (b[m, p[l]] - b[m, p[k]])
        # Note: b[m, p[l]] is b[m, j] and b[m, p[k]] is b[m, i]
        # So we need columns j and i of b.
        b_col_j = b[:, j]
        b_col_i = b[:, i]
        
        a_row_i = a[i, :]
        a_row_j = a[j, :]
        
        # Dot product of (a_row_i - a_row_j) and (b_col_j - b_col_i)
        # This sums over all m.
        total_dot = np.dot(a_row_i - a_row_j, b_col_j - b_col_i)
        
        # Subtract contributions for m = k and m = l which are included in the dot product
        # but need special handling because the 'destination' indices p[k] and p[l] change 
        # for the terms involving i and j directly, and the 'source' m=k,l terms 
        # interact with the new destinations.
        
        # The formula derived:
        # Delta = sum_{m!=k,l} (A[i,m]-A[j,m])(B[m, j] - B[m, i]) 
        #       + (A[i,k]-A[j,k])(B[k, j] - B[k, i])  <-- Wait, in the sum m=k, 
        #         the term is (A[i,k]-A[j,k])(B[k, p[l]] - B[k, p[k]])
        #         Here p[l]=j, p[k]=i. So it is (A[i,k]-A[j,k])(B[k, j] - B[k, i]).
        #         This IS included in the dot product because m=k is in the range.
        #         So we don't subtract it, we just keep it?
        #         Let's re-verify.
        #         The dot product calculates sum_{m=0}^{N-1} (A[i,m]-A[j,m]) * (B[m, j] - B[m, i]).
        #         This includes m=k and m=l.
        #         For m=k: term is (A[i,k]-A[j,k]) * (B[k, j] - B[k, i]).
        #         For m=l: term is (A[i,l]-A[j,l]) * (B[l, j] - B[l, i]).
        #         
        #         However, the actual cost change also involves the direct interaction between i and j 
        #         changing from (i at k, j at l) to (j at k, i at l).
        #         The dot product accounts for interactions of i and j with ALL other nodes m.
        #         Does it account for the interaction between i and j themselves?
        #         Yes, when m=j (which is p[l]) and m=i (which is p[k])?
        #         No, the indices in the dot product are:
        #         Left: A[i,m] - A[j,m]
        #         Right: B[m, j] - B[m, i]
        #         
        #         Let's check the specific terms in the total cost change:
        #         Old Cost parts involving i or j:
        #         1. i interacts with all m: sum_m A[i,m] B[p[m], i] (since i is at k, p[k]=i)
        #         2. j interacts with all m: sum_m A[j,m] B[p[m], j] (since j is at l, p[l]=j)
        #         3. Others interact with i: sum_{m!=i,j} A[m,i] B[p[m], p[i]] = sum_{m!=i,j} A[m,i] B[p[m], i]
        #         4. Others interact with j: sum_{m!=i,j} A[m,j] B[p[m], p[j]] = sum_{m!=i,j} A[m,j] B[p[m], j]
        #         
        #         New Cost parts:
        #         1'. j at k: sum_m A[j,m] B[p'[m], j]. p'[k]=j, p'[l]=i.
        #             For m=k: A[j,k] B[j, j].
        #             For m=l: A[j,l] B[i, j].
        #             For m!=k,l: A[j,m] B[p[m], j].
        #         2'. i at l: sum_m A[i,m] B[p'[m], i].
        #             For m=k: A[i,k] B[j, i].
        #             For m=l: A[i,l] B[i, i].
        #             For m!=k,l: A[i,m] B[p[m], i].
        #         
        #         Change = (New 1' + New 2') - (Old 1 + Old 2) ... wait, this double counts i-j interaction?
        #         The total cost is sum_{x,y} A[x,y] B[p[x], p[y]].
        #         Terms involving i (as x): sum_y A[i,y] B[p[i], p[y]] = sum_y A[i,y] B[i, p[y]].
        #         Terms involving j (as x): sum_y A[j,y] B[p[j], p[y]] = sum_y A[j,y] B[j, p[y]].
        #         Terms involving i (as y): sum_x A[x,i] B[p[x], p[i]] = sum_x A[x,i] B[p[x], i].
        #         Terms involving j (as y): sum_x A[x,j] B[p[x], p[j]] = sum_x A[x,j] B[p[x], j].
        #         
        #         This decomposition is messy. Let's stick to the delta formula derived earlier:
        #         Delta = sum_{m != k, l} (A[i,m] - A[j,m]) (B[m, j] - B[m, i]) 
        #                 + (A[i,k] - A[j,k]) (B[k, j] - B[k, i])
        #                 + (A[i,l] - A[j,l]) (B[l, j] - B[l, i])
        #                 + A[i,j] (B[j, i] - B[i, j]) 
        #                 + A[j,i] (B[i, j] - B[j, i])
        #         
        #         The dot product `total_dot` computes:
        #         sum_{m=0}^{N-1} (A[i,m] - A[j,m]) (B[m, j] - B[m, i])
        #         = sum_{m != k, l} ... + (A[i,k]-A[j,k])(B[k, j]-B[k, i]) + (A[i,l]-A[j,l])(B[l, j]-B[l, i])
        #         
        #         So the first three lines of the Delta formula are exactly `total_dot`.
        #         
        #         The remaining terms are the direct interaction changes:
        #         Old direct: A[i,j] B[i,j] + A[j,i] B[j,i]
        #         New direct: A[i,j] B[j,i] + A[j,i] B[i,j]
        #         Change: A[i,j] (B[j,i] - B[i,j]) + A[j,i] (B[i,j] - B[j,i])
        #         
        #         So Delta = total_dot + A[i,j] * (b[j, i] - b[i, j]) + A[j,i] * (b[i, j] - b[j, i])
        
        direct_change = a[i, j] * (b[j, i] - b[i, j]) + a[j, i] * (b[i, j] - b[j, i])
        
        return total_dot + direct_change

    # Initial Solution: Random
    # We can try a few different starting points
    best_p = None
    best_cost = np.inf
    
    # Number of restarts
    # Each restart runs SA for a certain number of steps or time
    steps_per_run = 5000
    
    while time.time() < deadline - 5.0:  # Leave 5s for final processing
        # Initialize permutation
        p = np.random.permutation(n).astype(np.int64)
        current_cost = get_cost(p)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # SA Parameters
        # Initial temperature should be high enough to accept bad moves
        # Estimate typical delta
        # For random swaps, delta is roughly proportional to N * max_val
        # Let's estimate by sampling a few deltas
        deltas = []
        for _ in range(10):
            k = np.random.randint(n)
            l = np.random.randint(n)
            if k != l:
                deltas.append(abs(delta_swap(p, k, l, a, b)))
        if not deltas:
            continue
        avg_delta = np.mean(deltas)
        if avg_delta == 0:
            avg_delta = 1.0
            
        T_start = avg_delta * 10
        T_end = avg_delta * 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / steps_per_run)
        
        current_T = T_start
        local_best_p = p.copy()
        local_best_cost = current_cost
        
        for step in range(steps_per_run):
            if time.time() > deadline - 2.0:
                break
            
            k = np.random.randint(n)
            l = np.random.randint(n)
            if k == l:
                continue
                
            d = delta_swap(p, k, l, a, b)
            
            if d < 0 or np.random.rand() < np.exp(-d / current_T):
                # Accept
                p[k], p[l] = p[l], p[k]
                current_cost += d
                
                if current_cost < local_best_cost:
                    local_best_cost = current_cost
                    local_best_p = p.copy()
                    
            current_T *= cooling_rate
            
        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_p = local_best_p.copy()

    return best_p.tolist()
