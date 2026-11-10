import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for safety

    # Step 1: Generate all subsets of {1, ..., 100} that sum to 200
    subsets = []
    
    def backtrack(start, current_sum, current_subset):
        if time.time() > deadline:
            return
        if current_sum == 200:
            if current_subset:
                subsets.append(tuple(current_subset))
            return
        if current_sum > 200:
            return
        
        for i in range(start, 101):
            if current_sum + i > 200:
                break
            backtrack(i + 1, current_sum + i, current_subset + [i])
            
    backtrack(1, 0, [])
    
    # If no subsets sum to 200, we can return all 1..100
    if not subsets:
        return list(range(1, 101))
        
    # Step 2: Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        n_vars = 100
        n_constraints = len(subsets)
        
        # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
        c = -np.ones(n_vars)
        
        # Build sparse matrix A
        rows = []
        cols = []
        data = []
        b = np.zeros(n_constraints)
        
        for j, sub in enumerate(subsets):
            b[j] = len(sub) - 1
            for i in sub:
                rows.append(j)
                cols.append(i - 1)
                data.append(1)
                
        A = csr_matrix((data, (rows, cols)), shape=(n_constraints, n_vars))
        
        constraints = LinearConstraint(A, lb=-np.inf, ub=b)
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=0, ub=1)
        
        # Set time limit to ensure we return before deadline
        time_limit = max(1, int(deadline - time.time()) - 5)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
        
        if res.status == 0:  # Optimal
            x = res.x
            selected = [i + 1 for i in range(n_vars) if x[i] > 0.5]
            if selected:
                return selected
        else:
            # If not optimal, check if a feasible solution was found (e.g., status 3: time limit)
            if hasattr(res, 'x') and res.x is not None and np.any(res.x > 0.5):
                # Verify feasibility
                is_feasible = True
                for j, sub in enumerate(subsets):
                    s = sum(res.x[i-1] for i in sub)
                    if s > len(sub) - 1 + 1e-9:
                        is_feasible = False
                        break
                if is_feasible:
                    selected = [i + 1 for i in range(n_vars) if res.x[i] > 0.5]
                    if selected:
                        return selected
    except Exception:
        pass

    # Step 3: Fallback - Greedy Hitting Set
    active_bad_subsets = [set(s) for s in subsets]
    removed = set()
    
    while active_bad_subsets:
        if time.time() > deadline:
            break
            
        freq = [0] * 101
        for s in active_bad_subsets:
            for i in s:
                freq[i] += 1
                
        max_freq = 0
        best_elem = -1
        for i in range(1, 101):
            if freq[i] > max_freq:
                max_freq = freq[i]
                best_elem = i
                
        if best_elem == -1:
            break
            
        removed.add(best_elem)
        active_bad_subsets = [s for s in active_bad_subsets if best_elem not in s]
        
    result = [i for i in range(1, 101) if i not in removed]
    
    if not result:
        # Fallback to seed logic if something went wrong
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
        
    return result
