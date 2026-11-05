import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = start_time + 82.0  # Safety margin before 90s kill
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict list: pairs (i, j) with i < j and lcm(nums[i], nums[j]) <= 256
    conflicts = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            # lcm(a,b) = a*b//g <= 256
            if a * b <= 256 * g:
                conflicts.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # Objective: maximize sum(1/nums[i] * x[i])
    # For minimization, minimize -sum(1/nums[i] * x[i])
    c = np.array([-1.0 / nums[i] for i in range(n)])
    
    # Constraints: x[i] + x[j] <= 1 for each conflict (i,j)
    m = len(conflicts)
    if m > 0:
        A = np.zeros((m, n))
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1.0
            A[k, j] = 1.0
        lb = np.zeros(m)
        ub = np.ones(m)
    else:
        A = None
        lb = None
        ub = None
    
    # Variable bounds: 0 <= x[i] <= 1, integer
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    
    if A is not None:
        constraints = LinearConstraint(A, lb=lb, ub=ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n)), lb=0, ub=1)
    
    integrality = np.ones(n)  # All variables are binary/integer
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        result = None
    
    # Fallback: greedy if MIP fails
    if result is None or not result.success or result.x is None:
        # Greedy: sort by weight descending, add if no conflict
        order = sorted(range(n), key=lambda i: -1.0/nums[i])
        selected = set()
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        best_set = selected
    else:
        # Extract selected indices
        best_set = set(i for i in range(n) if result.x[i] > 0.5)
    
    # Local search polish: try to improve by swapping or adding elements
    if time.time() < DEADLINE:
        def valid_set(s):
            lst = list(s)
            for idx_a in range(len(lst)):
                for idx_b in range(idx_a + 1, len(lst)):
                    a, b = nums[lst[idx_a]], nums[lst[idx_b]]
                    g = math.gcd(a, b)
                    if a * b <= 256 * g:
                        return False
            return True
        
        def sum_weight(s):
            return sum(1.0 / nums[i] for i in s)
        
        best = set(best_set)
        best_val = sum_weight(best)
        improved = True
        max_iters = 500
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1
            # Try adding each non-selected element
            for i in range(n):
                if i not in best:
                    new_set = best | {i}
                    if valid_set(new_set):
                        new_val = sum_weight(new_set)
                        if new_val > best_val + 1e-12:
                            best = new_set
                            best_val = new_val
                            improved = True
                            break
            # Try swapping: remove one, add another
            if not improved:
                for i in list(best):
                    for j in range(n):
                        if j not in best:
                            new_set = (best - {i}) | {j}
                            if valid_set(new_set):
                                new_val = sum_weight(new_set)
                                if new_val > best_val + 1e-12:
                                    best = new_set
                                    best_val = new_val
                                    improved = True
                                    break
                    if improved:
                        break
            # Try removing a low-weight element if it allows adding higher ones
            if not improved:
                for i in list(best):
                    if len(best) > 1:
                        new_set = best - {i}
                        # Try adding back any element that was conflicting
                        for j in range(n):
                            if j not in new_set:
                                test_set = new_set | {j}
                                if valid_set(test_set):
                                    test_val = sum_weight(test_set)
                                    if test_val > best_val + 1e-12:
                                        best = test_set
                                        best_val = test_val
                                        improved = True
                                        break
                        if improved:
                            break
    
    # Final verification
    result_list = [nums[i] for i in best]
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                # Invalid - fall back to greedy
                order = sorted(range(n), key=lambda i: -1.0/nums[i])
                selected = set()
                for i in order:
                    if all(j not in selected for j in adj[i]):
                        selected.add(i)
                result_list = [nums[i] for i in selected]
                break
    
    if not result_list:
        result_list = [2]
    
    return result_list
