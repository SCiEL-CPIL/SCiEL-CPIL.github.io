import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict list: pairs (i, j) with i < j and lcm(nums[i], nums[j]) <= 256
    conflicts = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            # lcm(a,b) = a*b//g <= 256
            if a * b <= 256 * g:
                conflicts.append((i, j))
    
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
        adj = [set() for _ in range(n)]
        for i, j in conflicts:
            adj[i].add(j)
            adj[j].add(i)
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        best_set = selected
    else:
        # Extract selected indices
        best_set = set(i for i in range(n) if result.x[i] > 0.5)
    
    # Verify validity
    result_list = [nums[i] for i in best_set]
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                # Invalid - fall back to greedy
                order = sorted(range(n), key=lambda i: -1.0/nums[i])
                selected = set()
                adj = [set() for _ in range(n)]
                for i, j in conflicts:
                    adj[i].add(j)
                    adj[j].add(i)
                for i in order:
                    if all(j not in selected for j in adj[i]):
                        selected.add(i)
                result_list = [nums[i] for i in selected]
                break
    
    if not result_list:
        result_list = [2]
    
    return result_list
