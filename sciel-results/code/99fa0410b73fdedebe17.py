import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    limit = 256
    nums = list(range(2, limit + 1))
    n = len(nums)
    
    # Find all bad triples (i, j, k) corresponding to nums[i], nums[j], nums[k]
    # such that they cannot all be in the set together.
    # Condition: a|bc OR b|ac OR c|ab
    bad_triples = []
    
    # Optimization: iterate i < j < k
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            for k in range(j + 1, n):
                c = nums[k]
                # Check divisibility
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    bad_triples.append((i, j, k))
                    
    if not bad_triples:
        return nums

    # Setup ILP
    # Maximize sum(x) => Minimize -sum(x)
    c_obj = np.full(n, -1.0)
    
    # Constraints: x_i + x_j + x_k <= 2 for each bad triple
    A = lil_matrix((len(bad_triples), n), dtype=float)
    b_ub = np.full(len(bad_triples), 2.0)
    
    for idx, (i, j, k) in enumerate(bad_triples):
        A[idx, i] = 1.0
        A[idx, j] = 1.0
        A[idx, k] = 1.0
        
    A = A.tocsr()
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            x_sol = res.x
            selected = []
            for i in range(n):
                if x_sol[i] > 0.5:
                    selected.append(nums[i])
            return selected
    except Exception:
        pass

    # Fallback: Greedy (descending)
    result = []
    for x in reversed(nums):
        valid = True
        # Check against existing elements
        # We need to ensure that for any a, b in result, and x, the triple (a,b,x) is not bad.
        # Bad if a|bx or b|ax or x|ab.
        for i in range(len(result)):
            a = result[i]
            for j in range(i + 1, len(result)):
                b = result[j]
                if (b * x) % a == 0 or (a * x) % b == 0 or (a * b) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            result.append(x)
            
    return result
