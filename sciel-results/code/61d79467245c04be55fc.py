import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute all 3-APs: list of (a, b, c) where a < b < c and b = (a+c)/2
    triples = []
    for a in range(1, n + 1):
        for d in range(1, n - a + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            triples.append((a, b, c))
    
    # For each element x, precompute which other elements it conflicts with
    # conflict[x] = set of y such that {x, y} is part of a 3-AP with some z
    # Actually, we need: if x and y are both in the set, can we add z to make a 3-AP?
    # Better: for each pair (x, y) with x < y, if (x+y) is even, mid = (x+y)//2
    #   - if mid is in set and mid != x and mid != y: conflict
    #   - if x+2*(y-x) <= n and that's in set: conflict (x, y, x+2d)
    #   - if y+2*(y-x) <= n and that's in set: conflict (y, y+d, y+2d) where d=y-x, but that's not right
    # 
    # Let me think more carefully. A 3-AP is (a, a+d, a+2d).
    # If we have elements x and y in the set:
    # Case 1: x and y are endpoints: a=x, c=y, d=(y-x)/2 must be integer
    #   mid = (x+y)//2, need mid in set, mid != x, mid != y
    # Case 2: x is first, y is second: a=x, b=y, d=y-x, c=x+2*(y-x)=2y-x
    #   need c in set, c != x, c != y
    # Case 3: x is second, y is first: a=y, b=x, d=x-y, c=y+2*(x-y)=2x-y
    #   need c in set, c != x, c != y
    # Case 4: x is second, y is third: a=y-2d... wait, let me redo
    #   If y is third: a + 2d = y, and a+d = x, so d = y-x, a = 2x-y
    #   need a in set, a != x, a != y
    # Case 5: x is third, y is second: a+d = y, a+2d = x, d = x-y, a = 2y-x
    #   need a in set, a != x, a != y
    # Case 6: x is first, y is third: a=x, c=y, d=(y-x)/2, need d integer
    #   mid = (x+y)//2, need mid in set, mid != x, mid != y
    # 
    # So the conflicts for adding x to a set S are:
    # For each y in S:
    #   - if (x+y) even and (x+y)//2 in S and (x+y)//2 != x and (x+y)//2 != y: conflict
    #   - if 2*y - x in S and 2*y - x != x and 2*y - x != y: conflict (x is third, y is middle)
    #   - if 2*x - y in S and 2*x - y != x and 2*x - y != y: conflict (y is third, x is middle)
    
    def can_add_fast(x, s_set):
        """Check if x can be added to set represented by s_set (a set of ints)."""
        for y in s_set:
            # x and y as endpoints
            s = x + y
            if s % 2 == 0:
                mid = s // 2
                if mid != x and mid != y and mid in s_set:
                    return False
            # x is middle: y, x, z where z = 2*x - y
            z = 2 * x - y
            if z != x and z != y and 1 <= z <= n and z in s_set:
                return False
            # y is middle: x, y, z where z = 2*y - x
            z2 = 2 * y - x
            if z2 != x and z2 != y and 1 <= z2 <= n and z2 in s_set:
                return False
        return True

    def greedy_fill(s_list, s_set):
        """Add all valid elements in increasing order. s_list is bool array, s_set is set of ints."""
        for x in range(1, n + 1):
            if not s_list[x]:
                if can_add_fast(x, s_set):
                    s_list[x] = True
                    s_set.add(x)
        return s_list, s_set

    def get_sum(s_list):
        total = 0.0
        for i in range(1, n + 1):
            if s_list[i]:
                total += recip[i]
        return total

    def local_search(s_list, s_set):
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            current_sum = get_sum(s_list)
            
            # Try removing one element and refilling
            for rem in range(1, n + 1):
                if not s_list[rem]:
                    continue
                
                trial_list = s_list[:]
                trial_set = set(s_set)
                trial_list[rem] = False
                trial_set.discard(rem)
                
                greedy_fill(trial_list, trial_set)
                new_sum = get_sum(trial_list)
                
                if new_sum > current_sum + 1e-12:
                    s_list = trial_list
                    s_set = trial_set
                    current_sum = new_sum
                    improved = True
                    break
            
        return s_list, s_set

    # 1. Initial Greedy
    in_set = [False] * (n + 1)
    in_set_set = set()
    greedy_fill(in_set, in_set_set)
    best_set = in_set[:]
    best_set_set = in_set_set.copy()
    best_sum = get_sum(best_set)
    
    # 2. Local Search on Initial
    improved_list, improved_set = local_search(best_set[:], best_set_set.copy())
    if get_sum(improved_list) > best_sum:
        best_set = improved_list
        best_set_set = improved_set
        best_sum = get_sum(best_set)
        
    # 3. Randomized Restart / Perturbation
    random.seed(42)
    
    while time.time() < deadline - 8.0:
        # Perturb best_set
        perturbed_list = best_set[:]
        perturbed_set = set(best_set_set)
        
        # Remove 1 to 5 random elements
        num_removals = random.randint(1, 5)
        elements_in_set = list(perturbed_set)
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_list[rem_val] = False
            perturbed_set.discard(rem_val)
            elements_in_set.pop(idx)
        
        # Refill
        greedy_fill(perturbed_list, perturbed_set)
        
        # Local Search
        improved_list, improved_set = local_search(perturbed_list, perturbed_set)
        
        new_sum = get_sum(improved_list)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_set = improved_list
            best_set_set = improved_set

    # --- MILP Attempt ---
    # Try to solve the problem using MILP with a time limit
    try:
        rows = []
        cols = []
        vals = []
        b_vec = []
        
        for a, b, c in triples:
            row_idx = len(b_vec)
            rows.append(row_idx); cols.append(a - 1); vals.append(1.0)
            rows.append(row_idx); cols.append(b - 1); vals.append(1.0)
            rows.append(row_idx); cols.append(c - 1); vals.append(1.0)
            b_vec.append(2.0)
        
        A = csr_matrix((vals, (rows, cols)), shape=(len(b_vec), n))
        c_obj = [-recip[i] for i in range(1, n + 1)]
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        constraint = LinearConstraint(A, lb=-np.inf, ub=b_vec)
        
        milp_time_limit = max(1.0, deadline - time.time() - 5.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"time_limit": milp_time_limit, "presolve": "default"})
        
        if res.status == 0 and res.x is not None:
            milp_set_list = [False] * (n + 1)
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_set_list[i + 1] = True
            
            # Validate MILP result
            milp_set = set(i for i in range(1, n+1) if milp_set_list[i])
            valid = True
            for a, b, c in triples:
                if a in milp_set and b in milp_set and c in milp_set:
                    valid = False
                    break
            
            if valid:
                milp_sum = get_sum(milp_set_list)
                if milp_sum > best_sum:
                    best_set = milp_set_list
                    best_set_set = milp_set
                    best_sum = milp_sum
                    
    except Exception:
        pass

    # Extract result
    result = [i for i in range(1, n + 1) if best_set[i]]
    
    # Final validation
    s_set = set(result)
    for a, b, c in triples:
        if a in s_set and b in s_set and c in s_set:
            raise AssertionError("3-AP found")
            
    return result
