import sys
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element using ILP.
    """
    required_diffs = list(range(1, 31))
    
    # Lower bound for D:
    # k elements -> k(k-1)/2 differences.
    # We need at least 30 differences.
    # k=9 -> 36 diffs.
    # Min D for k=9 Sidon set is roughly sqrt(2*30) ~ 8? No, Sidon sets grow like k^2.
    # A(30) is the max size of Sidon set in [1, 30]. It's small.
    # We need to cover 1..30.
    # Let's iterate D starting from 30.
    
    start_time = time.time()
    timeout = 110
    
    for D in range(30, 200):
        if time.time() - start_time > timeout:
            return greedy_construct()
            
        result = solve_ilp(D)
        if result is not None:
            return result
            
    return greedy_construct()

def solve_ilp(D):
    """
    Solves the ILP for a fixed D.
    Variables:
    x_i for i in 0..D (binary)
    z_{i,j} for 0 <= i < j <= D (binary), represents if both i and j are selected.
    """
    n = D + 1
    # Total variables: n + n(n-1)/2
    # This might be too large for large D.
    # For D=100, n=101, pairs ~ 5000. Total vars ~ 5100.
    # Constraints:
    # 1. x_0 = 1
    # 2. sum x_i <= 32
    # 3. z_{i,j} <= x_i
    # 4. z_{i,j} <= x_j
    # 5. z_{i,j} >= x_i + x_j - 1
    # 6. For each d in 1..D, sum_{j-i=d} z_{i,j} <= 1  (Sidon property)
    # 7. For each d in 1..30, sum_{j-i=d} z_{i,j} >= 1  (Coverage)
    
    # Optimization: We only care about d in 1..30 for coverage, but Sidon property applies to ALL differences.
    # However, if a difference > 30 is repeated, it violates Sidon.
    # So constraint 6 must apply to all d from 1 to D.
    
    # This ILP is quite large. Let's check if we can reduce variables.
    # We know x_0 = 1.
    # We can fix x_0 = 1.
    
    # Let's try a smaller D first.
    
    # Build A matrix for milp.
    # Variables:
    # 0..n-1 : x_0 ... x_D
    # n..n+p-1 : z_{i,j} for i<j
    
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    p = len(pairs)
    total_vars = n + p
    
    # Objective: Minimize 0 (feasibility problem)
    # Actually, we just want to find *any* solution.
    c = np.zeros(total_vars)
    
    # Bounds: all binary [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Constraints
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # 1. x_0 = 1
    # Variable index 0
    A_rows.append(np.eye(1, total_vars, k=0)[0]) # Just 1 at index 0
    lb_rows.append(1)
    ub_rows.append(1)
    
    # 2. Sum x_i <= 32
    row = np.zeros(total_vars)
    row[:n] = 1
    A_rows.append(row)
    lb_rows.append(0)
    ub_rows.append(32)
    
    # 3,4,5. Linearization constraints for each pair (i, j)
    # z_{i,j} - x_i <= 0
    # z_{i,j} - x_j <= 0
    # z_{i,j} - x_i - x_j >= -1  =>  -z_{i,j} + x_i + x_j <= 1
    
    for idx, (i, j) in enumerate(pairs):
        z_idx = n + idx
        
        # z <= x_i
        r1 = np.zeros(total_vars)
        r1[z_idx] = 1
        r1[i] = -1
        A_rows.append(r1)
        lb_rows.append(0)
        ub_rows.append(0)
        
        # z <= x_j
        r2 = np.zeros(total_vars)
        r2[z_idx] = 1
        r2[j] = -1
        A_rows.append(r2)
        lb_rows.append(0)
        ub_rows.append(0)
        
        # z >= x_i + x_j - 1  =>  -z + x_i + x_j <= 1
        r3 = np.zeros(total_vars)
        r3[z_idx] = -1
        r3[i] = 1
        r3[j] = 1
        A_rows.append(r3)
        lb_rows.append(0)
        ub_rows.append(1)
        
    # 6. Sidon property: For each d in 1..D, sum z_{i, i+d} <= 1
    for d in range(1, D + 1):
        row = np.zeros(total_vars)
        count = 0
        for i in range(n - d):
            j = i + d
            # Find index of pair (i, j)
            # Pairs are ordered i<j. 
            # Index calculation:
            # Number of pairs with first element < i is i * n - i*(i-1)/2 ? No.
            # It's easier to use a dictionary or precomputed map.
            pass
        
    # This manual construction is slow and error-prone.
    # Let's use a dictionary for pair indices.
    pair_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_index[(i, j)] = idx
            idx += 1
            
    # Re-build A_rows for Sidon constraints
    # Note: A_rows already has the linearization constraints.
    # We need to add the sum constraints.
    
    # To be efficient, we should build the whole A matrix at once or append.
    # Since we are in a loop for D, and D is small initially, let's just build it.
    
    # Reset A_rows to include only what we have so far? 
    # The code above appended to A_rows. 
    # Let's restart the construction cleanly inside solve_ilp.
    
    A_list = []
    lb_list = []
    ub_list = []
    
    # x_0 = 1
    r = np.zeros(total_vars)
    r[0] = 1
    A_list.append(r)
    lb_list.append(1)
    ub_list.append(1)
    
    # Sum x <= 32
    r = np.zeros(total_vars)
    r[:n] = 1
    A_list.append(r)
    lb_list.append(0)
    ub_list.append(32)
    
    # Linearization
    for (i, j), z_idx in pair_index.items():
        # z - x_i <= 0
        r = np.zeros(total_vars)
        r[z_idx + n] = 1
        r[i] = -1
        A_list.append(r)
        lb_list.append(0)
        ub_list.append(0)
        
        # z - x_j <= 0
        r = np.zeros(total_vars)
        r[z_idx + n] = 1
        r[j] = -1
        A_list.append(r)
        lb_list.append(0)
        ub_list.append(0)
        
        # -z + x_i + x_j <= 1
        r = np.zeros(total_vars)
        r[z_idx + n] = -1
        r[i] = 1
        r[j] = 1
        A_list.append(r)
        lb_list.append(0)
        ub_list.append(1)
        
    # Sidon: sum_{i} z_{i, i+d} <= 1 for all d
    for d in range(1, D + 1):
        r = np.zeros(total_vars)
        for i in range(n - d):
            j = i + d
            if (i, j) in pair_index:
                r[n + pair_index[(i, j)]] = 1
        A_list.append(r)
        lb_list.append(0)
        ub_list.append(1)
        
    # Coverage: sum_{i} z_{i, i+d} >= 1 for d in 1..30
    for d in range(1, 31):
        r = np.zeros(total_vars)
        for i in range(n - d):
            j = i + d
            if (i, j) in pair_index:
                r[n + pair_index[(i, j)]] = 1
        A_list.append(r)
        lb_list.append(1)
        ub_list.append(np.inf)
        
    A = np.array(A_list)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    
    constraint = LinearConstraint(A, lb, ub)
    
    integrality = np.array([1] * total_vars) # All binary
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    except Exception:
        return None
        
    if res.status != 0:
        return None
        
    x = res.x
    # Extract x_i
    elements = [i for i in range(n) if x[i] > 0.5]
    
    # Verify
    if 0 not in elements:
        return None
    diffs = set()
    valid = True
    for i in range(len(elements)):
        for j in range(i+1, len(elements)):
            d = elements[j] - elements[i]
            if d in diffs:
                valid = False
                break
            diffs.add(d)
        if not valid:
            break
            
    if not valid:
        return None
        
    if not all(d in diffs for d in range(1, 31)):
        return None
        
    return elements

def greedy_construct():
    """Fallback greedy constructor."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

sys.setrecursionlimit(10000)
