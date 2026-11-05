import time
import numpy as np
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Precompute primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(200)
    
    # Map prime to index in primes list
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Variables: x_{p, r}
    # Indexing: var_idx = sum_{q < p} q + r
    # Let's create a list of (p, r, var_index)
    vars_list = []
    var_index = 0
    prime_var_ranges = {} # p -> (start_idx, end_idx)
    for p in primes:
        start_idx = var_index
        for r in range(p):
            vars_list.append((p, r))
            var_index += 1
        end_idx = var_index
        prime_var_ranges[p] = (start_idx, end_idx)
        
    n_vars = var_index
    
    # Precompute coverage for each variable for a given L
    # This depends on L, so we do it inside the feasibility check.
    
    def check_feasible(L, time_limit=15):
        if L <= 0:
            return True, []
        
        # Build A matrix for coverage constraints: A * x >= 1
        # Rows: 1..L
        # Cols: n_vars
        # A[i-1, j] = 1 if i % p == r for var j
        
        rows = []
        cols = []
        data = []
        
        for j, (p, r) in enumerate(vars_list):
            # Indices i in 1..L such that i % p == r
            # i = k*p + r
            # If r == 0, i = p, 2p, ...
            # If r > 0, i = r, r+p, ...
            # Note: i must be >= 1.
            if r == 0:
                start_i = p
            else:
                start_i = r
            if start_i > L:
                continue
            # Generate indices
            # np.arange(start_i, L + 1, p)
            # This creates a list of row indices
            idxs = np.arange(start_i, L + 1, p)
            if len(idxs) > 0:
                # Convert to 0-based for 1..L
                idxs = idxs - 1
                rows.extend(idxs)
                cols.extend([j] * len(idxs))
                data.extend([1] * len(idxs))
        
        if not rows:
            return False, []
            
        A_c = sparse.csc_matrix((data, (rows, cols)), shape=(L, n_vars))
        
        # Constraints:
        # 1. Coverage: A_c * x >= 1
        # 2. At most one residue per prime: sum_{r} x_{p,r} <= 1
        # 3. Budget: sum (1/p) x_{p,r} <= 1
        
        # Build constraint 2 and 3
        rows2 = []
        cols2 = []
        data2 = []
        
        # Constraint 2: For each prime p, sum of x_{p,r} <= 1
        for p in primes:
            start_idx, end_idx = prime_var_ranges[p]
            # Indices start_idx to end_idx - 1
            idxs = np.arange(start_idx, end_idx)
            rows2.extend([0] * len(idxs)) # Placeholder, we will build separate constraints
            # Actually, it's easier to build the full constraint matrix for all constraints
            # Let's combine them.
            
        # Let's build the full constraint matrix for milp
        # milp expects:
        # A_ub * x <= b_ub
        # A_eq * x == b_eq
        # lb <= x <= ub
        
        # We have:
        # -A_c * x <= -1  (from A_c * x >= 1)
        # Sum_{r in p} x_{p,r} <= 1
        # Sum (1/p) x_{p,r} <= 1
        
        # Total rows = L + len(primes) + 1
        
        total_rows = L + len(primes) + 1
        rows_all = []
        cols_all = []
        data_all = []
        b_ub = []
        
        # 1. Coverage: -A_c * x <= -1
        # A_c is (L, n_vars). We need -A_c.
        # Since A_c is sparse, we can just negate data.
        A_neg = -A_c
        # Get COO
        A_coo = A_neg.tocoo()
        rows_all.extend(A_coo.row)
        cols_all.extend(A_coo.col)
        data_all.extend(A_coo.data)
        b_ub.extend([-1] * L)
        
        # 2. Prime constraints: sum_{r} x_{p,r} <= 1
        for p in primes:
            start_idx, end_idx = prime_var_ranges[p]
            idxs = np.arange(start_idx, end_idx)
            row_idx = L + prime_to_idx[p]
            rows_all.extend([row_idx] * len(idxs))
            cols_all.extend(idxs)
            data_all.extend([1] * len(idxs))
            b_ub.append(1)
            
        # 3. Budget: sum (1/p) x_{p,r} <= 1
        # Coefficients are 1/p for each var.
        row_idx = L + len(primes)
        for j, (p, r) in enumerate(vars_list):
            rows_all.append(row_idx)
            cols_all.append(j)
            data_all.append(1.0 / p)
        b_ub.append(1.0)
        
        A_ub = sparse.csc_matrix((data_all, (rows_all, cols_all)), shape=(total_rows, n_vars))
        
        # Bounds: 0 <= x <= 1
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        integrality = np.ones(n_vars)
        
        try:
            res = milp(
                c=np.zeros(n_vars), # Objective doesn't matter, just feasibility
                constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
            
            if res.status == 0: # Optimal
                # Extract solution
                x = res.x
                classes = []
                for j, (p, r) in enumerate(vars_list):
                    if x[j] > 0.5:
                        classes.append([p, r])
                return True, classes
            else:
                return False, []
                
        except Exception as e:
            # Timeout or error
            return False, []

    # Binary search for max L
    low = 10
    high = 5000
    best_L = 0
    best_classes = []
    
    while low <= high:
        if time.time() > deadline - 20:
            break
            
        mid = (low + high) // 2
        feasible, classes = check_feasible(mid)
        
        if feasible:
            best_L = mid
            best_classes = classes
            low = mid + 1
        else:
            high = mid - 1
            
    # If we found a feasible set, verify it
    if best_classes:
        # Verify budget and coverage
        s = Fraction(0)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s <= 1:
            # Check coverage up to best_L
            covered = np.zeros(best_L + 1, dtype=bool)
            for p, r in best_classes:
                start = p if r == 0 else r
                if start <= best_L:
                    idx = np.arange(start, best_L + 1, p)
                    covered[idx] = True
            if np.all(covered[1:]):
                return best_classes
    
    # Fallback if ILP failed or found nothing
    # Use a simple greedy
    def greedy_fallback():
        primes_f = sieve(100)
        classes = []
        s = Fraction(0)
        covered = np.zeros(2000, dtype=bool)
        for p in primes_f:
            if s + Fraction(1, p) > 1:
                continue
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = p if r == 0 else r
                if start > 2000:
                    continue
                idx = np.arange(start, 2000 + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([p, best_r])
            s += Fraction(1, p)
            start = p if best_r == 0 else best_r
            if start <= 2000:
                idx = np.arange(start, 2000 + 1, p)
                covered[idx] = True
        return classes
        
    fb_classes = greedy_fallback()
    if fb_classes:
        return fb_classes
        
    return [[2, 1]]

