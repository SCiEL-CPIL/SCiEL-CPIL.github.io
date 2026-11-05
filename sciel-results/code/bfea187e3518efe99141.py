import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = 256
    # Generate all AP constraints
    # We store (indices, length) for each AP prefix
    # Actually, we just need the indices.
    
    # To speed up, we can precompute the constraint matrix A and vectors b
    # But building it might take time.
    
    # Let's try C=1, then C=2, etc.
    
    # Helper to check feasibility for a given C
    def check_feasibility(C):
        # Variables x_i in [0,1]
        # Constraints:
        # For each AP (indices i1...ik, length k):
        # L - C <= 2 * sum(x_i) <= L + C
        # => 2 * sum(x_i) - L >= -C
        # => -2 * sum(x_i) + L <= C  (or 2*sum(x_i) <= L+C)
        
        # We construct the constraint matrix on the fly or precompute?
        # Precomputing 180k rows x 256 cols is ~46MB of floats. OK.
        
        pass

    # Precomputation
    # List of APs
    aps = []
    for d in range(1, n):
        for s in range(d):
            # Get indices
            idxs = list(range(s, n, d))
            # Add all prefixes
            # Prefix 1: [s]
            # Prefix 2: [s, s+d]
            # ...
            # We can just store the full list of indices for the maximal AP,
            # and then in the constraint generation, we iterate prefixes.
            # But for MILP, we need explicit rows.
            # It is more efficient to build the matrix rows.
            pass
            
    # Actually, building the matrix in Python loops for 180k rows might be slow.
    # Let's optimize the generation.
    
    # Alternative: Use a heuristic first to get a good C, then use MILP to verify/improve?
    # Or just use MILP directly.
    
    # Let's try to build the constraints efficiently.
    # A = []
    # b_lower = []
    # b_upper = []
    
    # For each AP:
    #   For each prefix:
    #      Row has 1s at indices.
    #      Lower: L - C
    #      Upper: L + C
    #      (Divided by 2? No, keep integers or floats)
    #      2 * sum >= L - C  =>  sum >= (L-C)/2
    #      2 * sum <= L + C  =>  sum <= (L+C)/2
    
    # We can use LinearConstraint(A, lb, ub)
    
    # Let's estimate time.
    # If we precompute A, lb, ub for C=1, it's fast.
    # Then solve.
    # If infeasible, update lb, ub for C=2 and solve.
    
    # We need to be careful with the size of A.
    # 180,000 rows, 256 cols.
    # Sparse matrix is better. scipy.optimize.milp supports sparse.
    
    from scipy.sparse import lil_matrix
    
    def get_constraints(C):
        A = lil_matrix((0, n), dtype=float)
        lb = []
        ub = []
        
        # Add constraints for all APs
        # d from 1 to n-1
        for d in range(1, n):
            for s in range(d):
                # Get maximal AP
                # Indices: s, s+d, ...
                # We need to add constraints for every prefix.
                # Let current_sum_indices = []
                # curr_sum_val = 0 (not needed, we just count length)
                # length = 0
                # idx = s
                # while idx < n:
                #    length += 1
                #    Add constraint for prefix of length 'length'
                #    ...
                #    idx += d
                
                # Optimization:
                # Instead of adding every prefix as a separate row immediately,
                # we can just iterate.
                
                idx = s
                length = 0
                while idx < n:
                    length += 1
                    # Create row
                    row = lil_matrix((1, n), dtype=float)
                    row[0, idx] = 1.0
                    A = A.vstack(row)
                    
                    # Constraints:
                    # sum(x) >= (L - C) / 2
                    # sum(x) <= (L + C) / 2
                    lb.append((length - C) / 2.0)
                    ub.append((length + C) / 2.0)
                    
                    idx += d
        
        return A.tocsr(), np.array(lb), np.array(ub)

    # This loop is O(n^2 log n) which is ~180k iterations.
    # Creating lil_matrix and vstacking 180k times is SLOW in Python.
    
    # Better approach:
    # Build the sparse matrix using COO format or just lists of (row, col, val).
    
    def get_constraints_fast(C):
        rows = []
        cols = []
        vals = []
        lb = []
        ub = []
        row_idx = 0
        
        for d in range(1, n):
            for s in range(d):
                idx = s
                length = 0
                while idx < n:
                    length += 1
                    rows.append(row_idx)
                    cols.append(idx)
                    vals.append(1.0)
                    
                    lb.append((length - C) / 2.0)
                    ub.append((length + C) / 2.0)
                    
                    row_idx += 1
                    idx += d
        
        from scipy.sparse import coo_matrix
        A = coo_matrix((vals, (rows, cols)), shape=(row_idx, n)).tocsr()
        return A, np.array(lb), np.array(ub)
    
    # This is much faster.
    
    best_signs = None
    best_C = None
    
    # Try C = 1, 2, 3...
    for C in range(1, 10):
        if time.time() > deadline:
            break
            
        try:
            A, lb, ub = get_constraints_fast(C)
            # Objective: 0
            # Variables: 256 binary
            # Bounds: [0, 1]
            
            # Note: lb and ub can be negative.
            # x_i are binary.
            
            res = milp(
                c=np.zeros(n),
                constraints=LinearConstraint(A, lb, ub),
                integrality=np.ones(n),
                bounds=Bounds(0, 1)
            )
            
            if res.status == 0: # Optimal
                # Extract solution
                # x_i is 0 or 1
                x = res.x
                # Convert to signs
                signs = [1 if xi > 0.5 else -1 for xi in x]
                return signs
            else:
                # Infeasible, try next C
                continue
                
        except Exception as e:
            # If MILP fails, fallback
            break
            
    # Fallback: Heuristic
    # If we reach here, we didn't find a solution via MILP in time or it failed.
    # Use a fast local search.
    
    import random
    rng = random.Random(42)
    
    # Initialize
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    def calc_max_disc(s):
        max_d = 0
        for d in range(1, n):
            for st in range(d):
                curr_sum = 0
                idx = st
                while idx < n:
                    curr_sum += s[idx]
                    if abs(curr_sum) > max_d:
                        max_d = abs(curr_sum)
                    idx += d
        return max_d
    
    current_disc = calc_max_disc(signs)
    
    while time.time() < deadline:
        # Find worst AP
        # This is expensive, so maybe just random flips?
        # Or flip a random index.
        i = rng.randint(0, n-1)
        signs[i] *= -1
        new_disc = calc_max_disc(signs)
        if new_disc < current_disc:
            current_disc = new_disc
        else:
            signs[i] *= -1 # Revert
            
    return signs
