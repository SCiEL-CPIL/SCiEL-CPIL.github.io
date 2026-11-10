import time
import numpy as np
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # --- Method 1: Exact Solution via Integer Linear Programming ---
    # We formulate the problem as a 0-1 ILP.
    # Maximize sum(x_i) subject to sum(x_i for i in F) <= |F|-1 for all forbidden subsets F.
    
    def solve_with_milp():
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import itertools
            
            # 1. Enumerate all subsets of {1, ..., 100} that sum to 200
            # dp[s] stores a set of bitmasks representing subsets of {1..100} that sum to s
            # Bit i in mask corresponds to element (i+1)
            dp = [set() for _ in range(201)]
            dp[0].add(0)
            
            # We only care about sums up to 200
            for i in range(1, 101):
                val = i
                bit = 1 << (i - 1)
                # Iterate backwards to avoid using the same element twice
                for s in range(200, val - 1, -1):
                    if dp[s - val]:
                        # Add new subsets formed by including i
                        new_masks = {m | bit for m in dp[s - val]}
                        dp[s].update(new_masks)
                
                if time.time() > deadline - 5.0:
                    return None
            
            forbidden_subsets = list(dp[200])
            
            if not forbidden_subsets:
                # No subset sums to 200, so we can take all
                return list(range(1, 101))
            
            # 2. Construct the constraint matrix
            # Variables x_0 ... x_99 correspond to elements 1 ... 100
            n_vars = 100
            n_constraints = len(forbidden_subsets)
            
            # A is n_constraints x n_vars
            # b is n_constraints
            # c is n_vars (objective coefficients)
            
            # Using sparse matrix for efficiency if n_constraints is large
            # But n_constraints is likely small enough for dense or simple list of lists
            # Let's count first.
            
            # We need A such that A @ x <= b
            # For each constraint k (forbidden subset F_k):
            # sum_{j in F_k} x_j <= |F_k| - 1
            
            # Build A as a list of lists or numpy array
            # If n_constraints is too large, we might run out of memory or time.
            # Let's assume it's manageable.
            
            A = np.zeros((n_constraints, n_vars), dtype=int)
            b = np.zeros(n_constraints, dtype=int)
            
            for k, mask in enumerate(forbidden_subsets):
                # Determine which elements are in the mask
                # And count them
                count = 0
                for j in range(100):
                    if mask & (1 << j):
                        A[k, j] = 1
                        count += 1
                b[k] = count - 1
            
            # Objective: Maximize sum(x_i)
            # milp minimizes c @ x, so we set c = -1
            c = -np.ones(n_vars)
            
            # Bounds: x_i in [0, 1]
            bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
            
            # Constraints: A @ x <= b
            # LinearConstraint(A, lb=-np.inf, ub=b)
            # Note: milp expects constraints in form lb <= A @ x <= ub
            # So we want -inf <= A @ x <= b
            constraints = LinearConstraint(A, lb=-np.inf, ub=b)
            
            # Integrality: all variables are integer
            integrality = np.ones(n_vars)
            
            # Solve
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                # res.x contains the solution
                # res.x might have small floating point errors, round to nearest int
                x = np.round(res.x).astype(int)
                selected = [i + 1 for i in range(100) if x[i] == 1]
                if selected:
                    return selected
                else:
                    return [1] # Fallback, should not happen
            
            return None
            
        except Exception as e:
            #print(f"MILP failed: {e}")
            return None

    best_solution = solve_with_milp()
    
    if best_solution is not None:
        # Verify the solution
        # Check if any subset sums to 200
        # This is a good sanity check
        def is_valid(subset):
            # Use bitset to check if 200 is reachable
            bs = np.zeros(201, dtype=bool)
            bs[0] = True
            for x in subset:
                if x > 200: continue
                # new_sums = bs shifted by x
                # We only care up to 200
                new_sums = np.zeros(201, dtype=bool)
                new_sums[x:] = bs[:201-x]
                bs = bs | new_sums
                if bs[200]:
                    return False
            return True

        if is_valid(best_solution):
            return best_solution
        else:
            # If MILP gave an invalid solution (shouldn't happen), fall back
            pass

    # --- Method 2: Heuristic Fallback ---
    # If MILP fails or is too slow, use a strong heuristic.
    
    def greedy_construct(order):
        selected = []
        bs = np.zeros(201, dtype=bool)
        bs[0] = True
        for x in order:
            if x > 200: continue
            new_sums = np.zeros(201, dtype=bool)
            new_sums[x:] = bs[:201-x]
            if not (new_sums[200]):
                selected.append(x)
                bs = bs | new_sums
        return selected

    def local_search(initial_set):
        # Try to add any missing element
        selected = list(initial_set)
        sel_set = set(selected)
        bs = np.zeros(201, dtype=bool)
        bs[0] = True
        for x in selected:
            new_sums = np.zeros(201, dtype=bool)
            new_sums[x:] = bs[:201-x]
            bs = bs | new_sums
            
        # Add as many as possible
        improved = True
        while improved:
            improved = False
            for x in range(1, 101):
                if x not in sel_set:
                    new_sums = np.zeros(201, dtype=bool)
                    new_sums[x:] = bs[:201-x]
                    if not new_sums[200]:
                        selected.append(x)
                        sel_set.add(x)
                        bs = bs | new_sums
                        improved = True
        return selected

    best = []
    best_size = 0
    
    # Try several random orders
    for seed in [42, 123, 456, 789, 1011, 2023, 3141, 2718, 1618, 9999]:
        if time.time() > deadline - 10.0:
            break
        rng = random.Random(seed)
        order = list(range(1, 101))
        rng.shuffle(order)
        
        # Greedy
        cand = greedy_construct(order)
        # Local Search
        cand = local_search(cand)
        
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
            
    # Iterated Local Search with Perturbation
    if time.time() < deadline - 5.0:
        current = best
        for _ in range(50): # Number of iterations
            if time.time() > deadline - 5.0:
                break
            
            # Perturb: remove k random elements
            k = random.randint(2, 5)
            perturbed = current[:]
            if len(perturbed) > k:
                to_remove = random.sample(perturbed, k)
                for x in to_remove:
                    perturbed.remove(x)
            
            # Re-optimize
            new_cand = local_search(perturbed)
            
            if len(new_cand) > len(current):
                current = new_cand
                best = current
                best_size = len(current)
            # Or accept with some probability (Simulated Annealing style)
            # For simplicity, only accept if better or equal (with some diversity)
            elif len(new_cand) == len(current) and random.random() < 0.1:
                current = new_cand

    if not best:
        # Final fallback: just return a valid small set
        return [1]

    return best

