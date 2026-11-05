"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free(values):
        """
        Find a maximum sum-free subset of 'values'.
        Uses ILP: maximize sum(x_i) subject to x_i + x_j + x_k <= 2 for all triples (i,j,k)
        where values[i] + values[j] = values[k].
        """
        n = len(values)
        if n == 0:
            return []
        
        # Find all conflict triples (i, j, k) such that values[i] + values[j] == values[k]
        # We need x_i + x_j + x_k <= 2 for each such triple.
        # Note: i, j, k are indices. The values are distinct, so i, j, k are distinct indices?
        # Not necessarily. If a+b=c, and a,b,c are distinct values, then indices are distinct.
        # Since values are distinct nonzero integers, if a+b=c, then a,b,c must be distinct values.
        # (If a=b, then 2a=c. If a=c, then b=0, but 0 is not in the set. If b=c, then a=0.)
        # So all conflict triples involve 3 distinct indices.
        
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # Efficiently find conflicts
        # For each pair (i, j), check if values[i] + values[j] is in the set.
        # This is O(n^2). For n=240, that's 57600 checks, which is fast.
        
        conflict_constraints = []
        
        # We can optimize by only checking i < j to avoid duplicates, but the constraint is symmetric in i,j.
        # Actually, the constraint is x_i + x_j + x_k <= 2.
        # We just need to add each unique triple once.
        
        # Let's build a set of conflicts to avoid duplicates if we iterate carefully, 
        # or just iterate all pairs.
        
        # Optimization: use a set of values for O(1) lookup
        val_set = set(values)
        
        # To reduce the number of constraints, we can note that if (i,j,k) is a conflict,
        # then (j,i,k) is the same constraint. So iterate i < j.
        
        for i in range(n):
            for j in range(i+1, n):
                s = values[i] + values[j]
                if s in val_set:
                    k = val_to_idx[s]
                    # We have a conflict (i, j, k)
                    # Constraint: x_i + x_j + x_k <= 2
                    # In milp, we define A_ub * x <= b_ub
                    # We'll accumulate rows.
                    # To be memory efficient, we might want to check time periodically.
                    # But n is small (<=240), so number of constraints is at most ~n^2/2 = 28800.
                    # This is very small for an ILP solver.
                    
                    # Create a sparse row for this constraint
                    # scipy milp takes dense or sparse. Let's use a list of tuples for sparse.
                    # Actually, let's just build the matrix incrementally if it's small,
                    # or use sparse.
                    
                    # Let's use a list to store (row, col, val) for sparse matrix construction.
                    # But since n is small, dense might be fine too.
                    # However, the number of constraints M can be up to ~n^2.
                    # M x n matrix. For n=240, M~28800, so 28800*240 ~ 7 million floats.
                    # That's ~56 MB. Acceptable.
                    
                    conflict_constraints.append((i, j, k))
        
        # If no conflicts, all elements can be included
        if not conflict_constraints:
            return list(values)
        
        # Build the ILP
        # Variables: x_0, ..., x_{n-1}, binary, 0 or 1.
        # Maximize: sum(x_i)
        # Subject to:
        #   x_i + x_j + x_k <= 2 for each conflict (i,j,k)
        #   0 <= x_i <= 1
        
        # scipy.optimize.milp minimizes c @ x. So we minimize -sum(x_i).
        c = -np.ones(n)
        
        # Constraints
        # A_ub @ x <= b_ub
        # We have M constraints.
        M = len(conflict_constraints)
        
        # Use sparse matrix for efficiency
        from scipy.sparse import lil_matrix
        
        A_ub = lil_matrix((M, n), dtype=float)
        b_ub = np.ones(M)
        
        for idx, (i, j, k) in enumerate(conflict_constraints):
            A_ub[idx, i] = 1
            A_ub[idx, j] = 1
            A_ub[idx, k] = 1
        
        A_ub = A_ub.tocsr()
        
        integrality = np.ones(n)  # All variables are binary
        bounds = Bounds(lb=0, ub=1)
        
        # Time check
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Fallback to simple heuristic
            return [x for x in values if x % 2]
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A_ub, ub=b_ub),
                integrality=integrality,
                bounds=bounds,
            )
            
            if res.status == 0:  # Optimal
                # res.x contains the solution
                # Due to floating point, values might be 0.999999 or 1.000001
                selected = [values[i] for i in range(n) if res.x[i] > 0.5]
                return selected
            else:
                # If ILP fails, use fallback
                return [x for x in values if x % 2]
                
        except Exception:
            return [x for x in values if x % 2]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        # Time check before starting a new instance
        if time.time() - start_time > deadline - 20:
            # Not enough time for a complex solve, use heuristic for remaining
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = instance(size, seed)
        subset = solve_sum_free(vals)
        results.append(subset)
        
        # Time check after solve
        if time.time() - start_time > deadline - 10:
            # Use heuristic for remaining
            for sz, sd in ((120, 802), (240, 803)):
                if sz != size:
                    # This logic is a bit fragile, let's just break and fill rest
                    pass
            # Simpler: just break and let the loop handle remaining with heuristic if time is tight
            # But the loop structure above handles it by checking at the start of the next iteration.
            # However, if we are at the last instance, we just return.
            pass

    return results

# EVOLVE-BLOCK-END
