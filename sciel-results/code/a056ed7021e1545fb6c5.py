import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        
        # 1. Build Conflict Graph (Adjacency Matrix)
        # Edge (i, j) exists if v_i + v_j in V (i != j) OR 2*v_i == v_j
        adj = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vi == vj:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # 2. Formulate ILP for Maximum Independent Set
        # Maximize sum(x_i)
        # Subject to: x_i + x_j <= 1 for all edges (i, j)
        #             x_i in {0, 1}
        
        c = -np.ones(n)  # Minimize -sum(x)
        
        # Constraints: A x <= b
        # We need constraints for every edge (i, j) where adj[i,j] == 1
        # x_i + x_j <= 1
        
        rows = []
        cols = []
        b_upper = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    rows.append([i, j])
                    cols.append([0, 1]) # coefficients for x_i and x_j
                    b_upper.append(1)
        
        if not rows:
            # No conflicts, take all
            return list(values)
        
        A = np.zeros((len(rows), n), dtype=float)
        for r, row in enumerate(rows):
            A[r, row[0]] = 1
            A[r, row[1]] = 1
        
        constraints = [
            LinearConstraint(A, ub=np.ones(len(rows)))
        ]
        
        bounds = Bounds(lb=0, ub=1)
        
        # Integrate time check
        if time.time() - start_time > DEADLINE - 2:
            # Fallback if we're out of time
            return [v for v in values if v % 2 != 0]

        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=np.ones(n),
                bounds=bounds
            )
            
            if res.status == 0:  # Optimal solution found
                x = res.x
                # Round to binary
                x_bin = (x > 0.5).astype(int)
                selected_indices = np.where(x_bin == 1)[0]
                result_vals = [values[i] for i in selected_indices]
                
                # Verify validity (should be guaranteed by ILP, but just in case)
                if is_sum_free(result_vals):
                    return result_vals
                else:
                    # Fallback to heuristic if ILP returned invalid (shouldn't happen)
                    pass
            
            # If status is not 0, or we didn't get a valid result, fallback
        except Exception:
            pass

        # Fallback: Greedy Independent Set
        # Sort by degree (ascending)
        degrees = np.sum(adj, axis=1)
        order = np.argsort(degrees)
        
        selected = []
        in_set = np.zeros(n, dtype=bool)
        
        for i in order:
            if in_set[i]:
                continue
            # Check if i conflicts with any already selected
            conflict = False
            for j in selected:
                if adj[i, j]:
                    conflict = True
                    break
            if not conflict:
                selected.append(i)
                in_set[i] = True
                # Mark neighbors as unavailable? No, we just check conflicts on the fly.
                # For speed in fallback, we can just check adj against selected.
        
        result_vals = [values[i] for i in selected]
        
        # Final safety check
        if not is_sum_free(result_vals):
            # Worst case: all odds
            result_vals = [v for v in values if v % 2 != 0]
            
        return result_vals

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # ILP is deterministic and fast for these sizes (n<=240).
    # We can give each instance a generous budget, but strictly monitor total time.
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        if remaining < 5:
             # Fallback for all remaining
             vals = instance(size, seed)
             res = [x for x in vals if x % 2 != 0]
             results.append(res)
             continue
        
        vals = instance(size, seed)
        # Give ILP a specific budget, but it will finish fast anyway.
        # We don't strictly need to pass budget to milp, it just runs.
        # But we must check time before starting if we are close to deadline.
        
        res = solve_instance(vals, remaining)
        
        # Verify
        if not is_sum_free(res):
             res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results
