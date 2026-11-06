import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def solve_ilp(vals, time_limit):
        """
        Solve the Maximum Sum-Free Subset problem using MILP.
        Variables: x_i in {0, 1} for each element i.
        Objective: Maximize sum(x_i)
        Constraints: For every triple (a, b, c) in vals such that a+b=c,
                     x_a + x_b + x_c <= 2
                     (This ensures not all three are selected.
                      Note: The problem states "no a+b=c with a,b,c in it".
                      If a=b, it's 2a=c. The constraint x_a + x_a + x_c <= 2
                      becomes 2x_a + x_c <= 2.
                      If x_a=1, x_c must be 0. Correct.
                      If x_a=0, x_c can be 1. Correct.
                      If a!=b, x_a+x_b+x_c <= 2.
                      If x_a=1, x_b=1, then x_c must be 0. Correct.
                     )
        """
        n = len(vals)
        if n == 0:
            return set()
        
        # Precompute triples
        vals_set = set(vals)
        triples = []
        # To avoid duplicate constraints for commutative addition (a+b=c and b+a=c),
        # we can just iterate all pairs. The ILP solver handles redundant constraints,
        # but it's better to minimize them.
        # Actually, for a+b=c, if we iterate all i, j, we get both (i,j) and (j,i).
        # Constraint: x_i + x_j + x_k <= 2.
        # These are identical constraints. We should avoid duplicates.
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        seen = set()
        for i in range(n):
            a = vals[i]
            for j in range(n):
                b = vals[j]
                c = a + b
                if c in vals_set:
                    k = val_to_idx[c]
                    # Sort indices to identify unique triple sets for constraint generation
                    # Note: a+b=c is symmetric in a,b. So (i,j,k) and (j,i,k) are same constraint.
                    # Also, if a=b, i=j.
                    key = tuple(sorted((i, j, k)))
                    if key not in seen:
                        seen.add(key)
                        triples.append((i, j, k))
        
        if not triples:
            return set(vals)
            
        # Build A and b for constraints: A @ x <= b
        # Number of constraints
        m = len(triples)
        
        # Using sparse matrix for efficiency if m is large
        # However, scipy.milp in newer versions might prefer dense or specific sparse formats.
        # Let's try dense first if n and m are small enough.
        # n <= 240, m <= n^2 = 57600.
        # 57600 x 240 matrix is ~14 million floats. ~112 MB. This might be too heavy for dense.
        # Let's use scipy.sparse.
        
        from scipy.sparse import csr_matrix
        
        row_idx = []
        col_idx = []
        data = []
        
        for r, (i, j, k) in enumerate(triples):
            if i == j:
                # Constraint: 2*x_i + x_k <= 2
                row_idx.append(r)
                col_idx.append(i)
                data.append(2)
                row_idx.append(r)
                col_idx.append(k)
                data.append(1)
            else:
                # Constraint: x_i + x_j + x_k <= 2
                row_idx.append(r)
                col_idx.append(i)
                data.append(1)
                row_idx.append(r)
                col_idx.append(j)
                data.append(1)
                row_idx.append(r)
                col_idx.append(k)
                data.append(1)
                
        A = csr_matrix((data, (row_idx, col_idx)), shape=(m, n))
        b = np.ones(m)
        
        # Objective: Maximize sum(x) -> Minimize -sum(x)
        c = -np.ones(n)
        
        # Bounds: x in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x must be integer
        integrality = np.ones(n)
        
        # Constraints: A @ x <= b
        # LinearConstraint(A, lb=-np.inf, ub=b)
        
        constraints = LinearConstraint(A, lb=-np.inf, ub=b)
        
        # Set time limit
        # scipy.milp uses HiGHS by default.
        # time_limit is in seconds.
        # We have roughly (deadline - time.time()) - 2 seconds for safety.
        current_time = time.time()
        remaining = deadline - current_time - 2.0
        if remaining < 1.0:
             # Not enough time for ILP, return empty or greedy
             return set()
             
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": remaining, "mip_rel_gap": 0.01, "presolve": True}
            )
            
            if res.status == 0: # Optimal
                x = res.x
                # Threshold for numerical stability
                selected = set(vals[i] for i in range(n) if x[i] > 0.5)
                return selected
            elif res.status == 9: # Time limit reached, but feasible solution found?
                # HiGHS status 9 is "Time limit reached". 
                # res.x might contain the best incumbent.
                if res.x is not None:
                    x = res.x
                    selected = set(vals[i] for i in range(n) if x[i] > 0.5)
                    # Verify sum-free property just in case of numerical issues
                    if is_sum_free(selected):
                        return selected
            # If status is other (e.g., infeasible, error), return empty
            return set()
        except Exception:
            return set()

    def greedy_solution(vals, seed):
        """Fallback heuristic if ILP fails or is too slow."""
        rng = random.Random(seed)
        best = set()
        for _ in range(20):
            shuffled = list(vals)
            rng.shuffle(shuffled)
            S = set()
            for x in shuffled:
                ok = True
                for a in S:
                    if (x - a) in S or (x + a) in S:
                        ok = False
                        break
                if ok:
                    S.add(x)
            if len(S) > len(best):
                best = S
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        # Try ILP first
        # Allocate more time to larger instances? 
        # Or equal time? 
        # Let's give 40s to n=60, 40s to n=120, 40s to n=240.
        # Total 120s. Plus overhead.
        
        time_per_instance = 40.0
        
        start_inst = time.time()
        S_ilp = solve_ilp(vals, time_per_instance)
        
        if not S_ilp:
            # Fallback to greedy if ILP returned empty or failed
            S_ilp = greedy_solution(vals, seed)
            
        # Verify
        if not is_sum_free(S_ilp):
            S_ilp = greedy_solution(vals, seed)
            if not is_sum_free(S_ilp):
                # Last resort: odd numbers (always sum-free? No, 1+3=4. 
                # Actually, set of numbers with same sign and same parity?
                # If all positive: a+b > a, a+b > b. 
                # If all positive, sum-free means no a+b=c.
                # A simple valid set is all numbers with the same sign and large magnitude?
                # Just take all positive numbers. If a,b > 0, a+b > max(a,b). 
                # So if we pick all positive numbers, is it sum-free?
                # No, 1, 2, 3 are all positive. 1+2=3. Not sum-free.
                # Bourgain bound guarantees (n+2)/3.
                # We can just take a random subset of size n/3 and verify/repair?
                # For safety, use a very conservative set: 
                # All numbers x such that x > max(vals)/2? 
                # If x, y > M/2, then x+y > M. If M is max(vals), x+y is not in vals.
                # So the set of elements greater than max(vals)/2 is sum-free.
                # Size might be small, but it's valid.
                M = max(vals)
                S_fallback = set(x for x in vals if x > M / 2.0)
                if not S_fallback:
                    S_fallback = set(vals[:1])
                S_ilp = S_fallback

        results.append(sorted(S_ilp))
        
        # Check time
        if time.time() > deadline - 5:
             # If we are running out of time, we must ensure we return something valid for remaining instances
             # But the loop structure handles each instance. If we run out, the next solve_ilp will have time_limit < 1 and return empty, 
             # triggering greedy. Greedy is fast.
             pass

    return results

# Quick test to ensure no syntax errors
if __name__ == "__main__":
    res = construct()
    print(res)
