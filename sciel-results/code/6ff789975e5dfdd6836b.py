import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin for 150s limit

    def instance(size, seed, limit=1000):
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
        
        # 1. Build Conflict Graph
        # We want to find a Max Independent Set in a graph where an edge (i, j) 
        # exists if v_i + v_j is in V (including i=j case which is 2*v_i in V).
        # Actually, the condition is: no a, b, c in S such that a+b=c.
        # This is equivalent to S being an independent set in a graph G on V
        # where edge (i, j) exists if v_i + v_j is in V.
        # Note: If i=j, then 2*v_i must not be in V if v_i is in S? 
        # Wait. If a=b, then 2a=c. So if we pick a, we cannot pick c=2a.
        # Also if we pick a and b, we cannot pick c=a+b.
        # So the "conflict" is between pairs {a,b} and element c=a+b.
        # This is a 3-uniform hypergraph independent set problem, or can be viewed as:
        # Let G be a graph on V. Edge (i, j) exists if v_i + v_j is in V.
        # If we select a set S, we require that for all a, b in S, a+b is NOT in S.
        # This is NOT simply an independent set in a standard graph because the constraint 
        # involves 3 variables (a,b,c). 
        # However, we can model it as: 
        # For each pair (i, j) with i <= j, if v_i + v_j = v_k, then we cannot have 
        # {i, j, k} all in S.
        # This is a 3-SAT like constraint or 3-uniform hypergraph independent set.
        
        # Alternative View:
        # Let's stick to the conflict graph idea but be careful.
        # If we define edges between i and k if there exists j such that v_i + v_j = v_k? No.
        
        # Let's use the hypergraph approach or simply local search on the set.
        # For local search, we need to evaluate "conflicts" quickly.
        # A set S is sum-free iff for all a, b in S, a+b not in S.
        
        # Representation:
        # We will represent the solution as a boolean array x of length n.
        # We need to check validity and perform flips.
        
        # Precompute sums
        # sums[i][j] = index k such that values[i] + values[j] == values[k], or -1 if not in V.
        # This allows O(1) check for conflict introduction/removal.
        
        sums = np.full((n, n), -1, dtype=np.int32)
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        for i in range(n):
            vi = values[i]
            for j in range(i, n):
                vj = values[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    sums[i, j] = k
                    if i != j:
                        sums[j, i] = k

        # Conflict count:
        # A "conflict" is a tuple (a, b, c) in S such that a+b=c.
        # We can count the number of such triples.
        # Let C(x) be the number of triples (i, j, k) with x_i=x_j=x_k=1 and sums[i,j]=k.
        # We want to maximize sum(x) subject to C(x)=0.
        
        # Local Search Strategy:
        # 1. Greedy Initialization
        # 2. Local Search: Flip bits to reduce conflicts or increase size.
        #    - If C > 0, try to remove elements involved in conflicts.
        #    - If C == 0, try to add elements that don't create conflicts.
        
        def get_conflicts(x):
            # x is bool array
            count = 0
            # Iterate over all pairs (i, j) with i <= j
            # If x[i] and x[j], check if x[k] where k=sums[i,j]
            # This is O(n^2) per evaluation which is too slow for many iterations.
            # We need incremental updates.
            pass

        # Better: Maintain conflict count incrementally.
        # Let's define a function to calculate the "energy" or validity.
        # Since n is small (max 240), O(n^2) is ~57k ops. 
        # If we do 1000 restarts * 1000 steps, that's 500M ops. Might be tight in Python.
        # We should use NumPy for the heavy lifting or C-optimized loops if possible, 
        # but pure Python loops for n=240 are okay if not too many iterations.
        
        # Let's try a different angle: 
        # The problem is equivalent to finding a large independent set in a 3-uniform hypergraph.
        # Greedy heuristic:
        # Sort elements. 
        # Often, taking all elements from one "side" (e.g., all positive, or all > n/2) works well.
        # Actually, for sum-free sets in [-N, N], the set of numbers in (N/2, N] is sum-free.
        # Size ~ N/2. Total N. Ratio 0.5.
        # The theoretical bound is (n+2)/3 ~ 0.33n.
        # So 0.5n is a very good lower bound.
        
        # Strategy:
        # 1. Try "Upper Half": All x > 0 such that 2x > max(V)? No, just x > max(V)/2.
        # 2. Try "All Positive": If 2x > max(V) for all x in Pos? No.
        #    If we take all positive numbers, a+b=c is possible if c is also positive and in V.
        #    So "All Positive" is not necessarily sum-free.
        
        # Let's use a Local Search with incremental conflict tracking.
        
        def initial_greedy():
            # Start with empty set, add elements one by one if they don't create conflict.
            # To maximize size, we should add elements that have fewest conflicts?
            # Or just iterate in random order?
            
            # Heuristic: Elements with large absolute value are less likely to be sums of two others.
            # Sort by abs value descending.
            order = sorted(range(n), key=lambda i: abs(values[i]), reverse=True)
            
            x = np.zeros(n, dtype=bool)
            # We need to track conflicts.
            # Let's just build the set and check at the end? No, incremental is better.
            
            # Actually, let's just use the "Upper Half" heuristic as a strong baseline.
            # max_val = max(values)
            # candidates = [i for i in range(n) if values[i] > max_val / 2]
            # This set is sum-free because if a, b > max_val/2, then a+b > max_val, 
            # so a+b cannot be in V (since max(V) = max_val).
            # This guarantees a sum-free set of size approx n/2 (if distribution is uniform).
            
            max_val = max(values)
            min_val = min(values)
            
            # Strategy A: Upper half
            x_a = np.zeros(n, dtype=bool)
            for i in range(n):
                if values[i] > max_val / 2:
                    x_a[i] = True
            
            # Strategy B: Lower half (negative)
            x_b = np.zeros(n, dtype=bool)
            for i in range(n):
                if values[i] < min_val / 2:
                    x_b[i] = True
            
            # Strategy C: Random Greedy
            x_c = np.zeros(n, dtype=bool)
            rng = random.Random(42)
            perm = list(range(n))
            rng.shuffle(perm)
            # We need to check if adding v_i creates a conflict with existing set.
            # Conflict if exists a, b in S such that a+b=v_i OR v_i+a=b OR v_i+b=a.
            # i.e., v_i is a sum of two in S, or v_i is a summand for a sum in S.
            
            s_set = set()
            for i in perm:
                vi = values[i]
                conflict = False
                # Check if vi = a + b for a, b in s_set
                # Check if vi + a = b for a, b in s_set
                # Check if a + vi = b for a, b in s_set
                # This is O(|S|^2) per insertion. Total O(n^3) worst case. n=240 -> 13M. Acceptable.
                
                for a in s_set:
                    for b in s_set:
                        if a + b == vi:
                            conflict = True
                            break
                        if a + vi == b:
                            conflict = True
                            break
                        if vi + b == a:
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    x_c[i] = True
                    s_set.add(vi)
            
            # Evaluate sizes
            size_a = np.sum(x_a)
            size_b = np.sum(x_b)
            size_c = np.sum(x_c)
            
            best_x = x_a
            if size_b > np.sum(best_x):
                best_x = x_b
            if size_c > np.sum(best_x):
                best_x = x_c
                
            return best_x

        x = initial_greedy()
        
        # Local Search Phase
        # We want to improve x.
        # Move: Flip one bit.
        # If x_i = 0, flipping to 1 increases size by 1. We must ensure no new conflicts.
        # If x_i = 1, flipping to 0 decreases size by 1. We only do this if it allows adding multiple other elements?
        # Or if the current set has conflicts (which it shouldn't if we start valid and only add valid ones).
        
        # Let's assume x is valid (sum-free).
        # We try to add elements.
        # If we can't add any single element, we try 2-flips: remove i, add j, k?
        
        # Efficient check for adding element i to valid set S:
        # i is compatible if:
        # 1. No a, b in S such that a+b = v_i
        # 2. No a, b in S such that v_i + a = b  (i.e., b - v_i = a in S)
        # 3. No a, b in S such that a + v_i = b  (i.e., a - v_i = b in S)
        
        # Precompute for current S:
        # S_sum: set of all a+b for a,b in S.
        # S_diff: set of all a-b for a,b in S.
        
        def can_add(x, i):
            vi = values[i]
            # Get indices of currently selected elements
            # For efficiency, maintain list of selected indices
            # But x is array. Let's just iterate.
            # n is small.
            sel = [j for j in range(n) if x[j]]
            if not sel:
                return True
            
            # Check 1: vi = a + b
            # Check 2: vi + a = b => b - a = vi
            # Check 3: a + vi = b => b - a = -vi
            
            # Build sets of sums and diffs of S
            # This is O(|S|^2).
            s_set = set(sel)
            for a in sel:
                va = values[a]
                for b in sel:
                    vb = values[b]
                    if va + vb == vi:
                        return False
                    if va + vi == vb:
                        return False
                    if vi + vb == va:
                        return False
            return True

        # This is O(n^3) per step if we do this naively in a loop.
        # Let's optimize.
        
        # Better Local Search:
        # 1. Start with valid x.
        # 2. Loop:
        #    - Find an element i not in x that can be added.
        #    - If found, add it.
        #    - If not found, try to "shake": remove a random element, see if we can add more.
        
        # Given time constraints, let's do a simpler version:
        # Multiple restarts of Random Greedy, keep the best.
        
        best_size = 0
        best_x = np.zeros(n, dtype=bool)
        
        # Time budget for restarts
        # n=240, O(n^3) greedy is ~13M ops. In Python ~0.5s? 
        # We can do ~20 restarts in 10s.
        
        # Let's implement a faster "Random Greedy with Backtracking" or just many random orderings.
        
        def random_greedy(seed_val):
            rng = random.Random(seed_val)
            perm = list(range(n))
            rng.shuffle(perm)
            
            x = np.zeros(n, dtype=bool)
            s_vals = []
            
            for i in perm:
                vi = values[i]
                conflict = False
                # Optimization: if len(s_vals) is large, this is slow.
                # But n=240 is small.
                for a in s_vals:
                    # Check a + b = vi (b in s_vals)
                    # Check a + vi = b (b in s_vals)
                    # Check vi + b = a (b in s_vals)
                    # We can check:
                    # 1. vi - a in s_set?
                    # 2. a + vi in s_set?
                    # 3. a - vi in s_set?
                    
                    # We need s_set for O(1) lookup.
                    pass
                
                # Re-implement with set
                # This is getting complex. Let's stick to the O(n^2) check per item but vectorize?
                # No, let's just run multiple random greedy passes.
                
                # Actually, the "Upper Half" heuristic is very strong.
                # Let's combine:
                # 1. Upper Half
                # 2. Random Greedy (few times)
                # 3. Local Search on the best result: try to add any remaining element.
                
                return x # Placeholder

        # Let's refine the Random Greedy to be fast.
        def fast_random_greedy(seed_val):
            rng = random.Random(seed_val)
            perm = list(range(n))
            rng.shuffle(perm)
            
            x = np.zeros(n, dtype=bool)
            s_set = set()
            s_list = []
            
            for i in perm:
                vi = values[i]
                ok = True
                # Check if vi can be added
                # 1. vi = a + b for a, b in S
                # 2. vi + a = b for a, b in S  => b - a = vi
                # 3. a + vi = b for a, b in S  => b - a = -vi
                
                # Precompute sums and diffs of S? 
                # Updating sums/diffs on insertion is O(|S|).
                # Checking against new vi is O(1) if we have sets.
                
                # Let's maintain:
                # S: set of values
                # S_Sums: set of a+b
                # S_Diffs: set of a-b
                
                # Initialization:
                # S_Sums = {0} ? No.
                
                # Let's just do the O(|S|^2) check. It's simple and n is small.
                for a in s_list:
                    va = values[a]
                    for b in s_list:
                        vb = values[b]
                        if va + vb == vi:
                            ok = False
                            break
                        if va + vi == vb:
                            ok = False
                            break
                        if vi + vb == va:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    x[i] = True
                    s_list.append(i)
                    s_set.add(vi)
            
            return x

        # Run several random greeds
        num_restarts = 50
        current_time = time.time()
        
        for r in range(num_restarts):
            if time.time() - current_time > time_budget * 0.5:
                break
            x = fast_random_greedy(r)
            size = np.sum(x)
            if size > best_size:
                best_size = size
                best_x = x.copy()
        
        # Also try the deterministic upper/lower half
        max_val = max(values)
        min_val = min(values)
        
        x_up = np.zeros(n, dtype=bool)
        for i in range(n):
            if values[i] > max_val / 2:
                x_up[i] = True
        if np.sum(x_up) > best_size:
            best_size = np.sum(x_up)
            best_x = x_up.copy()
            
        x_low = np.zeros(n, dtype=bool)
        for i in range(n):
            if values[i] < min_val / 2:
                x_low[i] = True
        if np.sum(x_low) > best_size:
            best_size = np.sum(x_low)
            best_x = x_low.copy()

        # Local Search: Try to add any element to best_x
        # Check if any element not in best_x can be added.
        # We need to verify best_x is sum-free first (it should be).
        
        # If we have time, try 1-flip improvements (add only, since we start valid).
        # If we can't add any, we are stuck. 
        # To escape, we might need to remove one and add two?
        # Given time, let's just return the best of the greedy runs.
        
        res = [values[i] for i in range(n) if best_x[i]]
        
        # Final verification
        if not is_sum_free(res):
             # Fallback to safe set
             res = [x for x in values if x > max_val / 2]
        
        return res

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    weights = [1, 2, 4]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
             vals = instance(size, seed)
             res = [x for x in vals if x > max(vals) / 2]
             results.append(res)
             continue
        
        budget = remaining * (weights[i] / total_weight)
        budget = max(budget, 2.0)
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        if not is_sum_free(res):
             res = [x for x in vals if x > max(vals) / 2]
        
        results.append(res)
        
    return results
