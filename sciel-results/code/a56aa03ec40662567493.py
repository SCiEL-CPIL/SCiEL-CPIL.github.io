"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def solve_sum_free(vals, seed=42, time_limit=30.0):
        """
        Find a large sum-free subset of vals.
        Uses WalkSAT-style local search on the 2-SAT formulation:
        For each triple (a, b, c) with a+b=c, at most two can be present.
        This is not strictly 2-SAT (it's a 3-variable constraint), but we can
        model it as: not (a and b and c). We use a heuristic local search.
        """
        n = len(vals)
        if n == 0:
            return []
        
        # Build constraint list: for each pair (a,b), if a+b is in vals, we have a constraint
        val_to_idx = {v: i for i, v in enumerate(vals)}
        constraints = []  # list of (i, j, k) where vals[i] + vals[j] = vals[k]
        seen_pairs = set()
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Each constraint involves 3 variables; store as tuple of indices
                    # We'll use a set to avoid duplicate constraints
                    key = tuple(sorted((i, j, k)))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        constraints.append((i, j, k))
        
        # Initial solution: use parity (odd numbers) as a baseline
        sol = [1 if v % 2 != 0 else 0 for v in vals]
        
        def count_violations(sol):
            cnt = 0
            for (i, j, k) in constraints:
                if sol[i] and sol[j] and sol[k]:
                    cnt += 1
            return cnt
        
        def score(sol):
            return sum(sol)
        
        best_sol = sol[:]
        best_score = score(sol)
        
        rng = random.Random(seed)
        start_time = time.time()
        max_iter = 5000000  # safety cap
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            cur_violations = count_violations(sol)
            cur_score = score(sol)
            
            if cur_score > best_score:
                best_sol = sol[:]
                best_score = cur_score
            
            if cur_violations == 0:
                # Try to improve by adding more elements
                # Find indices that are 0 and can be set to 1 without creating violations
                candidates = []
                for idx in range(n):
                    if sol[idx] == 0:
                        # Check if setting sol[idx]=1 would create a violation
                        # It would create a violation if there exist (i,j,k) where idx is one of them
                        # and the other two are already 1
                        can_add = True
                        for (i, j, k) in constraints:
                            if idx == i:
                                if sol[j] and sol[k]:
                                    can_add = False
                                    break
                            elif idx == j:
                                if sol[i] and sol[k]:
                                    can_add = False
                                    break
                            elif idx == k:
                                if sol[i] and sol[j]:
                                    can_add = False
                                    break
                        if can_add:
                            candidates.append(idx)
                
                if candidates:
                    # Add a random candidate
                    idx = rng.choice(candidates)
                    sol[idx] = 1
                    continue
                else:
                    # No more elements can be added; try to flip a 1 to 0 to allow others
                    ones = [i for i in range(n) if sol[i]]
                    if not ones:
                        break
                    idx = rng.choice(ones)
                    sol[idx] = 0
                    continue
            
            # WalkSAT: pick a violated constraint, flip a random variable in it
            # Find a violated constraint
            violated = []
            for (i, j, k) in constraints:
                if sol[i] and sol[j] and sol[k]:
                    violated.append((i, j, k))
            
            if not violated:
                # Shouldn't happen if cur_violations > 0, but just in case
                break
            
            (i, j, k) = rng.choice(violated)
            
            # Decide which variable to flip
            # Weighted random: prefer flipping variables that appear in many other violated constraints
            # For simplicity, pick randomly among the three
            choices = [i, j, k]
            idx = rng.choice(choices)
            sol[idx] = 1 - sol[idx]
        
        # Return the best solution found
        return [vals[i] for i in range(n) if best_sol[i]]
    
    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        subset = solve_sum_free(vals, seed=seed)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
