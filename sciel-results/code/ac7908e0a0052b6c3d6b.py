<<<<<<< SEARCH
    def swap_improve(S, C_init, max_rounds=20, time_limit=None):
        """Try swapping one element out and two in, or one out and one in."""
        C = list(C_init)
        C_set = set(C)
        best_C = C
        best_len = len(C)
        excluded = [x for x in S if x not in C_set]
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            # Try removing each element and adding two excluded ones
            for i in range(len(C)):
                if time_limit is not None and time.time() > time_limit:
                    break
                C_no_i = [x for j, x in enumerate(C) if j != i]
                C_no_i_set = set(C_no_i)
                # Try adding pairs from excluded
                added_any = False
                for a in excluded:
                    if a in C_no_i_set:
                        continue
                    # Check if a can be added
                    valid_a = True
                    for b in C_no_i:
                        if (a - b) in C_no_i_set or (a + b) in C_no_i_set:
                            valid_a = False
                            break
                    if valid_a and (2 * a) in C_no_i_set:
                        valid_a = False
                    if not valid_a:
                        continue
                    new_C = C_no_i + [a]
                    new_C_set = C_no_i_set | {a}
                    # Try adding another
                    for b in excluded:
                        if b in new_C_set or b == a:
                            continue
                        valid_b = True
                        for c in new_C:
                            if (b - c) in new_C_set or (b + c) in new_C_set:
                                valid_b = False
                                break
                        if valid_b and (2 * b) in new_C_set:
                            valid_b = False
                        if valid_b:
                            candidate = new_C + [b]
                            if len(candidate) > best_len:
                                best_C = candidate
                                best_len = len(candidate)
                                improved = True
                                C = candidate
                                C_set = set(C)
                                excluded = [x for x in S if x not in C_set]
                                added_any = True
                                break
                    if added_any:
                        break
            if not improved:
                break
        return best_C
=======
    def walk_sat_improve(S, C_init, time_limit=None, max_steps=50000):
        """WalkSAT-style local search for sum-free subset.
        
        Variables: x_i = 1 if S[i] is in the subset, 0 otherwise.
        Clauses: For each triple (a, b, c) where a+b=c (a,b,c in S, a=b allowed),
                 NOT(x_a AND x_b AND x_c) must hold.
        
        We maximize sum(x_i) subject to all clauses being satisfied.
        """
        n = len(S)
        idx_map = {v: i for i, v in enumerate(S)}
        
        # Precompute all constraint triples: (i, j, k) where S[i]+S[j]=S[k]
        # For each pair (i,j), check if S[i]+S[j] is in S
        constraints = []  # List of (i, j, k) where i,j are addends, k is sum
        for i in range(n):
            for j in range(i, n):  # i=j allowed
                s = S[i] + S[j]
                if s in idx_map:
                    k = idx_map[s]
                    constraints.append((i, j, k))
        
        # Initialize assignment from C_init
        x = [0] * n
        for v in C_init:
            if v in idx_map:
                x[idx_map[v]] = 1
        
        # Count violations per variable and per clause
        # violation_count[i] = number of violated clauses involving variable i
        violation_count = [0] * n
        clause_violated = [False] * len(constraints)
        
        def update_clause(ci, i, j, k):
            """Update clause violation status and violation counts."""
            was_violated = clause_violated[ci]
            is_violated = x[i] and x[j] and x[k]
            clause_violated[ci] = is_violated
            if was_violated and not is_violated:
                violation_count[i] -= 1
                violation_count[j] -= 1
                violation_count[k] -= 1
            elif not was_violated and is_violated:
                violation_count[i] += 1
                violation_count[j] += 1
                violation_count[k] += 1
        
        # Initialize violation counts
        for ci, (i, j, k) in enumerate(constraints):
            if x[i] and x[j] and x[k]:
                clause_violated[ci] = True
                violation_count[i] += 1
                violation_count[j] += 1
                violation_count[k] += 1
        
        best_x = x[:]
        best_score = sum(x)
        
        rng_ws = random.Random(12345)
        total_violations = sum(1 for v in clause_violated if v)
        
        for step in range(max_steps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            if total_violations == 0:
                # Try to add more elements
                added_any = False
                for _ in range(10):  # Try up to 10 random additions
                    if time_limit is not None and time.time() > time_limit:
                        break
                    # Find variables with x=0 that can be set to 1 without creating violations
                    candidates = [i for i in range(n) if x[i] == 0 and violation_count[i] == 0]
                    if not candidates:
                        break
                    # Prefer candidates with fewer potential conflicts
                    # Just pick random for now
                    idx = rng_ws.choice(candidates)
                    x[idx] = 1
                    # Update clauses involving idx
                    for ci, (i, j, k) in enumerate(constraints):
                        if idx in (i, j, k):
                            update_clause(ci, i, j, k)
                    total_violations = sum(1 for v in clause_violated if v)
                    added_any = True
                    if sum(x) > best_score:
                        best_x = x[:]
                        best_score = sum(x)
                    if total_violations > 0:
                        # Revert
                        x[idx] = 0
                        for ci, (i, j, k) in enumerate(constraints):
                            if idx in (i, j, k):
                                update_clause(ci, i, j, k)
                        total_violations = sum(1 for v in clause_violated if v)
                        break
                    if not added_any:
                        break
                if sum(x) > best_score:
                    best_x = x[:]
                    best_score = sum(x)
                continue
            
            # Pick a violated clause
            violated_clauses = [ci for ci in range(len(constraints)) if clause_violated[ci]]
            if not violated_clauses:
                continue
            ci = rng_ws.choice(violated_clauses)
            i, j, k = constraints[ci]
            
            # Pick which variable to flip (weighted by violation count)
            # Higher violation count = more beneficial to flip
            weights = [violation_count[i], violation_count[j], violation_count[k]]
            total_w = sum(weights)
            if total_w == 0:
                continue
            r = rng_ws.random() * total_w
            cum = 0
            flip_var = i
            for var, w in zip([i, j, k], weights):
                cum += w
                if r <= cum:
                    flip_var = var
                    break
            
            # Flip the variable
            x[flip_var] = 1 - x[flip_var]
            
            # Update all clauses involving flip_var
            for ci2, (i2, j2, k2) in enumerate(constraints):
                if flip_var in (i2, j2, k2):
                    update_clause(ci2, i2, j2, k2)
            
            total_violations = sum(1 for v in clause_violated if v)
            
            # Update best if no violations and higher score
            if total_violations == 0:
                cur_score = sum(x)
                if cur_score > best_score:
                    best_x = x[:]
                    best_score = cur_score
        
        # Convert best_x back to set
        result = [S[i] for i in range(n) if best_x[i]]
        return result
>>>>>>> REPLACE

