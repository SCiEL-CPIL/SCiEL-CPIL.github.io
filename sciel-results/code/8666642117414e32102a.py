import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Preprocessing: Convert to 0-based indexing and store structure
    # clause_literals[i] = list of (var, is_positive)
    # var_clauses[var] = list of (clause_idx, is_positive)
    
    clause_literals = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        lits = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            lits.append((var, pos))
            var_clauses[var].append((i, pos))
        clause_literals.append(lits)

    deadline = time.time() + 285
    rng = random.Random(42)

    # --- DPLL with Unit Propagation (Limited) ---
    # This handles easy instances very quickly.
    
    def dpll(assignment, depth=0):
        if time.time() > deadline - 10:
            return None
            
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                # Check if clause i is unit or falsified
                unassigned = []
                sat = False
                for var, pos in clause_literals[i]:
                    if assignment[var] is None:
                        unassigned.append((var, pos))
                    else:
                        if assignment[var] == pos:
                            sat = True
                            break
                
                if sat:
                    continue
                    
                if len(unassigned) == 0:
                    return None # Falsified
                elif len(unassigned) == 1:
                    var, pos = unassigned[0]
                    assignment[var] = pos
                    changed = True
        
        # Check if all satisfied
        all_sat = True
        for i in range(m):
            sat = False
            for var, pos in clause_literals[i]:
                if assignment[var] is not None and assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        
        if all_sat:
            # Fill remaining with False (or random)
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = False
            return assignment[:]

        # If no unassigned variables left but not all sat (should be caught by falsified above)
        # Find an unassigned variable
        branch_var = None
        for v in range(n):
            if assignment[v] is None:
                branch_var = v
                break
        
        if branch_var is None:
            return None # Should have been caught

        # Branch
        # Try True
        assignment[branch_var] = True
        res = dpll(assignment, depth + 1)
        if res is not None:
            return res
            
        # Try False
        assignment[branch_var] = False
        res = dpll(assignment, depth + 1)
        if res is not None:
            return res
            
        assignment[branch_var] = None
        return None

    # Run DPLL with a small depth limit to avoid exponential blowup on hard instances
    # 50 vars, depth 50 is too much. Let's limit depth or time.
    # Actually, for 50 vars, DPLL might be fine if the formula is easy.
    # But to be safe, we'll run it with a time check inside.
    
    initial_assignment = [None] * n
    dpll_start = time.time()
    # We don't want to spend all time on DPLL if it's hard.
    # Let's run it, but the time check inside dpll will stop it.
    # However, dpll as written doesn't check time on every node effectively if recursion is deep.
    # Let's add a global time check.
    
    # A simpler approach: Just run WalkSAT with optimized state.
    # WalkSAT is very robust for 50 vars.
    # Let's stick to the optimized WalkSAT which is faster than the original.

    def walksat(assignment, max_iters):
        # Initialize state
        # clause_sat_count[i] = number of satisfied literals in clause i
        clause_sat_count = [0] * m
        num_unsat = 0
        
        for i in range(m):
            count = 0
            for var, pos in clause_literals[i]:
                if assignment[var] == pos:
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                num_unsat += 1
        
        if num_unsat == 0:
            return assignment, 0
        
        p = 0.5 # Probability of greedy choice
        no_improve = 0
        restart_limit = 5000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            if num_unsat == 0:
                return assignment, 0

            # Pick a random unsatisfied clause
            # To pick randomly, we could scan, but that's O(m).
            # For m ~ 200-500, scanning is fine.
            unsat_indices = [i for i in range(m) if clause_sat_count[i] == 0]
            clause_idx = rng.choice(unsat_indices)
            
            # Score each literal in the clause
            # Score = number of other clauses that would be satisfied (gain) - number that would be unsatisfied (loss)
            # Note: The clause itself is unsatisfied, so flipping a literal in it will satisfy it (gain 1 for this clause, but we track num_unsat separately).
            # Actually, if we flip a literal in an unsat clause, that clause becomes sat (num_unsat decreases by 1).
            # We need to evaluate the impact on *other* clauses.
            
            scores = []
            best_score = -1000
            best_idx = -1
            
            for k, (var, pos) in enumerate(clause_literals[clause_idx]):
                score = 0
                # Check impact on other clauses containing this var
                for ci, polarity in var_clauses[var]:
                    if ci == clause_idx:
                        continue
                    
                    # Current state of clause ci
                    cur_sat = clause_sat_count[ci] > 0
                    
                    # New state of clause ci if we set var to pos
                    # If polarity == pos, the literal in ci is satisfied by this flip.
                    # If polarity != pos, the literal in ci is falsified by this flip.
                    
                    if polarity == pos:
                        # This literal becomes True.
                        # If cur_sat was False, it becomes True (gain 1).
                        # If cur_sat was True, it stays True (no change in sat status, but count increases).
                        # We only care about the 0->1 transition for num_unsat.
                        if not cur_sat:
                            score += 1
                    else:
                        # This literal becomes False.
                        # If cur_sat was True, we need to check if it was the *only* satisfied literal.
                        # If clause_sat_count[ci] == 1, and this is the only one, it becomes unsat (loss 1).
                        if cur_sat and clause_sat_count[ci] == 1:
                            score -= 1
                            
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_idx = k
            
            # Decide which literal to flip
            if rng.random() < p:
                chosen_idx = best_idx
            else:
                chosen_idx = rng.randrange(len(clause_literals[clause_idx]))
            
            var, pos = clause_literals[clause_idx][chosen_idx]
            
            # Apply flip
            old_val = assignment[var]
            assignment[var] = pos
            
            # Update state
            # For each clause containing var:
            for ci, polarity in var_clauses[var]:
                # Determine if the literal in ci was satisfied before and after
                was_sat = (old_val == polarity)
                is_sat = (pos == polarity)
                
                if was_sat and not is_sat:
                    clause_sat_count[ci] -= 1
                    if clause_sat_count[ci] == 0:
                        num_unsat += 1
                elif not was_sat and is_sat:
                    clause_sat_count[ci] += 1
                    if clause_sat_count[ci] == 1:
                        num_unsat -= 1

            # Check improvement
            # We don't track a global "satisfied count" easily, but we track num_unsat.
            # If num_unsat decreased, it's an improvement.
            # The original code tracked total satisfied. Here we track unsat.
            # Let's just use num_unsat as the metric.
            
            # Simple restart logic based on time or steps
            # The original used no_improve. We can approximate by checking if num_unsat is stuck.
            # But with random walks, it's hard to define "improve" strictly.
            # Let's just run until time limit or success.
            
            if num_unsat == 0:
                return assignment, 0

        return assignment, num_unsat

    # Main Loop: Multiple restarts
    best_assignment = [False] * n
    best_unsat = m # Start with all unsat
    
    # We can try a few different p values or just stick to 0.5
    # Let's run many restarts
    max_restarts = 100
    iters_per_restart = 50000 # High, but we break on time
    
    for r in range(max_restarts):
        if time.time() > deadline - 1:
            break
        
        # Random init
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        assignment, unsat_count = walksat(assignment, iters_per_restart)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
    return best_assignment
