import random
import time
import numpy as np

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
    
    # --- Spectral Initialization via SVD of Incidence Matrix ---
    # Build signed incidence matrix A (m x n) where A[i,j] = +1 if literal +j+1 in clause i,
    # -1 if literal -j+1 in clause i, 0 otherwise.
    # For planted instances, the satisfying assignment should be correlated with the
    # principal right singular vector of A.
    try:
        A = np.zeros((m, n), dtype=np.float64)
        for i, c in enumerate(clauses):
            for l in c:
                var = abs(l) - 1
                A[i, var] = 1.0 if l > 0 else -1.0
        
        # SVD: A = U * S * Vt
        # The right singular vectors (rows of Vt) capture variable structure.
        # For planted 3-SAT, the top right singular vector often correlates with the solution.
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        # Use top few singular vectors to compute a soft assignment score
        # Weight by singular values to emphasize dominant structure
        if len(S) > 0 and S[0] > 1e-10:
            # Combine top k singular vectors weighted by their singular values
            k = min(5, len(S))
            weights = S[:k] / (S[:k].sum() + 1e-10)
            # Project onto variable space
            var_scores = Vt[:k].T @ weights  # shape (n,)
            
            # Derive initial assignment from sign pattern
            spectral_init = [var_scores[i] > 0 for i in range(n)]
        else:
            spectral_init = [rng.random() < 0.5 for _ in range(n)]
    except Exception:
        spectral_init = [rng.random() < 0.5 for _ in range(n)]

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
        
        # Track best state for potential recovery
        best_num_unsat = num_unsat
        best_assignment_copy = assignment[:]
        
        for iter_num in range(max_iters):
            if time.time() > deadline:
                break
            
            if num_unsat == 0:
                return assignment, 0
            
            # Periodically check if we've improved and update best
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment_copy = assignment[:]
            
            # Pick a random unsatisfied clause
            # To pick randomly, we could scan, but that's O(m).
            # For m ~ 200-500, scanning is fine.
            unsat_indices = [i for i in range(m) if clause_sat_count[i] == 0]
            if not unsat_indices:
                return assignment, 0
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

            if num_unsat == 0:
                return assignment, 0

        # Return best state found during this walk
        if best_num_unsat < num_unsat:
            return best_assignment_copy, best_num_unsat
        return assignment, num_unsat

    # Main Loop: Multiple restarts with spectral initialization bias
    best_assignment = [False] * n
    best_unsat = m # Start with all unsat
    
    # First, try the spectral initialization directly
    assignment_spec = spectral_init[:]
    assignment_spec, unsat_count_spec = walksat(assignment_spec, 50000)
    
    if unsat_count_spec == 0:
        return assignment_spec
    
    best_unsat = unsat_count_spec
    best_assignment = assignment_spec[:]
    
    # Then do random restarts, but bias initial assignments toward spectral solution
    max_restarts = 100
    iters_per_restart = 50000 # High, but we break on time
    
    for r in range(max_restarts):
        if time.time() > deadline - 1:
            break
        
        # Mix spectral init with random: with some probability, use spectral,
        # otherwise random, to explore both neighborhoods
        if r < 20:
            # First 20 restarts: use spectral init with slight perturbation
            assignment = [spectral_init[i] if rng.random() < 0.7 else (not spectral_init[i]) for i in range(n)]
        else:
            # Remaining restarts: pure random for diversity
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        assignment, unsat_count = walksat(assignment, iters_per_restart)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
    return best_assignment
