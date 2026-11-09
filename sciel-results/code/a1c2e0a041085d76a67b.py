"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause structure
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            cd.append((var, pos))
        clause_data.append(cd)
    
    deadline = time.time() + 285  # Safety margin (330s total, leave 45s buffer)
    
    rng = random.Random(42)
    
    # Build variable-to-clauses index for fast scoring
    var_to_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var, pos in cd:
            var_to_clauses[var].append((i, pos))
    
    def is_clause_sat(ci, assignment):
        for var, pos in clause_data[ci]:
            if assignment[var] == pos:
                return True
        return False
    
    def count_satisfied(assignment):
        count = 0
        for i in range(num_clauses):
            if is_clause_sat(i, assignment):
                count += 1
        return count
    
    def dpll_solve(assignment, deadline):
        """
        DPLL with unit propagation and pure literal elimination.
        Returns (success, assignment) or (None, None) if timeout.
        assignment is a list of bool or None (unassigned).
        """
        # Check for timeout
        if time.time() > deadline:
            return None, None
        
        # Check if all clauses satisfied
        all_sat = True
        for ci in range(num_clauses):
            sat = False
            for var, pos in clause_data[ci]:
                if assignment[var] is None:
                    continue
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        
        if all_sat:
            # Fill remaining unassigned vars
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = rng.random() < 0.5
            return True, assignment
        
        # Unit propagation
        changed = True
        while changed:
            if time.time() > deadline:
                return None, None
            changed = False
            for ci in range(num_clauses):
                # Check if clause is already satisfied
                sat = False
                unassigned = []
                for var, pos in clause_data[ci]:
                    if assignment[var] is None:
                        unassigned.append((var, pos))
                    elif assignment[var] == pos:
                        sat = True
                        break
                if sat:
                    continue
                # If exactly one unassigned literal, it must be true
                if len(unassigned) == 1:
                    var, pos = unassigned[0]
                    if assignment[var] is not None and assignment[var] != pos:
                        return False, None  # Contradiction
                    assignment[var] = pos
                    changed = True
                elif len(unassigned) == 0:
                    return False, None  # Contradiction
        
        if time.time() > deadline:
            return None, None
        
        # Pure literal elimination
        changed = True
        while changed:
            if time.time() > deadline:
                return None, None
            changed = False
            for var in range(n):
                if assignment[var] is not None:
                    continue
                # Check if var appears with only one polarity in unsatisfied clauses
                pos_in_unsat = False
                neg_in_unsat = False
                for ci, pos in var_to_clauses[var]:
                    # Check if this clause is already satisfied
                    sat = False
                    for v2, p2 in clause_data[ci]:
                        if assignment[v2] is not None and assignment[v2] == p2:
                            sat = True
                            break
                    if sat:
                        continue
                    if pos:
                        pos_in_unsat = True
                    else:
                        neg_in_unsat = True
                    if pos_in_unsat and neg_in_unsat:
                        break
                if pos_in_unsat and not neg_in_unsat:
                    assignment[var] = True
                    changed = True
                elif neg_in_unsat and not pos_in_unsat:
                    assignment[var] = False
                    changed = True
        
        if time.time() > deadline:
            return None, None
        
        # Check again if all satisfied
        all_sat = True
        for ci in range(num_clauses):
            sat = False
            for var, pos in clause_data[ci]:
                if assignment[var] is None:
                    continue
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        
        if all_sat:
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = rng.random() < 0.5
            return True, assignment
        
        # Find a variable to branch on (choose one in an unsatisfied clause)
        branch_var = None
        branch_pos = None
        for ci in range(num_clauses):
            sat = False
            for var, pos in clause_data[ci]:
                if assignment[var] is None:
                    if not sat:
                        branch_var = var
                        branch_pos = pos
                elif assignment[var] == pos:
                    sat = True
                    break
            if branch_var is not None:
                break
        
        if branch_var is None:
            # All variables assigned but not all clauses satisfied
            return False, None
        
        # Try both branches
        for val in [True, False]:
            if time.time() > deadline:
                return None, None
            assignment[branch_var] = val
            success, result = dpll_solve(assignment, deadline)
            if success:
                return True, result
            if result is not None:
                # Contradiction, backtrack
                assignment[branch_var] = None
            else:
                # Timeout
                assignment[branch_var] = None
                return None, None
        
        assignment[branch_var] = None
        return False, None
    
    def walk_sat_trial(assignment, max_iters, deadline):
        """
        Improved WalkSAT with adaptive noise and better scoring.
        """
        cur_count = count_satisfied(assignment)
        
        if cur_count == num_clauses:
            return assignment, cur_count
        
        no_improve = 0
        restart_threshold = 500
        noise_param = 0.5  # Adaptive noise parameter
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(num_clauses):
                if not is_clause_sat(i, assignment):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment, num_clauses
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clause_data[clause_idx]
            
            # Score each literal in the clause
            scores = []
            for var, pos in clause:
                # Count how many currently-unsatisfied clauses would be satisfied
                # if we set var = pos
                gain = 0
                # Also count how many currently-satisfied clauses would be broken
                loss = 0
                for ci, cpos in var_to_clauses[var]:
                    if cpos == pos:
                        # Setting var=pos satisfies this literal in clause ci
                        if not is_clause_sat(ci, assignment):
                            gain += 1
                    else:
                        # Setting var=pos makes this literal false; if it was the only
                        # true literal in a satisfied clause, that clause becomes unsat
                        if is_clause_sat(ci, assignment):
                            # Check if this literal was the only true one
                            only_true = True
                            for v2, p2 in clause_data[ci]:
                                if v2 == var:
                                    continue
                                if assignment[v2] == p2:
                                    only_true = False
                                    break
                            if only_true:
                                loss += 1
                scores.append(gain - loss)
            
            # Adaptive noise: increase noise when stuck
            if no_improve > restart_threshold // 2:
                noise_param = min(1.0, noise_param + 0.05)
            else:
                noise_param = max(0.1, noise_param - 0.02)
            
            # Select literal with probability proportional to exp(score / temperature)
            # Use soft-max style selection
            temp = max(0.1, noise_param)
            max_score = max(scores)
            weights = []
            for s in scores:
                # Soft-max weight
                w = 1.0 + (s - max_score) / temp if s >= max_score else 0.1
                weights.append(max(0.01, w))
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            var, pos = clause[chosen_idx]
            assignment[var] = pos
            
            new_count = count_satisfied(assignment)
            if new_count > cur_count:
                cur_count = new_count
                no_improve = 0
                if cur_count == num_clauses:
                    return assignment, cur_count
            else:
                no_improve += 1
            
            if no_improve >= restart_threshold:
                break
        
        return assignment, cur_count
    
    # First, try DPLL with a time budget
    dpll_budget = 15  # seconds for DPLL (reduced to leave more time for spectral init + WalkSAT)
    dpll_deadline = time.time() + dpll_budget
    
    dpll_assignment = [None] * n
    success, result = dpll_solve(dpll_assignment, dpll_deadline)
    
    if success:
        # Convert to list of booleans
        return [bool(x) for x in result]
    
    # DPLL failed or timed out; fall back to WalkSAT
    # Use remaining time for WalkSAT
    walk_deadline = deadline
    
    # --- Spectral initialization ---
    # Build clause-variable incidence matrix M (num_clauses x n)
    # M[i, v] = +1 if clause i contains positive literal v
    #           -1 if clause i contains negative literal v
    #            0 otherwise
    M = np.zeros((num_clauses, n), dtype=np.float64)
    for ci, cd in enumerate(clause_data):
        for var, pos in cd:
            M[ci, var] = 1.0 if pos else -1.0
    
    # Compute SVD to find low-rank structure
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Use the first few right singular vectors (rows of Vt) to project
        # onto variable space. The sign pattern of the dominant right singular
        # vector gives a candidate assignment aligned with the planted solution.
        num_components = min(5, n, len(s))
        
        # Weight components by singular values (larger singular values = more important)
        weighted_V = Vt[:num_components, :] * s[:num_components, np.newaxis]
        
        # Project: for each variable, compute the weighted sum of its projections
        # across the top singular vectors
        var_scores = np.sum(weighted_V, axis=0)
        
        # Derive initial assignment from sign pattern
        spectral_assignment = [var_scores[v] > 0 for v in range(n)]
        
        # Evaluate spectral assignment
        spectral_count = count_satisfied(spectral_assignment)
        
    except Exception:
        spectral_assignment = [False] * n
        spectral_count = count_satisfied(spectral_assignment)
    
    best_assignment = spectral_assignment[:]
    best_count = spectral_count
    
    # If DPLL made partial progress and it's better, use that instead
    if any(x is not None for x in dpll_assignment):
        # Fill in unassigned variables randomly
        for v in range(n):
            if dpll_assignment[v] is None:
                dpll_assignment[v] = rng.random() < 0.5
        dpll_filled = [bool(x) for x in dpll_assignment]
        dpll_count = count_satisfied(dpll_filled)
        if dpll_count > best_count:
            best_assignment = dpll_filled
            best_count = dpll_count
    
    # Multiple WalkSAT restarts
    max_restarts = 300
    iters_per_restart = 2500
    
    for restart in range(max_restarts):
        if time.time() > walk_deadline - 5:
            break
        
        # Choose initialization strategy:
        # - 30%: perturb best solution
        # - 30%: use spectral initialization (possibly with small perturbation)
        # - 40%: random
        r_init = rng.random()
        if best_count >= 0 and r_init < 0.3:
            # Perturb best solution
            assignment = best_assignment[:]
            num_flips = rng.randint(1, max(1, n // 10))
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        elif r_init < 0.6:
            # Use spectral initialization with optional small perturbation
            assignment = spectral_assignment[:]
            if rng.random() < 0.5:
                num_flips = rng.randint(1, max(1, n // 15))
                for _ in range(num_flips):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        assignment, cur_count = walk_sat_trial(assignment, iters_per_restart, walk_deadline)
        
        if cur_count > best_count:
            best_count = cur_count
            best_assignment = assignment[:]
        
        if best_count == num_clauses:
            return best_assignment
    
    return best_assignment
# EVOLVE-BLOCK-END
