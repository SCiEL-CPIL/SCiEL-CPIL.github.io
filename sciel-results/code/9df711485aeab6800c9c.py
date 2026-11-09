import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n

    deadline = time.time() + 290  # Safety margin
    rng = random.Random(42)
    
    # Pre-process: Build incidence lists and literal mappings
    # var_to_clauses[var] = list of (clause_idx, literal_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            var = abs(l) - 1
            var_to_clauses[var].append((i, l))
            
    # State
    assignment = [False] * n
    clause_true_count = [0] * num_clauses
    clause_satisfied = [False] * num_clauses
    satisfied_count = 0
    
    def update_clause_status(clause_idx, var, old_val, new_val):
        """Update clause status when a variable flips. Returns delta in satisfied_count."""
        nonlocal satisfied_count
        # Find the literal in this clause corresponding to var
        # We stored (clause_idx, literal) in var_to_clauses, but here we are iterating 
        # over var_to_clauses[var], so we have the literal.
        # Wait, the function signature above is slightly different. 
        # Let's refactor to pass the literal directly.
        pass

    # Better update logic
    def flip_var(var, new_val):
        """Flip variable 'var' to 'new_val'. Updates all relevant clauses."""
        nonlocal satisfied_count
        old_val = assignment[var]
        if old_val == new_val:
            return 0
        
        assignment[var] = new_val
        delta = 0
        
        for clause_idx, lit in var_to_clauses[var]:
            # Determine if this literal was true before and is true now
            lit_val_before = (lit > 0) == old_val
            lit_val_after = (lit > 0) == new_val
            
            if lit_val_before and not lit_val_after:
                # Literal became false
                clause_true_count[clause_idx] -= 1
                if clause_true_count[clause_idx] == 0:
                    # Clause became unsatisfied
                    clause_satisfied[clause_idx] = False
                    satisfied_count -= 1
                    delta -= 1
            elif not lit_val_before and lit_val_after:
                # Literal became true
                clause_true_count[clause_idx] += 1
                if clause_true_count[clause_idx] == 1:
                    # Clause became satisfied
                    clause_satisfied[clause_idx] = True
                    satisfied_count += 1
                    delta += 1
            # If lit_val_before and lit_val_after: count unchanged, status unchanged
            # If not lit_val_before and not lit_val_after: count unchanged, status unchanged
            
        return delta

    def init_state(assignment_init):
        nonlocal satisfied_count
        assignment = assignment_init[:]
        clause_true_count = [0] * num_clauses
        clause_satisfied = [False] * num_clauses
        satisfied_count = 0
        
        for var in range(n):
            val = assignment[var]
            for clause_idx, lit in var_to_clauses[var]:
                if (lit > 0) == val:
                    clause_true_count[clause_idx] += 1
        
        for i in range(num_clauses):
            if clause_true_count[i] > 0:
                clause_satisfied[i] = True
                satisfied_count += 1
                
        return assignment, clause_true_count, clause_satisfied, satisfied_count

    best_assignment = [False] * n
    best_count = -1
    
    max_restarts = 200
    restart_threshold = 1000 # Iterations without improvement
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, clause_true_count, clause_satisfied, satisfied_count = init_state(assignment)
        cur_count = satisfied_count
        
        if cur_count > best_count:
            best_count = cur_count
            best_assignment = assignment[:]
            
        if cur_count == num_clauses:
            return assignment
        
        no_improve = 0
        iter_count = 0
        max_iters_per_restart = 10000
        
        while iter_count < max_iters_per_restart:
            if time.time() > deadline - 5:
                return best_assignment
            
            # Find unsatisfied clauses
            # Optimization: Maintain a list of unsatisfied clauses? 
            # Or just pick a random index and check? 
            # Given N=50, M~200-300, picking random and checking is fast.
            # But finding a specific unsat clause is better.
            # Let's just pick a random clause index and if it's satisfied, try another.
            # Actually, maintaining a list of unsat clauses is better for O(1) access.
            
            # For simplicity and speed in Python, let's just pick a random clause.
            # If it's satisfied, we can't flip based on it. 
            # WalkSAT picks an UNSAT clause.
            
            # Let's find an unsat clause.
            # Since we have clause_satisfied array, we can pick a random index 
            # and check. If satisfied, pick again. 
            # With high satisfaction rate, this might be slow.
            # Better: Keep a list of unsat clauses.
            
            # Let's rebuild unsat list periodically or maintain it.
            # Maintaining it is complex with updates. 
            # Given small N, let's just iterate or pick random few times.
            
            # Strategy: Pick 5 random clauses, if any unsat, use it. Else, scan.
            unsat_idx = -1
            for _ in range(5):
                idx = rng.randrange(num_clauses)
                if not clause_satisfied[idx]:
                    unsat_idx = idx
                    break
            
            if unsat_idx == -1:
                # Scan
                for i in range(num_clauses):
                    if not clause_satisfied[i]:
                        unsat_idx = i
                        break
            
            if unsat_idx == -1:
                return assignment # Should not happen if cur_count < num_clauses
            
            clause = clauses[unsat_idx]
            
            # Score each literal in this clause
            # We want to flip a literal such that it satisfies this clause (obviously)
            # and minimizes the damage to other clauses.
            # Score = - (number of other clauses that would become unsatisfied)
            # Or: Gain = (clauses that become satisfied) - (clauses that become unsatisfied)
            # Since the chosen clause is unsat, flipping any literal in it will satisfy it.
            # So Gain is always +1 for this clause.
            # We just need to minimize the loss to other clauses.
            
            best_score = -1000
            best_lit = None
            
            for lit in clause:
                var = abs(lit) - 1
                new_val = (lit > 0)
                
                # Calculate cost of flipping var to new_val
                # Cost = number of clauses that are currently satisfied and would become unsatisfied
                cost = 0
                for c_idx, l in var_to_clauses[var]:
                    if c_idx == unsat_idx:
                        continue # We know this one becomes satisfied
                    if clause_satisfied[c_idx]:
                        # Check if this clause relies on 'var' being in its current state
                        # i.e., if the literal 'l' in this clause is currently true
                        # and flipping var would make it false
                        # AND no other literal in this clause is true
                        # Actually, if clause_satisfied[c_idx] is True, it has at least one true literal.
                        # If we flip var, and 'l' was true, it becomes false.
                        # The clause remains satisfied only if another literal is true.
                        # So we need to check if 'l' was the ONLY true literal.
                        # We have clause_true_count[c_idx].
                        # If clause_true_count[c_idx] == 1 and 'l' was true, then it becomes unsat.
                        
                        lit_val_before = (l > 0) == assignment[var]
                        if lit_val_before:
                            if clause_true_count[c_idx] == 1:
                                cost += 1
                            # If clause_true_count > 1, it stays satisfied even if l becomes false.
                
                score = -cost # Maximize score (minimize cost)
                # Add small random tie-breaker
                score += rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            # Flip the best literal
            if best_lit is not None:
                var = abs(best_lit) - 1
                new_val = (best_lit > 0)
                
                # Perform flip
                # We need to update the state. 
                # My flip_var function updates nonlocal variables.
                # But I defined flip_var to use nonlocal satisfied_count etc.
                # I need to make sure the state variables are accessible.
                
                # Let's inline the flip logic here to avoid nonlocal issues in nested scope 
                # if I redefined them in init_state.
                # Wait, init_state returns the lists. I should use those lists directly.
                
                old_val = assignment[var]
                if old_val == new_val:
                    # No change needed? But we picked a lit from unsat clause, so it must be false.
                    # So old_val != new_val is guaranteed.
                    pass
                else:
                    assignment[var] = new_val
                    delta = 0
                    for c_idx, l in var_to_clauses[var]:
                        lit_val_before = (l > 0) == old_val
                        lit_val_after = (l > 0) == new_val
                        
                        if lit_val_before and not lit_val_after:
                            clause_true_count[c_idx] -= 1
                            if clause_true_count[c_idx] == 0:
                                clause_satisfied[c_idx] = False
                                satisfied_count -= 1
                                delta -= 1
                        elif not lit_val_before and lit_val_after:
                            clause_true_count[c_idx] += 1
                            if clause_true_count[c_idx] == 1:
                                clause_satisfied[c_idx] = True
                                satisfied_count += 1
                                delta += 1
                            
                    cur_count = satisfied_count
                    
                    if cur_count > best_count:
                        best_count = cur_count
                        best_assignment = assignment[:]
                        
                    if cur_count == num_clauses:
                        return assignment
                        
                    if cur_count > cur_count - delta: # Improved
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    if no_improve >= restart_threshold:
                        break
                        # break out of while
                    iter_count += 1
        
    return best_assignment
