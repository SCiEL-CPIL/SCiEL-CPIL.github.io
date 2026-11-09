import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Deadline: 325 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-process clauses for faster access and incremental updates
    # var_to_clauses: list of lists of clause indices
    # clause_literals: list of lists of (var_idx, polarity) where polarity is 1 or -1
    # Actually, let's keep it simple:
    # clause_vars[i] = [var_idx_1, var_idx_2, var_idx_3]
    # clause_pols[i] = [pol_1, pol_2, pol_3]
    
    clause_vars = []
    clause_pols = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        vars_in_c = []
        pols_in_c = []
        for l in c:
            var_idx = abs(l) - 1
            pol = 1 if l > 0 else -1
            vars_in_c.append(var_idx)
            pols_in_c.append(pol)
            var_to_clauses[var_idx].append(i)
        clause_vars.append(vars_in_c)
        clause_pols.append(pols_in_c)
        
    m = len(clauses)
    
    def init_state(assignment):
        """Calculate initial unsat counts for all clauses."""
        unsat_counts = [0] * m
        num_unsat = 0
        for i in range(m):
            count = 0
            for j in range(3): # Assuming 3-SAT, but let's be generic
                v = clause_vars[i][j]
                p = clause_pols[i][j]
                # If assignment[v] is True (1) and pol is -1, or assignment[v] is False (0) and pol is 1, it's unsat
                if (assignment[v] and p == -1) or (not assignment[v] and p == 1):
                    count += 1
            unsat_counts[i] = count
            if count == len(clause_vars[i]): # Clause is unsat
                num_unsat += 1
        return unsat_counts, num_unsat

    def get_unsat_clause_indices(unsat_counts):
        """Return indices of clauses that are currently unsatisfied."""
        return [i for i, c in enumerate(unsat_counts) if c == len(clause_vars[i])]

    # Parameters
    max_steps_per_run = 100000 # Can be much higher now
    noise = 0.05 # Standard WalkSAT noise
    
    rng = random.Random(0)
    
    # Main loop with restarts
    while time.time() < deadline:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_counts, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
            
        steps = 0
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                return assignment
                
            if num_unsat == 0:
                return assignment
                
            # Get unsatisfied clauses
            # Optimization: We can just pick a random index and check if it's unsat,
            # but for small n, building the list is fine. 
            # Better: Maintain a list of unsat clause indices? 
            # For now, just filter. m is small.
            unsat_indices = [i for i in range(m) if unsat_counts[i] == len(clause_vars[i])]
            
            if not unsat_indices:
                return assignment

            # Pick a random unsat clause
            clause_idx = rng.choice(unsat_indices)
            
            # Heuristic: Pick the literal that minimizes the increase in unsat clauses
            # i.e., maximizes the number of newly satisfied clauses.
            best_var = -1
            best_delta = -1000000 # We want to maximize (newly_sat - newly_unsat)
            # Actually, standard WalkSAT minimizes the number of unsat clauses after the flip.
            # Let's calculate the "cost" of flipping each variable in the clause.
            
            best_cost = float('inf')
            best_var_to_flip = -1
            
            for j in range(len(clause_vars[clause_idx])):
                v = clause_vars[clause_idx][j]
                p = clause_pols[clause_idx][j]
                
                # Simulate flip of v
                # We need to update unsat_counts for all clauses containing v
                # and count the change in num_unsat.
                
                # Store original values to revert
                # Since we only flip one variable, we can just iterate through var_to_clauses[v]
                
                delta_unsat = 0
                for c_idx in var_to_clauses[v]:
                    # Current count
                    old_count = unsat_counts[c_idx]
                    
                    # Determine if this variable currently satisfies or unsatisfies this clause
                    # Polarity in this clause c_idx might be different from clause_idx
                    c_pols = clause_pols[c_idx]
                    c_vars = clause_vars[c_idx]
                    
                    # Find position of v in c_idx? 
                    # This is O(k) lookup. Better to precompute?
                    # For n=50, k=3, it's fast enough.
                    # Let's find if v is in c_vars.
                    # Actually, we know v is in c_idx because c_idx is in var_to_clauses[v].
                    
                    # Calculate new count
                    # If v currently satisfies the clause (contributes to sat), flipping it makes it unsat (if it was the only one sat? no, just adds to unsat count).
                    # Let's just recalculate the contribution of v.
                    
                    # Current contribution of v to c_idx unsat count:
                    # If (assignment[v] and c_pol == -1) or (not assignment[v] and c_pol == 1), it is UNSAT (contributes 1).
                    # Else it is SAT (contributes 0).
                    
                    # We need the polarity of v in c_idx.
                    # Let's find it.
                    v_pos_in_c = -1
                    for k, cv in enumerate(c_vars):
                        if cv == v:
                            v_pos_in_c = k
                            break
                    if v_pos_in_c == -1:
                        continue # Should not happen
                    
                    c_pol = c_pols[v_pos_in_c]
                    
                    is_unsat_currently = (assignment[v] and c_pol == -1) or (not assignment[v] and c_pol == 1)
                    # After flip, assignment[v] becomes not assignment[v]
                    is_unsat_after = (not assignment[v] and c_pol == -1) or (assignment[v] and c_pol == 1)
                    
                    # If it was unsat (1) and becomes sat (0), count decreases by 1.
                    # If it was sat (0) and becomes unsat (1), count increases by 1.
                    
                    if is_unsat_currently and not is_unsat_after:
                        delta_unsat -= 1
                        # Check if this clause becomes fully satisfied
                        if old_count == 1:
                            # It becomes 0, so num_unsat decreases by 1
                            # But we are counting delta in unsat_counts, not num_unsat directly yet.
                            # Let's stick to delta in unsat_counts sum? No, we care about num_unsat.
                            # It is easier to just calculate the new num_unsat.
                            pass
                    
                    # This manual delta tracking is error-prone. 
                    # Let's just apply the flip to a copy of unsat_counts? No, too slow.
                    # Let's just calculate the new num_unsat directly for this variable flip.
                    
                # Alternative: Just calculate the new num_unsat for the flip of v.
                # This is O(degree(v)).
                
                new_num_unsat = 0
                # We only need to check clauses containing v? No, other clauses are unaffected.
                # So new_num_unsat = old_num_unsat - (clauses containing v that become sat) + (clauses containing v that become unsat)
                
                becomes_sat = 0
                becomes_unsat = 0
                
                for c_idx in var_to_clauses[v]:
                    old_count = unsat_counts[c_idx]
                    
                    # Find polarity of v in c_idx
                    c_vars = clause_vars[c_idx]
                    c_pols = clause_pols[c_idx]
                    v_pos_in_c = -1
                    for k, cv in enumerate(c_vars):
                        if cv == v:
                            v_pos_in_c = k
                            break
                    if v_pos_in_c == -1:
                        continue
                    
                    c_pol = c_pols[v_pos_in_c]
                    
                    # Current status of v in c_idx
                    is_unsat_currently = (assignment[v] and c_pol == -1) or (not assignment[v] and c_pol == 1)
                    
                    # New status of v in c_idx
                    is_unsat_after = (not assignment[v] and c_pol == -1) or (assignment[v] and c_pol == 1)
                    
                    if is_unsat_currently and not is_unsat_after:
                        # Variable v was unsat, becomes sat.
                        # Clause unsat count decreases by 1.
                        new_count = old_count - 1
                        if new_count == 0:
                            becomes_sat += 1
                        # If old_count was > 1, it's still unsat, just less unsat.
                        
                    elif not is_unsat_currently and is_unsat_after:
                        # Variable v was sat, becomes unsat.
                        # Clause unsat count increases by 1.
                        new_count = old_count + 1
                        if old_count == len(clause_vars[c_idx]) - 1:
                            # It was 1 unsat literal away from being unsat? No.
                            # If old_count == len-1, then new_count == len.
                            # So it becomes unsat.
                            becomes_unsat += 1
                        # If old_count was 0, it becomes 1, still sat.
                        # If old_count was k, it becomes k+1.
                        # It only becomes "unsat" (fully) if new_count == len.
                        
                    # Note: If old_count == len (clause was already unsat), and we change a literal:
                    # If we make one more literal unsat? Impossible, max is len.
                    # If we make one literal sat (old_count len -> len-1), it's still unsat.
                    # If we make one literal unsat (old_count len -> len+1)? Impossible.
                    # Wait, if clause is unsat (count=len), all literals are unsat.
                    # If we flip one, it becomes sat. Count becomes len-1. Still unsat.
                    # So no change to num_unsat.
                    
                delta_num_unsat = becomes_unsat - becomes_sat
                cost = num_unsat + delta_num_unsat
                
                if cost < best_cost:
                    best_cost = cost
                    best_var_to_flip = v
            
            # Apply the best flip
            if best_var_to_flip != -1:
                v = best_var_to_flip
                assignment[v] = not assignment[v]
                
                # Update unsat_counts and num_unsat
                for c_idx in var_to_clauses[v]:
                    c_vars = clause_vars[c_idx]
                    c_pols = clause_pols[c_idx]
                    v_pos_in_c = -1
                    for k, cv in enumerate(c_vars):
                        if cv == v:
                            v_pos_in_c = k
                            break
                    if v_pos_in_c == -1:
                        continue
                    c_pol = c_pols[v_pos_in_c]
                    
                    # Recalculate contribution? No, we know the change.
                    # Let's just recalculate the whole clause count for simplicity and safety?
                    # No, that's O(k) which is 3. It's fine.
                    # But we can do it incrementally.
                    
                    # Old contribution
                    old_is_unsat = (assignment[v] and c_pol == -1) or (not assignment[v] and c_pol == 1)
                    # Wait, assignment[v] is already flipped in the line above!
                    # So we need to be careful.
                    
                    # Let's just recalculate the count for this clause from scratch.
                    # It's only 3 literals.
                    new_count = 0
                    for kk in range(len(c_vars)):
                        vv = c_vars[kk]
                        pp = c_pols[kk]
                        if (assignment[vv] and pp == -1) or (not assignment[vv] and pp == 1):
                            new_count += 1
                    old_count = unsat_counts[c_idx]
                    unsat_counts[c_idx] = new_count
                    
                    if old_count == len(c_vars) and new_count < len(c_vars):
                        num_unsat -= 1
                    elif old_count < len(c_vars) and new_count == len(c_vars):
                        num_unsat += 1
            
            # Noise
            if rng.random() < noise:
                # Pick random literal from the clause
                j = rng.randint(0, len(clause_vars[clause_idx]) - 1)
                v = clause_vars[clause_idx][j]
                assignment[v] = not assignment[v]
                
                # Update unsat_counts and num_unsat
                for c_idx in var_to_clauses[v]:
                    c_vars = clause_vars[c_idx]
                    c_pols = clause_pols[c_idx]
                    
                    new_count = 0
                    for kk in range(len(c_vars)):
                        vv = c_vars[kk]
                        pp = c_pols[kk]
                        if (assignment[vv] and pp == -1) or (not assignment[vv] and pp == 1):
                            new_count += 1
                    old_count = unsat_counts[c_idx]
                    unsat_counts[c_idx] = new_count
                    
                    if old_count == len(c_vars) and new_count < len(c_vars):
                        num_unsat -= 1
                    elif old_count < len(c_vars) and new_count == len(c_vars):
                        num_unsat += 1
            
            steps += 1

    return assignment
