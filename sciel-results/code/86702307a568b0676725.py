"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Precompute clause literals as (var_index, polarity) for speed
    # var_index is 0-based, polarity is +1 or -1
    cl = []
    for c in clauses:
        cl.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Precompute variable to clauses mapping
    # var_to_clauses[var] = list of (clause_idx, polarity_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(cl):
        for var, pol in clause:
            var_to_clauses[var].append((i, pol))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s/330s limit
    
    # Best known assignment (by number of satisfied clauses)
    best_sat_count = -1
    best_assign = None
    
    rng = random.Random(0)
    
    # WalkSAT parameters
    max_restarts = 1000
    base_steps = 10000
    noise_schedule = [0.25, 0.5, 0.75, 0.35, 0.65, 0.45, 0.55, 0.3, 0.7, 0.4]
    
    def init_state(assign):
        """Initialize clause satisfaction counts and unsat list."""
        clause_sat_count = [0] * m
        unsat_indices = []
        
        for i in range(m):
            sat_count = 0
            for var, pol in cl[i]:
                if assign[var] == (pol > 0):
                    sat_count += 1
            clause_sat_count[i] = sat_count
            if sat_count == 0:
                unsat_indices.append(i)
                
        return clause_sat_count, unsat_indices

    def flip_variable(assign, var, target_val, clause_sat_count, unsat_indices):
        """Flip variable to target_val and update state."""
        old_val = assign[var]
        if old_val == target_val:
            return
        
        assign[var] = target_val
        
        for c_idx, pol in var_to_clauses[var]:
            # Check if the literal (var, pol) is satisfied by target_val
            new_lit_sat = (pol > 0) == target_val
            # Check if it was satisfied by old_val
            old_lit_sat = (pol > 0) == old_val
            
            if new_lit_sat != old_lit_sat:
                # Update clause satisfaction count
                if new_lit_sat:
                    clause_sat_count[c_idx] += 1
                else:
                    clause_sat_count[c_idx] -= 1
                
                # Update unsat_indices if status changed
                was_unsat = clause_sat_count[c_idx] == 0
                is_unsat = clause_sat_count[c_idx] == 0
                
                if was_unsat and not is_unsat:
                    # Became satisfied
                    # Remove from unsat_indices
                    # Linear search is O(m), but m is small.
                    # For efficiency, we can just mark and clean up later, or use a set.
                    # Given m ~ 100-500, linear search is okay.
                    try:
                        unsat_indices.remove(c_idx)
                    except ValueError:
                        pass
                elif not was_unsat and is_unsat:
                    # Became unsatisfied
                    unsat_indices.append(c_idx)
                
                # Note: if it was unsat and remains unsat, or sat and remains sat, no change to list.
                # Wait, if it was unsat (count 0) and we add a sat literal, count becomes 1. It leaves unsat list.
                # If it was sat (count >= 1) and we remove a sat literal:
                #   If count was 1, it becomes 0. It joins unsat list.
                #   If count was > 1, it stays sat.
                # The logic above handles this correctly by checking `was_unsat` (count==0) and `is_unsat` (count==0).
                # However, `was_unsat` is calculated AFTER updating count? No.
                # I need to check status BEFORE and AFTER.
                
                # Let's re-implement this part carefully.
                # I will revert and use a cleaner logic in the replacement.
                pass

    # Re-define flip_variable with correct logic
    def flip_variable(assign, var, target_val, clause_sat_count, unsat_indices):
        old_val = assign[var]
        if old_val == target_val:
            return
        
        assign[var] = target_val
        
        for c_idx, pol in var_to_clauses[var]:
            new_lit_sat = (pol > 0) == target_val
            old_lit_sat = (pol > 0) == old_val
            
            if new_lit_sat != old_lit_sat:
                old_count = clause_sat_count[c_idx]
                new_count = old_count + (1 if new_lit_sat else -1)
                clause_sat_count[c_idx] = new_count
                
                old_unsat = (old_count == 0)
                new_unsat = (new_count == 0)
                
                if old_unsat and not new_unsat:
                    # Removed from unsat
                    try:
                        unsat_indices.remove(c_idx)
                    except ValueError:
                        pass
                elif not old_unsat and new_unsat:
                    # Added to unsat
                    unsat_indices.append(c_idx)

    def walksat(assign, noise, max_steps, rng):
        """Run WalkSAT, return (satisfied, assign, sat_count)"""
        clause_sat_count, unsat_indices = init_state(assign)
        
        # Use a list for unsat_indices, but removal is O(N).
        # For better performance, we can use a set or just accept O(N) if N is small.
        # With 50 vars, m is likely < 500. O(500) removal is fast.
        
        for step in range(max_steps):
            if not unsat_indices:
                return True, assign, m
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = cl[ci]
            
            # Find best variable to flip
            # Heuristic: Minimize the number of clauses that become unsatisfied (conflicts)
            # Since we are flipping to satisfy `ci`, `ci` becomes satisfied.
            # We look at other clauses containing the variable.
            
            best_var = None
            best_conflicts = float('inf')
            best_pol = None
            
            for var, pol in clause:
                target_val = (pol > 0)
                
                # Calculate conflicts
                # A conflict occurs if a currently satisfied clause containing `var`
                # becomes unsatisfied because `var` was the only satisfied literal.
                conflicts = 0
                
                for c_idx, c_pol in var_to_clauses[var]:
                    if c_idx == ci:
                        continue
                    
                    # Check if this clause is currently satisfied
                    if clause_sat_count[c_idx] > 0:
                        # Check if `var` is the ONLY satisfied literal in this clause
                        # If so, flipping `var` to `target_val` might unsatisfy it
                        # unless `target_val` also satisfies the literal in `c_idx`
                        
                        # Does `target_val` satisfy the literal (var, c_pol)?
                        if (c_pol > 0) == target_val:
                            # It still satisfies it, so no conflict
                            pass
                        else:
                            # It unsatisfies the literal.
                            # If this was the only satisfied literal, the clause becomes unsatisfied.
                            if clause_sat_count[c_idx] == 1:
                                conflicts += 1
                    # If clause is currently unsatisfied, flipping var to satisfy it is good (reduces unsat), 
                    # but WalkSAT heuristic usually focuses on not breaking good clauses.
                    # Some variants count "gains" too. Let's stick to min-conflicts first.
                    
                # Tie-breaking: random
                if conflicts < best_conflicts or (conflicts == best_conflicts and rng.random() < 0.5):
                    best_conflicts = conflicts
                    best_var = var
                    best_pol = pol
            
            # With probability noise, flip a random variable in the clause
            if rng.random() < noise:
                var, pol = clause[rng.randrange(len(clause))]
            else:
                var = best_var
                pol = best_pol
            
            target_val = (pol > 0)
            flip_variable(assign, var, target_val, clause_sat_count, unsat_indices)
            
            # Periodic time check
            if step % 100 == 0 and time.time() > deadline:
                break
        
        # Calculate final sat count
        sat_count = sum(1 for c in range(m) if clause_sat_count[c] > 0)
        return False, assign, sat_count

    # Main loop with restarts
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Choose noise level
        noise = noise_schedule[restart % len(noise_schedule)]
        
        # Initialize with random assignment
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        solved, assign, sat_count = walksat(assign, noise, base_steps, rng)
        
        if sat_count > best_sat_count:
            best_sat_count = sat_count
            best_assign = list(assign)
        
        if solved:
            return assign
    
    # If we didn't solve, return the best assignment found
    if best_assign is not None:
        return best_assign
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
