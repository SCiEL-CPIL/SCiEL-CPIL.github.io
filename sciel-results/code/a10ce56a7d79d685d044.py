"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: leave 5s safety margin from 330s limit
    deadline = time.time() + 325.0
    
    # Precompute clause indices for faster access
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append((i, lit))
    
    def is_satisfied(assignment):
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walk_sat(assignment, max_steps, rng, p, deadline):
        """
        Optimized WalkSAT with incremental updates and adaptive random probability.
        """
        # Initialize clause satisfaction counts
        clause_sat_count = [0] * m
        unsat_clauses = []
        
        for i, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return True
            
        unsat_set = set(unsat_clauses)
        unsat_list = list(unsat_clauses)
        
        # Track progress for adaptive behavior
        best_unsat = len(unsat_clauses)
        stall_counter = 0
        last_improvement_step = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
                
            if not unsat_set:
                return True
            
            # Adaptive random probability: increase when stalled
            effective_p = p
            if step - last_improvement_step > 500:
                effective_p = min(0.9, p + 0.2)
            
            # Pick a random unsatisfied clause from the list (maintained in sync)
            c_idx = unsat_list[rng.randrange(len(unsat_list))]
            
            clause = clauses[c_idx]
            
            # Evaluate flips for literals in this clause
            best_gain = -100
            best_var = -1
            flip_candidates = []
            
            for lit in clause:
                var = abs(lit) - 1
                current_val = assignment[var]
                desired_val = (lit > 0)
                
                # Since c_idx is unsat, all its literals are currently unsatisfied
                # Flipping 'var' will satisfy 'lit' in c_idx. Gain starts at +1.
                gain = 1
                
                # Check impact on other clauses containing 'var'
                for (ci2, lit2) in var_to_clauses[var]:
                    if ci2 == c_idx:
                        continue
                    
                    is_sat2 = assignment[var] == (lit2 > 0)
                    
                    if is_sat2:
                        # Currently satisfied. Flipping 'var' will make it unsatisfied.
                        if clause_sat_count[ci2] == 1:
                            gain -= 1
                    else:
                        # Currently unsatisfied. Flipping 'var' will make it satisfied.
                        if clause_sat_count[ci2] == 0:
                            gain += 1
                    
                flip_candidates.append((var, gain))
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            # Select variable to flip
            if rng.random() < effective_p:
                var = flip_candidates[rng.randrange(len(flip_candidates))][0]
            else:
                var = best_var
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update clause_sat_count and unsat tracking
            new_unsat_list = []
            for (ci2, lit2) in var_to_clauses[var]:
                old_count = clause_sat_count[ci2]
                new_is_sat = assignment[var] == (lit2 > 0)
                old_is_sat = not new_is_sat
                
                if new_is_sat and not old_is_sat:
                    clause_sat_count[ci2] = old_count + 1
                    if old_count == 0:
                        unsat_set.discard(ci2)
                elif not new_is_sat and old_is_sat:
                    clause_sat_count[ci2] = old_count - 1
                    if old_count == 1:
                        unsat_set.add(ci2)
            
            # Rebuild unsat_list periodically or when size changes significantly
            if step % 100 == 0 or len(unsat_set) != len(unsat_list):
                unsat_list = list(unsat_set)
            
            # Track progress
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                last_improvement_step = step
                stall_counter = 0
            else:
                stall_counter += 1
        
        return not unsat_set

    # Strategy: Multiple restarts with varying parameters
    base_steps = max(1000, n * 50)
    
    restart_count = 0
    p_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_assignment = None
    best_unsat_count = m + 1
    
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if is_satisfied(assignment):
            return assignment
        
        p = p_values[restart_count % len(p_values)]
        
        # Run WalkSAT
        if walk_sat(assignment, base_steps, rng, p, deadline):
            return assignment
        
        # Track best unsat count for potential polish
        unsat_count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
    # Final polish: try a short, focused search from the best assignment found
    if best_assignment is not None and time.time() < deadline - 3.0:
        rng = random.Random(999999)
        polish_steps = min(5000, int((deadline - time.time()) * 200))
        if walk_sat(best_assignment, polish_steps, rng, 0.3, deadline):
            return best_assignment
    
    # Fallback: return best assignment or random
    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
