"""Synthetic planted 3-SAT solver (50 variables) - WalkSAT family with restarts and time guard."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance.
    Returns a list of booleans (length n) satisfying all clauses, or best-effort if time runs out.
    Uses WalkSAT with adaptive restarts and a hard wall-clock deadline.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Hard deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Preprocess: store clauses as lists of (var_index, is_positive) for faster access
    # var_index is 0-based, is_positive is bool
    clauses_processed = []
    for c in clauses:
        processed_clause = []
        for lit in c:
            var_idx = abs(lit) - 1
            is_pos = (lit > 0)
            processed_clause.append((var_idx, is_pos))
        clauses_processed.append(processed_clause)
    
    m = len(clauses_processed)
    
    # Count how many times each variable appears in each polarity
    # For smarter initial assignment or tie-breaking
    var_count = [0] * n
    var_pos_count = [0] * n
    var_neg_count = [0] * n
    for c in clauses_processed:
        for var_idx, is_pos in c:
            var_count[var_idx] += 1
            if is_pos:
                var_pos_count[var_idx] += 1
            else:
                var_neg_count[var_idx] += 1
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Multiple restarts with increasing step budgets
    max_restarts = 50
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        # Initialize assignment
        # Use a slightly biased random initialization
        rng = random.Random(42 + restart * 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check if already satisfied (rare)
        unsat_count = 0
        for c in clauses_processed:
            sat = False
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        # Parameters: noise parameter (probability of random flip)
        noise = 0.25 + 0.05 * (restart % 4)  # Vary noise slightly per restart
        
        # Adaptive step budget: increase slightly with restarts but cap it
        steps = base_steps + restart * 200
        steps = min(steps, 20000)
        
        for step in range(steps):
            if time.time() >= deadline - 2.0:
                # Save best and exit
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
                return best_assignment if best_assignment is not None else assignment
            
            # Find an unsatisfied clause
            # Randomly check a few clauses to find an unsat one
            # For efficiency, we can maintain a list of unsat clauses, but for simplicity:
            # Pick a random clause; if satisfied, try a few more
            clause_idx = rng.randrange(m)
            clause = clauses_processed[clause_idx]
            
            # Check if clause is satisfied
            sat = False
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            
            if sat:
                # Try a few more random clauses to find an unsat one
                for _ in range(3):
                    clause_idx = rng.randrange(m)
                    clause = clauses_processed[clause_idx]
                    sat = False
                    for var_idx, is_pos in clause:
                        if assignment[var_idx] == is_pos:
                            sat = True
                            break
                    if not sat:
                        break
                if sat:
                    # All tried clauses are satisfied; assume we're close to a solution
                    # Do a full check to be sure
                    all_sat = True
                    for c in clauses_processed:
                        c_sat = False
                        for var_idx, is_pos in c:
                            if assignment[var_idx] == is_pos:
                                c_sat = True
                                break
                        if not c_sat:
                            all_sat = False
                            break
                    if all_sat:
                        return assignment
                    # Otherwise, pick a random clause and proceed
                    clause_idx = rng.randrange(m)
                    clause = clauses_processed[clause_idx]
            
            # Now clause is unsatisfied
            # Choose a literal to flip
            # With probability (1 - noise), flip the literal that satisfies the most other clauses
            # With probability noise, flip a random literal in the clause
            if rng.random() < noise:
                lit_idx = rng.randrange(len(clause))
                var_idx, is_pos = clause[lit_idx]
            else:
                # Greedy: pick literal that maximizes number of clauses satisfied if flipped
                best_var = -1
                best_gain = -1
                for lit_pos, (var_idx, is_pos) in enumerate(clause):
                    # If we flip var_idx, this clause becomes satisfied
                    # Count how many other clauses would be affected
                    # For each other clause containing var_idx, check if it becomes satisfied or unsatisfied
                    gain = 0
                    for other_c in clauses_processed:
                        # Check if other_c contains var_idx
                        contains_var = False
                        other_c_satisfied_now = False
                        other_c_satisfied_after = False
                        for v_idx, v_is_pos in other_c:
                            if v_idx == var_idx:
                                contains_var = True
                                # Current state
                                if assignment[v_idx] == v_is_pos:
                                    other_c_satisfied_now = True
                                # After flip: assignment[v_idx] becomes not assignment[v_idx]
                                if (not assignment[v_idx]) == v_is_pos:
                                    other_c_satisfied_after = True
                                break
                            else:
                                if assignment[v_idx] == v_is_pos:
                                    other_c_satisfied_now = True
                                    other_c_satisfied_after = True
                            if other_c_satisfied_now:
                                break
                        
                        if contains_var:
                            if other_c_satisfied_now and not other_c_satisfied_after:
                                gain -= 1
                            elif not other_c_satisfied_now and other_c_satisfied_after:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = lit_pos
                
                if best_var >= 0:
                    var_idx, is_pos = clause[best_var]
                else:
                    var_idx, is_pos = clause[0]
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_count approximately or do a full check periodically
            # For correctness, do a full check every 100 steps or when we think we're close
            if step % 100 == 99 or step == steps - 1:
                unsat_count = 0
                for c in clauses_processed:
                    c_sat = False
                    for var_idx2, is_pos2 in c:
                        if assignment[var_idx2] == is_pos2:
                            c_sat = True
                            break
                    if not c_sat:
                        unsat_count += 1
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
        
        # End of steps for this restart
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
