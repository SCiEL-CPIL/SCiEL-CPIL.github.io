import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process: remove tautologies and empty clauses, deduplicate
    processed_clauses = []
    seen = set()
    for clause in clauses:
        # Check for tautology
        lit_set = set()
        is_taut = False
        for l in clause:
            if -l in lit_set:
                is_taut = True
                break
            lit_set.add(l)
        if is_taut:
            continue
        # Normalize for dedup
        norm = tuple(sorted(lit_set))
        if norm in seen:
            continue
        seen.add(norm)
        processed_clauses.append(list(lit_set))
    
    clauses = processed_clauses
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Check for empty clause
    for c in clauses:
        if len(c) == 0:
            # Instance is UNSAT, but task says satisfiable, so this shouldn't happen
            # Return arbitrary assignment
            return [True] * n

    # Unit Propagation and Pure Literal Elimination to simplify
    assignment = [None] * n  # None = unassigned, True/False = assigned
    
    def is_true(lit, ass):
        v = abs(lit) - 1
        val = ass[v]
        if val is None:
            return False
        return (lit > 0 and val) or (lit < 0 and not val)

    def is_false(lit, ass):
        v = abs(lit) - 1
        val = ass[v]
        if val is None:
            return False
        return (lit > 0 and not val) or (lit < 0 and val)

    def is_unassigned(lit, ass):
        v = abs(lit) - 1
        return ass[v] is None

    # Apply Unit Propagation and Pure Literal Elimination
    changed = True
    while changed:
        changed = False
        
        # Unit Propagation
        for clause in clauses:
            unassigned_lits = [l for l in clause if is_unassigned(l, assignment)]
            if len(unassigned_lits) == 1:
                l = unassigned_lits[0]
                v = abs(l) - 1
                val = l > 0
                if assignment[v] is None:
                    assignment[v] = val
                    changed = True
                    break # Restart loop as assignments change
            elif len(unassigned_lits) == 0:
                # Check if satisfied
                if not any(is_true(l, assignment) for l in clause):
                    # Contradiction, but task says SAT. Just break.
                    pass
        
        # Pure Literal Elimination
        if changed:
            continue # Restart loop after unit prop
            
        # Find pure literals
        positive_vars = set()
        negative_vars = set()
        for clause in clauses:
            for l in clause:
                if l > 0:
                    positive_vars.add(l)
                else:
                    negative_vars.add(-l)
        
        pure_vars = positive_vars - negative_vars
        for v in pure_vars:
            if assignment[v] is None:
                assignment[v] = True
                changed = True
                break # Restart loop after pure lit elimination
        
        # Also check negative pure literals
        if not changed:
            pure_vars_neg = negative_vars - positive_vars
            for v in pure_vars_neg:
                if assignment[v] is None:
                    assignment[v] = False
                    changed = True
                    break

    # If all assigned, check if solution
    if all(v is not None for v in assignment):
        valid = all(any(is_true(l, assignment) for l in clause) for clause in clauses)
        if valid:
            return [bool(v) for v in assignment]

    # Prepare for WalkSAT
    # Convert to 0/1 representation for speed? Or keep True/False?
    # Let's use 0/1 for faster array access in numpy-like logic if we were using numpy, but pure python list is fine.
    # We will use a list of bools.
    # We need to handle the pre-assigned variables.
    
    start_time = time.time()
    timeout = 290.0 # Leave 10s buffer
    
    rng = random.Random(42)
    
    # Initial assignment for WalkSAT
    # For pre-assigned vars, keep their value. For others, random.
    base_assignment = []
    for v in range(n):
        if assignment[v] is not None:
            base_assignment.append(assignment[v])
        else:
            base_assignment.append(rng.random() < 0.5)
            
    def is_clause_satisfied(clause, ass):
        for l in clause:
            v = abs(l) - 1
            if (l > 0 and ass[v]) or (l < 0 and not ass[v]):
                return True
        return False

    def is_lit_satisfied(l, ass):
        v = abs(l) - 1
        if l > 0:
            return ass[v]
        else:
            return not ass[v]

    def is_lit_false(l, ass):
        v = abs(l) - 1
        if l > 0:
            return not ass[v]
        else:
            return ass[v]

    # WalkSAT with Restarts
    # Parameters:
    # noise: probability of flipping a random variable in the clause
    # max_flips_per_restart: how many flips before restarting
    # num_restarts: how many times to restart
    
    noise = 0.3
    max_flips_per_restart = 5000
    num_restarts = 500
    
    # We can adaptively increase noise or restarts if time permits.
    # For now, fixed robust parameters.
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Create a copy of the assignment for this restart
        # We keep the pre-assigned variables fixed? 
        # Usually in WalkSAT, we restart from scratch or from a previous failed state.
        # Since we have pre-assigned vars from UP, it's safer to keep them fixed 
        # if they don't conflict, but if they lead to a dead end, we might need to unfix them.
        # However, UP guarantees consistency for the unit clauses.
        # Let's just restart with random values for unassigned, and keep assigned.
        
        current_ass = base_assignment[:]
        
        # Check if already satisfied
        if all(is_clause_satisfied(c, current_ass) for c in clauses):
            return current_ass
            
        # Walk
        for flip in range(max_flips_per_restart):
            if time.time() - start_time > timeout:
                break
                
            # Find unsatisfied clauses
            unsat_clauses = [c for c in clauses if not is_clause_satisfied(c, current_ass)]
            
            if not unsat_clauses:
                return current_ass
                
            # Select a random unsatisfied clause
            # Weighted selection? Uniform is standard.
            clause = rng.choice(unsat_clauses)
            
            # Choose a literal to flip
            # With probability noise, pick random variable from clause
            # With probability 1-noise, pick variable that satisfies most other unsatisfied clauses
            if rng.random() < noise:
                # Pick random variable from clause
                lit = rng.choice(clause)
                v = abs(lit) - 1
                current_ass[v] = not current_ass[v]
            else:
                # Pick variable that maximizes gain
                best_gain = -1
                best_v = None
                # Pre-compute unsat clauses set for faster lookup? 
                # For 150 vars and ~300 clauses, iterating is fine.
                # We want to flip a variable v in clause such that
                # number of unsat clauses made sat - number of sat clauses made unsat is maximized.
                
                # Since we only flip one var, we can just count how many unsat clauses contain v.
                # Actually, we need to be careful.
                # Let's just count occurrences in unsat_clauses.
                var_counts = {}
                for c in unsat_clauses:
                    for l in c:
                        v = abs(l) - 1
                        if v not in var_counts:
                            var_counts[v] = 0
                        var_counts[v] += 1
                        
                # Filter to variables in the selected clause
                clause_vars = set(abs(l) - 1 for l in clause)
                best_gain = -1
                best_v = None
                for v in clause_vars:
                    # Gain is roughly var_counts[v]
                    # But we also lose clauses that were satisfied.
                    # To be precise, we'd need to check each clause.
                    # Approximation: just pick the one with highest count in unsat clauses.
                    gain = var_counts.get(v, 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        
                if best_v is not None:
                    current_ass[best_v] = not current_ass[best_v]
                    
        # End of flip loop for this restart

    # If we exhausted restarts or time, return the last assignment or random
    # To be safe, return a random assignment if we didn't find a solution
    return base_assignment
