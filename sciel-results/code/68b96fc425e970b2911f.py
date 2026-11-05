"""Improved planted 3-SAT solver using WalkSAT with adaptive parameters and multiple restarts."""
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Precompute clause structure for speed
    # For each clause, store (var_indices, signs) where sign=1 means positive, -1 means negative
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_c = [abs(l) - 1 for l in c]
        signs_c = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_c)
        clause_signs.append(signs_c)
    
    m = len(clauses)
    
    # Precompute for each variable, which clauses it appears in and with what sign
    # var_clauses[var] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for vi, si in zip(clause_vars[ci], clause_signs[ci]):
            var_clauses[vi].append((ci, si))
    
    def is_satisfied(clause_idx, assignment):
        """Check if clause is satisfied given current assignment."""
        for vi, si in zip(clause_vars[clause_idx], clause_signs[clause_idx]):
            if assignment[vi] == (si == 1):
                return True
        return False
    
    def count_true_literals(clause_idx, assignment):
        """Count how many literals in the clause are true."""
        count = 0
        for vi, si in zip(clause_vars[clause_idx], clause_signs[clause_idx]):
            if assignment[vi] == (si == 1):
                count += 1
        return count
    
    def walksat_once(rng, max_steps):
        """Run a single WalkSAT attempt. Returns (success, best_assignment)."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track best assignment found (fewest unsatisfied clauses)
        best_unsat_count = m + 1
        best_assignment = assignment[:]
        
        # Initial unsatisfied clause count
        unsat_clauses = []
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                unsat_clauses.append(ci)
        
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return True, assignment
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive parameters
        base_noise = 0.5
        noise = base_noise
        step_count = 0
        
        while step_count < max_steps:
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return True, assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Select an unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            
            # Count true literals in this clause
            true_count = count_true_literals(clause_idx, assignment)
            
            # Decide whether to flip a random literal (noise) or a best-improving literal
            if rng.random() < noise:
                # Random flip from the clause
                literal_idx = rng.randrange(len(clause_vars[clause_idx]))
                vi = clause_vars[clause_idx][literal_idx]
            else:
                # Best-improving: flip the variable that satisfies the most other clauses
                # but doesn't unsatisfy too many
                best_score = -1000
                best_vi = clause_vars[clause_idx][0]
                for literal_idx, vi in enumerate(clause_vars[clause_idx]):
                    si = clause_signs[clause_idx][literal_idx]
                    # If we flip vi, will this literal become true?
                    would_be_true = (not assignment[vi]) == (si == 1)
                    
                    # Score: number of clauses that would become satisfied minus
                    # number of clauses that would become unsatisfied
                    score = 0
                    if would_be_true:
                        score += 1  # This clause becomes satisfied
                    else:
                        score -= 1  # This clause remains unsatisfied (or becomes more unsatisfied)
                    
                    # Check impact on other clauses containing vi
                    for (other_ci, other_si) in var_clauses[vi]:
                        if other_ci == clause_idx:
                            continue
                        # Current state of this literal in other clause
                        currently_true = assignment[vi] == (other_si == 1)
                        # After flip
                        after_flip_true = (not assignment[vi]) == (other_si == 1)
                        
                        if currently_true and not after_flip_true:
                            # This clause was satisfied, would become unsatisfied
                            # Check if it has other true literals
                            if count_true_literals(other_ci, assignment) <= 1:
                                score -= 1
                        elif not currently_true and after_flip_true:
                            # This clause was unsatisfied, would become satisfied
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_vi = vi
                vi = best_vi
            
            # Flip the variable
            assignment[vi] = not assignment[vi]
            
            # Update unsatisfied clauses
            # Find which clauses are affected by flipping vi
            new_unsat = []
            for ci in range(m):
                if not is_satisfied(ci, assignment):
                    new_unsat.append(ci)
            
            current_unsat = len(new_unsat)
            unsat_clauses = new_unsat
            
            # Adapt noise parameter
            noise = base_noise + 0.1 * (current_unsat / m)
            noise = min(noise, 0.8)
            
            step_count += 1
        
        return False, best_assignment
    
    # Multiple restarts with different random seeds
    best_assignment = None
    best_unsat_count = m + 1
    
    attempt = 0
    while time.time() < deadline:
        # Create a new RNG for each attempt with different seed
        rng = random.Random(attempt * 7919 + 13)
        
        # Increase max steps for later attempts
        max_steps = 5000 + attempt * 2000
        
        success, result_assignment = walksat_once(rng, max_steps)
        
        if success:
            return result_assignment
        
        # Track best
        unsat_count = sum(1 for ci in range(m) if not is_satisfied(ci, result_assignment))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = result_assignment
        
        attempt += 1
        
        # If we've done many attempts, increase step budget
        if attempt % 5 == 0:
            pass  # Already handled by max_steps formula
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        rng = random.Random(42)
        return [rng.random() < 0.5 for _ in range(n)]
