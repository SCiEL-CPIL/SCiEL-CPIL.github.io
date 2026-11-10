"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using a hybrid approach:
    1. Try a fast local search (WalkSAT variant with clause weighting)
    2. If that fails, try a more systematic approach with backtracking limited
    3. Use multiple random restarts with different strategies
    
    The key insight for planted 3-SAT is that the solution is close to uniform
    random assignments in terms of distribution, but the structure of clauses
    provides strong signals for local search.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess: build variable-to-clause index for efficiency
    var_to_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(idx)
    
    def is_satisfied(clause, assignment):
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                return True
        return False
    
    def count_unsat_clauses(assignment):
        """Count number of unsatisfied clauses"""
        count = 0
        for clause in clauses:
            if not is_satisfied(clause, assignment):
                count += 1
        return count
    
    def walk_sat(assignment, max_flips, rng, noise=0.5):
        """
        WalkSAT with clause weighting and restart-like behavior.
        Returns True if satisfied, False otherwise.
        """
        n_vars = len(assignment)
        
        # Track clause weights for bias
        clause_weights = [1.0] * num_clauses
        
        for flip in range(max_flips):
            # Find unsatisfied clauses
            unsat_clauses = []
            for idx, clause in enumerate(clauses):
                if not is_satisfied(clause, assignment):
                    unsat_clauses.append(idx)
            
            if not unsat_clauses:
                return True
            
            # Weighted selection of unsatisfied clause
            total_weight = sum(clause_weights[idx] for idx in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_idx = unsat_clauses[0]
            for idx in unsat_clauses:
                cumulative += clause_weights[idx]
                if r <= cumulative:
                    chosen_idx = idx
                    break
            
            clause = clauses[chosen_idx]
            
            # Count how many literals in this clause would be satisfied by flipping
            # For each variable in the clause, compute gain/loss
            best_var = None
            best_score = -1
            scores = []
            
            for lit in clause:
                var = abs(lit) - 1
                current_val = assignment[var]
                desired_val = (lit > 0)
                
                # If we flip this variable, how many clauses are affected?
                gain = 0
                loss = 0
                for cidx in var_to_clauses[var]:
                    c = clauses[cidx]
                    # Check if this clause is currently satisfied
                    currently_sat = is_satisfied(c, assignment)
                    # Check if it would be satisfied after flip
                    # After flip, variable var has value not current_val
                    would_be_sat = False
                    for l in c:
                        v = abs(l) - 1
                        new_val = (not current_val) if v == var else assignment[v]
                        if new_val == (l > 0):
                            would_be_sat = True
                            break
                    
                    if currently_sat and not would_be_sat:
                        loss += clause_weights[cidx]
                    elif not currently_sat and would_be_sat:
                        gain += clause_weights[cidx]
                
                score = gain - loss
                scores.append((score, var))
            
            # Choose variable to flip: with probability noise, pick randomly;
            # otherwise pick the one with best score
            if rng.random() < noise:
                _, best_var = rng.choice(scores)
            else:
                best_score, best_var = max(scores, key=lambda x: x[0])
            
            # Flip the variable
            assignment[best_var] = not assignment[best_var]
            
            # Update clause weights: increase weight of unsatisfied clauses
            # that contain the flipped variable, decay others
            for cidx in var_to_clauses[best_var]:
                if not is_satisfied(clauses[cidx], assignment):
                    clause_weights[cidx] = min(clause_weights[cidx] * 1.2, 10.0)
                else:
                    clause_weights[cidx] = max(clause_weights[cidx] * 0.95, 1.0)
        
        return False
    
    # Strategy 1: Multiple WalkSAT restarts with varying parameters
    start_time = time.time()
    time_limit = 290  # Leave buffer for timeout
    
    best_solution = None
    best_unsat = float('inf')
    
    rng = random.Random(42)
    
    # Try several restarts with different noise levels
    noise_levels = [0.5, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.5]
    flips_per_try = [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000]
    
    for trial in range(len(noise_levels)):
        if time.time() - start_time > time_limit:
            break
        
        noise = noise_levels[trial]
        max_flips = flips_per_try[trial]
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        success = walk_sat(assignment, max_flips, rng, noise)
        
        if success:
            return assignment
        
        # Track best solution found
        unsat_count = count_unsat_clauses(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_solution = assignment[:]
        
        # If we're getting close, try a more focused search
        if unsat_count <= 5 and time.time() - start_time < time_limit:
            # Try to fix the remaining unsatisfied clauses
            for _ in range(10000):
                if time.time() - start_time > time_limit:
                    break
                unsat = [c for c in clauses if not is_satisfied(c, assignment)]
                if not unsat:
                    return assignment
                # Pick a random unsatisfied clause and flip one of its variables
                clause = rng.choice(unsat)
                lit = rng.choice(clause)
                var = abs(lit) - 1
                assignment[var] = not assignment[var]
                if all(is_satisfied(c, assignment) for c in clauses):
                    return assignment
    
    # Strategy 2: If local search failed, try a simple DPLL with limited depth
    if time.time() - start_time < time_limit - 30:
        result = dpll_limited(instance, time_limit - (time.time() - start_time) - 5)
        if result is not None:
            return result
    
    # Return best solution found
    if best_solution is not None:
        return best_solution
    return [False] * n


def dpll_limited(instance, time_budget):
    """
    Limited DPLL for small instances or as a fallback.
    Returns assignment if found, None if not found within time.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    start = time.time()
    
    def unit_propagate(clauses, assignment):
        """
        Apply unit propagation. Returns (new_clauses, new_assignment, conflict).
        assignment is a dict: var_index -> bool
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Remove satisfied literals
                new_lit = []
                satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    if var in assignment:
                        if assignment[var] == (lit > 0):
                            satisfied = True
                            break
                    else:
                        new_lit.append(lit)
                
                if satisfied:
                    continue
                
                if len(new_lit) == 0:
                    return None, None, True  # Conflict
                
                if len(new_lit) == 1:
                    lit = new_lit[0]
                    var = abs(lit) - 1
                    if var in assignment:
                        if assignment[var] == (lit > 0):
                            continue
                        else:
                            return None, None, True  # Conflict
                    assignment[var] = (lit > 0)
                    changed = True
                else:
                    new_clauses.append(new_lit)
        
        return new_clauses, assignment, False
    
    def dpll(clauses, assignment, depth):
        if time.time() - start > time_budget:
            return None
        
        if depth > 50:  # Limit depth
            return None
        
        new_clauses, assignment, conflict = unit_propagate(clauses, assignment)
        if conflict:
            return None
        if new_clauses is None:
            new_clauses = []
        if not new_clauses:
            # All clauses satisfied
            result = [False] * n
            for var in range(n):
                result[var] = assignment.get(var, False)
            return result
        
        # Pick a variable with highest occurrence
        var_counts = {}
        for clause in new_clauses:
            for lit in clause:
                var = abs(lit) - 1
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            result = [False] * n
            for var in range(n):
                result[var] = assignment.get(var, False)
            return result
        
        var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True first
        assignment[var] = True
        result = dpll(new_clauses, assignment, depth + 1)
        if result is not None:
            return result
        
        # Try False
        assignment[var] = False
        result = dpll(new_clauses, assignment, depth + 1)
        if result is not None:
            return result
        
        return None
    
    initial_assignment = {}
    return dpll(clauses, initial_assignment, 0)
# EVOLVE-BLOCK-END
