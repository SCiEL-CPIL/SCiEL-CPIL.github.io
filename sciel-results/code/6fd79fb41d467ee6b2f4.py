import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 325 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    # DPLL with unit propagation preprocessing
    def dpll_preprocess(clauses, n):
        """
        Returns (assignment_dict, remaining_clauses, unsat_flag)
        assignment_dict maps var_index (0-based) to bool
        remaining_clauses are the clauses after simplification
        """
        assignment = {}
        clauses_to_process = [list(c) for c in clauses]
        
        # Build index for fast lookup
        var_clauses = [[] for _ in range(n)]
        for ci, clause in enumerate(clauses_to_process):
            for lit in clause:
                v = abs(lit) - 1
                var_clauses[v].append(ci)
        
        changed = True
        while changed:
            changed = False
            
            # Unit propagation
            for ci in range(len(clauses_to_process)):
                if ci >= len(clauses_to_process):
                    continue
                clause = clauses_to_process[ci]
                if clause is None:
                    continue
                
                # Find unit clauses
                if len(clause) == 1:
                    lit = clause[0]
                    v = abs(lit) - 1
                    val = lit > 0
                    
                    # Check for conflict
                    if v in assignment and assignment[v] != val:
                        return None, None, True  # UNSAT
                    if v not in assignment:
                        assignment[v] = val
                        changed = True
                        # Simplify all clauses containing v
                        for other_ci in var_clauses[v]:
                            if other_ci < len(clauses_to_process) and clauses_to_process[other_ci] is not None:
                                other_clause = clauses_to_process[other_ci]
                                new_clause = [l for l in other_clause if abs(l) != v + 1]
                                if len(new_clause) == 0:
                                    # Clause satisfied
                                    clauses_to_process[other_ci] = None
                                else:
                                    clauses_to_process[other_ci] = new_clause
                        # Mark this clause as satisfied
                        clauses_to_process[ci] = None
            
            # Remove satisfied clauses and update
            new_clauses = []
            new_var_clauses = [[] for _ in range(n)]
            for ci, clause in enumerate(clauses_to_process):
                if clause is None:
                    continue
                # Check if clause is satisfied
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    if v in assignment:
                        if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                            satisfied = True
                            break
                if not satisfied:
                    new_ci = len(new_clauses)
                    new_clauses.append(clause)
                    for lit in clause:
                        v = abs(lit) - 1
                        new_var_clauses[v].append(new_ci)
            
            if len(new_clauses) != len([c for c in clauses_to_process if c is not None]):
                changed = True
            clauses_to_process = new_clauses
            var_clauses = new_var_clauses
        
        return assignment, clauses_to_process, False
    
    # Run DPLL preprocessing
    pre_assignment, remaining_clauses, is_unsat = dpll_preprocess(clauses, n)
    
    if is_unsat:
        # Should not happen for planted satisfiable instances, but handle gracefully
        return [False] * n
    
    # If DPLL solved it completely
    if not remaining_clauses:
        final_assignment = [False] * n
        for v, val in pre_assignment.items():
            final_assignment[v] = val
        return final_assignment
    
    # Build index for remaining clauses
    remaining_m = len(remaining_clauses)
    var_occurrences = [[] for _ in range(n)]
    for i, c in enumerate(remaining_clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # Map to use remaining clauses in WalkSAT
    clauses = remaining_clauses
    m = remaining_m
    
    # Helper: compute clause_sat_count from scratch
    def compute_clause_sat_count(assignment):
        counts = [0] * m
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    counts[i] += 1
        return counts
    
    # Helper: check if all clauses satisfied
    def is_satisfied(clause_sat_count):
        for count in clause_sat_count:
            if count == 0:
                return False
        return True
    
    # Helper: get unsat clause indices
    def get_unsat_indices(clause_sat_count):
        return [i for i, count in enumerate(clause_sat_count) if count == 0]
    
    # Helper: check if a literal is satisfied by current assignment
    def is_literal_satisfied(l, assignment):
        v = abs(l) - 1
        return (l > 0 and assignment[v]) or (l < 0 and not assignment[v])
    
    # Helper: apply a flip to variable v, updating clause_sat_count
    def apply_flip(v, assignment, clause_sat_count):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            if is_literal_satisfied(lit, assignment):
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1

    # WalkSAT step with improved heuristic and unsat set tracking
    def walksat_step(assignment, clause_sat_count, unsat_set, rng, noise):
        if not unsat_set:
            return True
        
        # Pick random unsat clause
        clause_list = list(unsat_set)
        clause_idx = rng.choice(clause_list)
        clause = clauses[clause_idx]
        
        # With probability noise, pick a random literal from the clause
        if rng.random() < noise:
            l = rng.choice(clause)
            v = abs(l) - 1
            old_count = clause_sat_count[clause_idx]
            apply_flip(v, assignment, clause_sat_count)
            new_count = clause_sat_count[clause_idx]
            if old_count == 0 and new_count > 0:
                unsat_set.discard(clause_idx)
            elif old_count > 0 and new_count == 0:
                unsat_set.add(clause_idx)
            # Update other affected clauses
            for ci, lit in var_occurrences[v]:
                if ci == clause_idx:
                    continue
                if clause_sat_count[ci] == 0:
                    unsat_set.add(ci)
                elif ci in unsat_set and clause_sat_count[ci] > 0:
                    unsat_set.discard(ci)
            return False
        
        # Otherwise, pick the literal that minimizes newly unsatisfied clauses
        best_var_idx = None
        best_score = float('inf')
        
        for l in clause:
            v = abs(l) - 1
            
            # Calculate the effect of flipping v
            newly_unsat = 0
            newly_sat = 0
            
            for ci, lit in var_occurrences[v]:
                old_count = clause_sat_count[ci]
                currently_sat = is_literal_satisfied(lit, assignment)
                
                if currently_sat:
                    if old_count == 1:
                        newly_unsat += 1
                else:
                    if old_count == 0:
                        newly_sat += 1
            
            # Score: minimize newly_unsat, maximize newly_sat
            # Add a small random tie-breaker to avoid deterministic cycles
            tie_breaker = rng.random() * 0.01
            score = newly_unsat * 10 - newly_sat + tie_breaker
            if score < best_score:
                best_score = score
                best_var_idx = v
        
        # Apply the best flip
        if best_var_idx is not None:
            old_count = clause_sat_count[clause_idx]
            apply_flip(best_var_idx, assignment, clause_sat_count)
            new_count = clause_sat_count[clause_idx]
            if old_count == 0 and new_count > 0:
                unsat_set.discard(clause_idx)
            elif old_count > 0 and new_count == 0:
                unsat_set.add(clause_idx)
            # Update other affected clauses
            for ci, lit in var_occurrences[best_var_idx]:
                if ci == clause_idx:
                    continue
                if clause_sat_count[ci] == 0:
                    unsat_set.add(ci)
                elif ci in unsat_set and clause_sat_count[ci] > 0:
                    unsat_set.discard(ci)
        
        return False
    
    # Main loop with multiple restarts and adaptive noise
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = float('inf')
    
    restart_count = 0
    stagnation_limit = 500 # If no improvement in this many steps, restart
    
    # Create initial assignment with DPLL pre-assignments
    base_assignment = [False] * n
    for v, val in pre_assignment.items():
        base_assignment[v] = val
    
    while time.time() < deadline:
        # Initialize assignment: use DPLL assignments, random for rest
        assignment = base_assignment[:]
        for v in range(n):
            if v not in pre_assignment:
                assignment[v] = rng.random() < 0.5
        
        clause_sat_count = compute_clause_sat_count(assignment)
        
        # Check if already satisfied
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Build unsat set
        unsat_set = set(get_unsat_indices(clause_sat_count))
        
        if not unsat_set:
            return assignment
        
        # Adaptive noise: start low, increase over steps
        base_noise = 0.2
        max_noise = 0.6
        max_steps_per_run = 10000 + restart_count * 100 # Slightly increase steps per restart
        
        steps = 0
        steps_since_improvement = 0
        current_unsat_count = len(unsat_set)
        
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                if best_assignment is not None:
                    return best_assignment
                return assignment
            
            if not unsat_set:
                return assignment
            
            # Adaptive noise: increase with steps to escape local minima
            progress = steps / max_steps_per_run
            # Use a sine wave for noise to oscillate between exploration and exploitation
            noise = base_noise + (max_noise - base_noise) * (0.5 + 0.5 * math.sin(progress * math.pi * 2))
            
            walksat_step(assignment, clause_sat_count, unsat_set, rng, noise)
            steps += 1
            steps_since_improvement += 1
            
            # Check for improvement
            if len(unsat_set) < current_unsat_count:
                current_unsat_count = len(unsat_set)
                steps_since_improvement = 0
                
                # Track best assignment
                if current_unsat_count < best_unsat_count:
                    best_unsat_count = current_unsat_count
                    best_assignment = assignment[:]
                
                # If solved, return immediately
                if current_unsat_count == 0:
                    return assignment
            
            # Stagnation check: if no improvement for stagnation_limit steps, restart
            if steps_since_improvement > stagnation_limit:
                break
        
        restart_count += 1
    
    if best_assignment is not None:
        return best_assignment
    return assignment
