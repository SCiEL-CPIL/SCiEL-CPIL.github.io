"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False
    
    # WalkSAT with restarts and improved heuristics
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Adaptive parameters
    base_walk_prob = 0.20
    max_walk_prob = 0.50
    progress_threshold = 500  # steps without improvement before increasing walk prob
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Initialize assignment randomly
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track current unsat count and unsat clause set
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        # Pre-compute per-variable unsat clause count for fast heuristic
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            fc = fast_clauses[c_idx]
            for var_idx, _ in fc:
                var_unsat_count[var_idx] += 1
        
        current_unsat = len(unsat_set)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk probability for this restart
        walk_prob = base_walk_prob
        steps_since_improvement = 0
        local_best_unsat = current_unsat
        max_steps = 15000  # More steps per restart with adaptive walk prob
        
        # WalkSAT loop with clause status tracking
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
            
            if len(unsat_set) < local_best_unsat:
                local_best_unsat = len(unsat_set)
                steps_since_improvement = 0
                # Reset walk probability when we make progress
                walk_prob = base_walk_prob
            else:
                steps_since_improvement += 1
                # Gradually increase walk probability when stuck
                if steps_since_improvement > progress_threshold:
                    walk_prob = min(max_walk_prob, base_walk_prob + (steps_since_improvement - progress_threshold) * 0.001)
            
            # Choose a clause from unsat_set
            # Weight toward clauses with fewer satisfied literals (harder clauses)
            clause_idx = random.choice(list(unsat_set))
            clause = fast_clauses[clause_idx]
            
            # With probability walk_prob, pick a random literal from the clause
            # Otherwise, use combined frequency + phase-savings heuristic
            if random.random() < walk_prob:
                var_idx, required = random.choice(clause)
            else:
                # Combined heuristic: frequency + phase savings
                # For each literal in the clause, compute:
                # 1. freq: how many unsat clauses contain this variable
                # 2. phase_savings: how many other literals in unsat clauses would be satisfied
                #    if we set this variable to the required value
                scores = []
                for var_idx, required in clause:
                    freq = var_unsat_count[var_idx]
                    
                    # Phase savings: count how many unsat clauses would gain a satisfied
                    # literal if we flip this variable to satisfy the current clause
                    phase_savings = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx in unsat_set and c_idx != clause_idx:
                            fc = fast_clauses[c_idx]
                            for v_idx, req in fc:
                                if v_idx == var_idx:
                                    # This clause has our variable
                                    # Check if setting var to 'required' would satisfy it
                                    # (it might already have another satisfied literal, but
                                    #  since it's in unsat_set, it doesn't)
                                    # So flipping var to satisfy this clause's literal for var
                                    # would help if the literal matches
                                    if req == required:
                                        phase_savings += 1
                                    break
                    
                    score = freq + phase_savings
                    scores.append(score)
                
                total_score = sum(scores)
                if total_score == 0:
                    var_idx, required = random.choice(clause)
                else:
                    # Weighted selection
                    r = random.random() * total_score
                    cumulative = 0
                    var_idx, required = clause[0]
                    for i, score in enumerate(scores):
                        cumulative += score
                        if r <= cumulative:
                            var_idx, required = clause[i]
                            break
            
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set and var_unsat_count incrementally
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_sat = is_satisfied(c_idx, assignment)
                
                if was_unsat and is_sat:
                    # Clause became satisfied: remove from unsat_set, decrement counts
                    unsat_set.discard(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] -= 1
                elif not was_unsat and not is_sat:
                    # Clause became unsatisfied: add to unsat_set, increment counts
                    unsat_set.add(c_idx)
                    fc = fast_clauses[c_idx]
                    for v_idx, _ in fc:
                        var_unsat_count[v_idx] += 1
            
            # Early exit if solved
            if not unsat_set:
                return assignment
        
        # End of restart
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
