import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 280 seconds (safety margin for 300s limit)
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Pre-compute clause structure for fast evaluation
    # For each clause, store (var_indices, signs) where signs[i] is +1 or -1
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_in_clause = [abs(l) - 1 for l in clause]
        signs_in_clause = [1 if l > 0 else -1 for l in clause]
        clause_vars.append(vars_in_clause)
        clause_signs.append(signs_in_clause)
    
    # Build variable-to-clause mapping for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_vars):
        for v in clause:
            var_to_clauses[v].append(ci)
    
    def evaluate(assignment):
        """Returns the number of unsatisfied clauses."""
        unsat_count = 0
        for ci in range(len(clauses)):
            sat = False
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count
    
    def get_unsat_clauses(assignment):
        """Returns list of unsatisfied clause indices."""
        unsat = []
        for ci in range(len(clauses)):
            sat = False
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat
    
    def compute_conflicts(assignment):
        """For each variable, compute (num_unsat_clauses, num_satisfied_literals_in_unsat)."""
        # First, identify unsatisfied clauses
        unsat_clauses = []
        for ci in range(len(clauses)):
            sat = False
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        # Count conflicts per variable
        conflict_count = [0] * n
        satisfied_in_unsat = [0] * n  # How many literals of this var are satisfied in unsat clauses
        
        for ci in unsat_clauses:
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                conflict_count[v] += 1
                # Check if this literal is satisfied
                if assignment[v] == (s == 1):
                    satisfied_in_unsat[v] += 1
        
        return conflict_count, satisfied_in_unsat, unsat_clauses
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    max_steps_per_run = 3000
    max_restarts = 2000
    p_random = 0.3  # Probability of random flip (WalkSAT style)
    p_greedy = 0.7  # Probability of greedy flip (Min-Conflicts style)
    
    rng = random.Random(0)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat
        unsat_count = evaluate(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Local search
        steps = 0
        last_improvement_step = 0
        current_unsat = unsat_count
        stagnation_threshold = 800
        
        while steps < max_steps_per_run:
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
                
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                last_improvement_step = steps
                
            # Stagnation check: if no improvement for threshold steps, break and restart
            if steps - last_improvement_step > stagnation_threshold:
                break
                
            # Compute conflict information
            conflict_count, satisfied_in_unsat, unsat_clauses = compute_conflicts(assignment)
            
            if not unsat_clauses:
                return assignment
            
            # Variable selection heuristic
            # Score = conflict_count - satisfied_in_unsat (higher is better to flip)
            # This favors variables that are in many unsat clauses but have few satisfied literals
            # (i.e., flipping them is likely to satisfy more clauses)
            
            candidate_vars = []
            for v in range(n):
                if conflict_count[v] > 0:
                    # Expected improvement: if we flip v, how many clauses might become satisfied?
                    # A clause becomes satisfied if it has exactly one unsatisfied literal and we flip that var
                    # For simplicity, use a heuristic score
                    score = conflict_count[v] * 2.0 - satisfied_in_unsat[v]
                    candidate_vars.append((score, v))
            
            if not candidate_vars:
                break
            
            # Sort by score descending
            candidate_vars.sort(reverse=True)
            
            # Select variable with probability proportional to score
            # Or use a simpler approach: pick from top candidates
            if rng.random() < p_random:
                # Random walk: pick a random variable from a random unsat clause
                ci = rng.choice(unsat_clauses)
                v = rng.choice(clause_vars[ci])
            else:
                # Greedy: pick from top candidates
                # Use top 5 candidates for diversity
                top_k = min(5, len(candidate_vars))
                top_candidates = [v for _, v in candidate_vars[:top_k]]
                v = rng.choice(top_candidates)
            
            # Flip variable
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            # Incrementally update unsat count
            # Only clauses containing v can change status
            delta = 0
            for ci in var_to_clauses[v]:
                # Check if clause was satisfied before flip
                was_sat = False
                for vv, s in zip(clause_vars[ci], clause_signs[ci]):
                    if vv == v:
                        # Use old value for this variable
                        if old_val == (s == 1):
                            was_sat = True
                            break
                    else:
                        if assignment[vv] == (s == 1):
                            was_sat = True
                            break
                
                # Check if clause is satisfied after flip
                is_sat = False
                for vv, s in zip(clause_vars[ci], clause_signs[ci]):
                    if vv == v:
                        # Use new value for this variable
                        if assignment[v] == (s == 1):
                            is_sat = True
                            break
                    else:
                        if assignment[vv] == (s == 1):
                            is_sat = True
                            break
                
                if was_sat and not is_sat:
                    delta += 1  # Lost satisfaction
                elif not was_sat and is_sat:
                    delta -= 1  # Gained satisfaction
            
            current_unsat += delta
            steps += 1
            
    # If loop finishes without finding solution, return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
