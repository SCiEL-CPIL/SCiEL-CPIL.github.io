"""Synthetic planted 3-SAT solver (150 variables) with preprocessing and improved WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Solve a planted satisfiable 3-CNF instance.
    
    Uses preprocessing (pure literal elimination + unit propagation) followed by
    clause-weighted WalkSAT with adaptive noise and restarts.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Preprocessing: build clause/literal structures
    # Convert to 0-based, track signs
    # clauses: list of lists of signed 1-based literals
    
    # --- Preprocessing: Pure Literal Elimination + Unit Propagation ---
    def preprocess(n, clauses):
        """Apply pure literal elimination and unit propagation.
        Returns (assignment_dict, remaining_clauses, n_vars) where assignment_dict
        maps var_index (0-based) to True/False for forced assignments.
        """
        assignment = {}  # var_idx -> bool
        
        # Work with a mutable copy
        remaining = [list(c) for c in clauses]
        
        # Pure literal elimination (iterative)
        changed = True
        while changed:
            changed = False
            # Count occurrences of each literal
            pos_count = [0] * n
            neg_count = [0] * n
            for c in remaining:
                for l in c:
                    if l > 0:
                        pos_count[l - 1] += 1
                    else:
                        neg_count[-l - 1] += 1
            
            # Find pure literals
            for v in range(n):
                if v in assignment:
                    continue
                if pos_count[v] > 0 and neg_count[v] == 0:
                    assignment[v] = True
                    changed = True
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    assignment[v] = False
                    changed = True
            
            if changed:
                # Remove satisfied clauses, simplify
                new_remaining = []
                for c in remaining:
                    sat = False
                    new_c = []
                    for l in c:
                        v = abs(l) - 1
                        if v in assignment:
                            if (l > 0) == assignment[v]:
                                sat = True
                                break
                        else:
                            new_c.append(l)
                    if not sat:
                        new_remaining.append(new_c)
                remaining = new_remaining
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for c in remaining:
                # Check if clause is unit
                unassigned = [l for l in c if (abs(l) - 1) not in assignment]
                if len(unassigned) == 1:
                    l = unassigned[0]
                    v = abs(l) - 1
                    assignment[v] = (l > 0)
                    changed = True
                elif len(unassigned) == 0:
                    # Empty clause - unsatisfiable (shouldn't happen for planted SAT)
                    pass
                else:
                    new_remaining.append(unassigned)
            remaining = new_remaining
        
        return assignment, remaining, n
    
    # Apply preprocessing
    forced_assignment, remaining_clauses, n = preprocess(n, clauses)
    
    # If all clauses are satisfied, we're done
    if not remaining_clauses:
        result = [False] * n
        for v, val in forced_assignment.items():
            result[v] = val
        return result
    
    # Identify free variables
    free_vars = [v for v in range(n) if v not in forced_assignment]
    free_var_set = set(free_vars)
    
    # Build clause structure for WalkSAT on free variables
    # Each clause is a list of (var_idx, sign) for free vars only
    walk_clauses = []
    for c in remaining_clauses:
        wc = []
        for l in c:
            v = abs(l) - 1
            if v in free_var_set:
                wc.append((v, l > 0))
        if wc:  # Only keep clauses with free vars
            walk_clauses.append(wc)
    
    # If no free vars needed, return forced assignment
    if not walk_clauses:
        result = [False] * n
        for v, val in forced_assignment.items():
            result[v] = val
        return result
    
    # --- WalkSAT with clause weighting and adaptive noise ---
    def walksat_solve(n, walk_clauses, free_vars, forced_assignment, deadline):
        """Clause-weighted WalkSAT with adaptive noise and restarts."""
        rng = random.Random(42)
        
        num_clauses = len(walk_clauses)
        if num_clauses == 0:
            return {v: False for v in free_vars}
        
        # Precompute clause literal counts for weighting
        clause_weights = [1.0] * num_clauses
        
        # Variable occurrence counts (for initialization bias)
        var_occ = [0] * n
        for c in walk_clauses:
            for v, sign in c:
                var_occ[v] += 1
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        max_restarts = 500
        max_steps_per_restart = 5000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: random, but biased by occurrence
            assignment = {}
            for v in free_vars:
                # Slight bias: if var appears in many clauses, random is fine
                assignment[v] = rng.random() < 0.5
            
            # Compute initial unsat count
            def count_unsat(assignment):
                cnt = 0
                for c in walk_clauses:
                    sat = False
                    for v, sign in c:
                        if assignment[v] == sign:
                            sat = True
                            break
                    if not sat:
                        cnt += 1
                return cnt
            
            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = dict(assignment)
            
            # WalkSAT loop
            for step in range(max_steps_per_restart):
                if time.time() > deadline:
                    break
                
                # Find unsatisfied clauses
                unsat_clauses = []
                for ci, c in enumerate(walk_clauses):
                    sat = False
                    for v, sign in c:
                        if assignment[v] == sign:
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    return assignment
                
                unsat_count = len(unsat_clauses)
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = dict(assignment)
                
                # Adaptive noise: higher when stuck
                # Noise parameter: probability of random flip
                noise = min(0.35, 0.05 + 0.02 * (step % 100) / 100)
                
                # Choose an unsatisfied clause (weighted by weight)
                if rng.random() < 0.7:
                    # Weighted selection: prefer clauses with more free literals
                    total_w = sum(clause_weights[ci] for ci in unsat_clauses)
                    r = rng.random() * total_w
                    cum = 0
                    chosen_ci = unsat_clauses[0]
                    for ci in unsat_clauses:
                        cum += clause_weights[ci]
                        if r <= cum:
                            chosen_ci = ci
                            break
                else:
                    chosen_ci = rng.choice(unsat_clauses)
                
                clause = walk_clauses[chosen_ci]
                
                if rng.random() < noise:
                    # Random flip
                    v, sign = rng.choice(clause)
                    assignment[v] = not assignment[v]
                else:
                    # Greedy: flip variable that satisfies most other unsat clauses
                    best_v = clause[0][0]
                    best_score = -1
                    for v, sign in clause:
                        # Score: how many other unsat clauses this flip would satisfy
                        score = 0
                        for ci2 in unsat_clauses:
                            if ci2 == chosen_ci:
                                continue
                            c2 = walk_clauses[ci2]
                            # Would flipping v satisfy c2?
                            for v2, sign2 in c2:
                                if v2 == v and sign2 != sign:
                                    # Currently v has value assignment[v], flipping changes it
                                    # After flip, assignment[v] becomes not assignment[v]
                                    # c2 is satisfied if not assignment[v] == sign2
                                    # i.e., assignment[v] != sign2
                                    if assignment[v] != sign2:
                                        score += 1
                                    break
                        score += len(clause) - 1  # Bonus for the chosen clause itself
                        if score > best_score:
                            best_score = score
                            best_v = v
                    assignment[best_v] = not assignment[best_v]
                
                # Update clause weights periodically
                if step % 100 == 0:
                    for ci in unsat_clauses:
                        clause_weights[ci] = min(10.0, clause_weights[ci] * 1.1)
                    # Decay others slightly
                    for ci in range(num_clauses):
                        if ci not in set(unsat_clauses):
                            clause_weights[ci] = max(1.0, clause_weights[ci] * 0.95)
            
            # Restart with perturbation if we had a good partial solution
            if best_assignment is not None and best_unsat_count < unsat_count:
                # Perturb around best
                assignment = dict(best_assignment)
                for _ in range(min(5, len(free_vars))):
                    v = rng.choice(free_vars)
                    assignment[v] = not assignment[v]
        
        return best_assignment if best_assignment else assignment
    
    # Run WalkSAT
    walk_result = walksat_solve(n, walk_clauses, free_vars, forced_assignment, deadline)
    
    # Combine forced and walk assignments
    result = [False] * n
    for v, val in forced_assignment.items():
        result[v] = val
    for v, val in walk_result.items():
        result[v] = val
    
    return result
# EVOLVE-BLOCK-END
