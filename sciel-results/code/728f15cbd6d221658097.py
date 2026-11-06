"""Improved 3-SAT solver using optimized WalkSAT with incremental tracking and restarts."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess clauses: convert to 0-based
    # Store as (var_index, sign) where sign is +1 or -1
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        processed_clauses.append(processed)
    
    # Build variable-to-clause mapping for efficient gain computation
    # var_to_clauses[var] = list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause literal arrays for faster satisfaction checks
    clause_vars = []
    clause_signs = []
    for clause in processed_clauses:
        vs = [v for v, s in clause]
        ss = [s for v, s in clause]
        clause_vars.append(vs)
        clause_signs.append(ss)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    rng = random.Random(42)
    
    def lit_satisfied(var, sign, assignment):
        if sign == 1:
            return assignment[var]
        else:
            return not assignment[var]
    
    def init_tracking(assignment):
        """Initialize clause satisfaction tracking arrays."""
        clause_sat_count = [0] * num_clauses
        for ci in range(num_clauses):
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            count = 0
            for i in range(len(vs)):
                if lit_satisfied(vs[i], ss[i], assignment):
                    count += 1
            clause_sat_count[ci] = count
        unsat_count = sum(1 for c in clause_sat_count if c == 0)
        return clause_sat_count, unsat_count
    
    def flip_var(assignment, var, clause_sat_count, unsat_count):
        """Flip a variable and update tracking incrementally. Returns new unsat_count."""
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        for ci, sign in var_to_clauses[var]:
            old_sat = lit_satisfied(var, sign, assignment) if old_val else not lit_satisfied(var, sign, assignment)
            # Actually, let's compute before and after properly
            # Before flip:
            if sign == 1:
                before_sat = old_val
            else:
                before_sat = not old_val
            
            # After flip:
            if sign == 1:
                after_sat = new_val
            else:
                after_sat = not new_val
            
            if before_sat and not after_sat:
                clause_sat_count[ci] -= 1
                if clause_sat_count[ci] == 0:
                    unsat_count += 1
            elif not before_sat and after_sat:
                clause_sat_count[ci] += 1
                if clause_sat_count[ci] == 1:
                    unsat_count -= 1
        return unsat_count
    
    def get_gain_fast(assignment, var, clause_sat_count):
        """Compute gain of flipping var using precomputed clause_sat_count."""
        gain = 0
        for ci, sign in var_to_clauses[var]:
            if clause_sat_count[ci] > 1:
                continue
            if clause_sat_count[ci] == 1:
                # Check if var is the only satisfying literal
                if sign == 1 and assignment[var]:
                    gain -= 1
                elif sign == -1 and not assignment[var]:
                    gain -= 1
            else:  # clause_sat_count[ci] == 0
                # Clause is unsatisfied, flipping var to satisfy it helps
                if sign == 1 and not assignment[var]:
                    gain += 1
                elif sign == -1 and assignment[var]:
                    gain += 1
        return gain
    
    # WalkSAT with restarts and adaptive step budget
    max_restarts = 200
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat_count, unsat_count = init_tracking(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive steps: more steps when unsat is low
        if unsat_count <= 10:
            steps_per_restart = 20000
        elif unsat_count <= 50:
            steps_per_restart = 10000
        else:
            steps_per_restart = 5000
        
        # WalkSAT steps with incremental tracking
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Find unsatisfied clauses (using tracking)
            unsat_clauses = [ci for ci in range(num_clauses) if clause_sat_count[ci] == 0]
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = processed_clauses[ci]
            
            # Choose variable to flip: with probability p, use gain; otherwise random
            if rng.random() < 0.85:  # 85% use gain-based
                best_var = None
                best_gain = -float('inf')
                for var, sign in clause:
                    gain = get_gain_fast(assignment, var, clause_sat_count)
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    unsat_count = flip_var(assignment, best_var, clause_sat_count, unsat_count)
            else:
                var, sign = rng.choice(clause)
                unsat_count = flip_var(assignment, var, clause_sat_count, unsat_count)
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
