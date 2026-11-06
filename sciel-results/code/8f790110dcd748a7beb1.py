# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin for 300s limit
    
    m = len(clauses)
    
    # Preprocessing: Convert to 0-based and separate vars/signs
    vars_matrix = np.zeros((m, 3), dtype=np.int32)
    signs_matrix = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            vars_matrix[i, j] = abs(l) - 1
            signs_matrix[i, j] = 1 if l > 0 else 0

    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    # Also store the index of the variable in the clause for quick lookup
    var_index_in_clause = {} # (var, clause_id) -> index 0,1,2
    for i in range(m):
        for j in range(3):
            v = vars_matrix[i, j]
            var_to_clauses[v].append(i)
            var_index_in_clause[(v, i)] = j

    rng = random.Random(42)
    
    def init_clause_counts(assignment):
        """Initialize clause_sat_count array."""
        counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            sat = 0
            for j in range(3):
                if assignment[vars_matrix[i, j]] == signs_matrix[i, j]:
                    sat += 1
            counts[i] = sat
        return counts

    def get_unsat_clauses(clause_sat_count):
        return np.where(clause_sat_count == 0)[0].tolist()

    def score_flip(assignment, var_to_flip, clause_sat_count):
        """
        Calculate the change in unsatisfied clause count if we flip var_to_flip.
        Returns the delta (new_unsat - old_unsat).
        """
        delta = 0
        v = var_to_flip
        curr_val = assignment[v]
        
        for ci in var_to_clauses[v]:
            # Get the sign of v in this clause
            idx = var_index_in_clause[(v, ci)]
            sign = signs_matrix[ci, idx]
            
            # Check if v was satisfying the clause
            was_sat = (curr_val == sign)
            curr_count = clause_sat_count[ci]
            
            if was_sat:
                # Flipping makes it false. Count decreases.
                if curr_count == 1:
                    # Clause becomes unsat
                    delta += 1
            else:
                # Flipping makes it true. Count increases.
                if curr_count == 0:
                    # Clause becomes sat
                    delta -= 1
        return delta

    def perform_flip(assignment, var_to_flip, clause_sat_count):
        """Update assignment and clause_sat_count after flipping var_to_flip."""
        v = var_to_flip
        old_val = assignment[v]
        new_val = 1 - old_val
        assignment[v] = new_val
        
        for ci in var_to_clauses[v]:
            idx = var_index_in_clause[(v, ci)]
            sign = signs_matrix[ci, idx]
            
            if old_val == sign:
                # Was sat, now not sat
                clause_sat_count[ci] -= 1
            else:
                # Was not sat, now sat
                clause_sat_count[ci] += 1

    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 2000
    base_steps = 5000
    step_growth = 1.02
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = np.random.randint(0, 2, n, dtype=np.int32)
        clause_sat_count = init_clause_counts(assignment)
        
        unsat_count = int(np.sum(clause_sat_count == 0))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            
        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 100000)
        
        # Variable activities for heuristic
        var_activities = np.zeros(n, dtype=np.float64)
        
        T = 5.0
        cooling = 0.999
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            unsat_indices = get_unsat_clauses(clause_sat_count)
            
            if not unsat_indices:
                return assignment.tolist()
            
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            
            # Get the literals in this clause
            c_vars = vars_matrix[ci]
            c_signs = signs_matrix[ci]
            
            # Identify false literals in this clause
            false_vars = []
            for k in range(3):
                v = c_vars[k]
                s = c_signs[k]
                if assignment[v] != s:
                    false_vars.append(v)
            
            if not false_vars:
                continue
            
            # Calculate deltas
            deltas = []
            best_var = -1
            min_delta = float('inf')
            
            for v in false_vars:
                d = score_flip(assignment, v, clause_sat_count)
                deltas.append(d)
                if d < min_delta:
                    min_delta = d
                    best_var = v
            
            # Heuristic: Weighted random choice
            # Probability proportional to exp(-delta / T) * activity?
            # Let's stick to simple WalkSAT logic but with better scoring.
            
            p_random = 0.2
            if rng.random() < p_random:
                var_to_flip = rng.choice(false_vars)
            else:
                # Pick best, or accept worse with SA probability
                if min_delta > 0:
                    prob = np.exp(-min_delta / max(T, 0.1))
                    if rng.random() < prob:
                        var_to_flip = best_var
                    else:
                        var_to_flip = rng.choice(false_vars)
                else:
                    var_to_flip = best_var
            
            # Perform flip
            perform_flip(assignment, var_to_flip, clause_sat_count)
            
            # Update activity
            var_activities[var_to_flip] += 1.0
            
            # Cool down
            T *= cooling
            
        restart += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
