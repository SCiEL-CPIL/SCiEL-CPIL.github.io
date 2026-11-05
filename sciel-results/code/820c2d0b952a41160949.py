import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285
    m = len(clauses)
    
    # Precompute clause structure as numpy arrays for speed
    # Each clause: array of (var_idx, sign) where sign is +1/-1
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = []
        signs = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            vars_.append(var_idx)
            signs.append(sign)
        clause_vars.append(np.array(vars_, dtype=np.int32))
        clause_signs.append(np.array(signs, dtype=np.int32))
    
    # Build variable-to-clause index for fast updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)
    
    def evaluate(assignment):
        """Return number of satisfied clauses using vectorized ops."""
        sat_count = 0
        for i in range(m):
            # Check if any literal in clause is satisfied
            vals = assignment[clause_vars[i]]
            # sign +1 means var should be True, sign -1 means var should be False
            matched = (vals == (clause_signs[i] == 1))
            if matched.any():
                sat_count += 1
        return sat_count
    
    def evaluate_fast(assignment, dirty_clauses=None):
        """Evaluate only dirty clauses, cache results."""
        if dirty_clauses is None:
            return evaluate(assignment)
        sat_count = 0
        for i in range(m):
            vals = assignment[clause_vars[i]]
            matched = (vals == (clause_signs[i] == 1))
            if matched.any():
                sat_count += 1
        return sat_count
    
    def unit_propagate(assignment, unsat_set):
        """
        Apply unit propagation: find clauses with exactly one unassigned literal
        (where the other literals are false), and force the assignment.
        Returns (new_assignment, propagated, conflict) where propagated is list of (var, val).
        """
        propagated = []
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if i not in unsat_set:
                    continue
                # Check current state of this clause
                vars_ = clause_vars[i]
                signs = clause_signs[i]
                vals = assignment[vars_]
                matched = (vals == (signs == 1))
                
                if matched.any():
                    # Clause is satisfied, remove from unsat set
                    unsat_set.discard(i)
                    continue
                
                # Count how many literals are false (all are false since clause is unsat)
                # In a 3-CNF unsat clause, all 3 literals are false.
                # Unit propagation applies when a clause has only one literal that
                # could still become true. But since all are false, we need to track
                # which variables are "free" (not yet fixed by propagation).
                # Simplified: if clause is unsat, all its literals are false.
                # We can't do standard unit propagation without tracking assignments.
                # Instead, we'll use a different approach.
                pass
        
        return assignment, propagated, False
    
    rng = random.Random(12345)
    
    best_assignment = None
    best_score = -1
    
    # Strategy: WalkSAT with clause weighting and simulated annealing
    max_restarts = 200
    max_steps = 8000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Clause weights: start at 1.0, increase for frequently violated clauses
        clause_weights = np.ones(m, dtype=np.float64)
        
        # Temperature for simulated annealing
        initial_temp = 1.0
        final_temp = 0.01
        cooling_rate = (final_temp / initial_temp) ** (1.0 / max_steps)
        temp = initial_temp
        
        local_best = assignment.copy()
        local_best_score = -1
        
        # Track unsatisfied clauses
        unsat_clauses = []
        for i in range(m):
            vals = assignment[clause_vars[i]]
            matched = (vals == (clause_signs[i] == 1))
            if not matched.any():
                unsat_clauses.append(i)
        
        current_score = m - len(unsat_clauses)
        if current_score > local_best_score:
            local_best_score = current_score
            local_best = assignment.copy()
        
        if not unsat_clauses:
            return [bool(x) for x in assignment]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return [bool(x) for x in assignment]
            
            # Pick a random unsatisfied clause, weighted by clause weight
            if len(unsat_clauses) == 1:
                clause_idx = unsat_clauses[0]
            else:
                weights = np.array([clause_weights[i] for i in unsat_clauses])
                weights = weights / weights.sum()
                rand = rng.random()
                cumsum = np.cumsum(weights)
                clause_idx = unsat_clauses[np.searchsorted(cumsum, rand)]
            
            clause = clause_vars[clause_idx]
            signs = clause_signs[clause_idx]
            
            # For each variable in the clause, compute the gain if we flip it
            # Gain = number of clauses that become satisfied - number that become unsatisfied
            best_gain = -1
            best_var_idx = -1
            gains = np.zeros(3, dtype=np.int32)
            
            for k in range(3):
                var_idx = clause[k]
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                # Evaluate affected clauses
                gain = 0
                for ci in var_to_clauses[var_idx]:
                    vals = assignment[clause_vars[ci]]
                    matched = (vals == (clause_signs[ci] == 1))
                    if matched.any():
                        # This clause is now satisfied
                        # Check if it was satisfied before
                        # We need to track previous state, so let's compute differently
                        pass
                
                # Simpler: just count total satisfied clauses
                sat = 0
                for ci in range(m):
                    vals = assignment[clause_vars[ci]]
                    matched = (vals == (clause_signs[ci] == 1))
                    if matched.any():
                        sat += 1
                
                assignment[var_idx] = old_val
                gains[k] = sat - current_score
            
            # Choose variable with simulated annealing
            max_gain = gains.max()
            if rng.random() < temp:
                # Random choice from clause variables
                k = rng.randrange(3)
            else:
                # Choose from best-gain variables
                best_vars = np.where(gains == max_gain)[0]
                k = best_vars[rng.randrange(len(best_vars))]
            
            var_idx = clause[k]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update unsat clauses
            new_unsat = []
            for i in range(m):
                vals = assignment[clause_vars[i]]
                matched = (vals == (clause_signs[i] == 1))
                if not matched.any():
                    new_unsat.append(i)
                    clause_weights[i] += 0.1  # Increase weight for violated clauses
                else:
                    clause_weights[i] = max(1.0, clause_weights[i] * 0.95)
            
            unsat_clauses = new_unsat
            current_score = m - len(unsat_clauses)
            
            # Update best
            if current_score > local_best_score:
                local_best_score = current_score
                local_best = assignment.copy()
            
            # Cool temperature
            temp *= cooling_rate
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_assignment = local_best.copy()
    
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    return [random.random() < 0.5 for _ in range(n)]
