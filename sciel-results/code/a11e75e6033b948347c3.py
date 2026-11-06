# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _preprocess(n, clauses):
    """
    Apply pure literal elimination, unit propagation, and tautology removal.
    Returns (reduced_clauses, fixed_vars, remaining_vars, mapping) where:
      - reduced_clauses: list of clauses over remaining_vars (0-based indices into remaining_vars)
      - fixed_vars: dict var_idx -> bool for variables that got fixed
      - remaining_vars: list of variable indices (0-based) that are still free
      - mapping: dict old_var_idx -> new_idx (or -1 if fixed)
    """
    # Work with 0-based variable indices
    # fixed[var] = None means free, True/False means fixed
    fixed = [None] * n
    # active_clauses: list of sets of (var, sign) where sign is True for positive
    # We'll track which clauses are satisfied
    clauses_set = []
    for c in clauses:
        cl = set()
        for lit in c:
            v = abs(lit) - 1
            s = (lit > 0)
            cl.add((v, s))
        clauses_set.append(cl)
    
    changed = True
    while changed:
        changed = False
        
        # Remove tautologies (clause contains both x and not-x)
        new_clauses = []
        for cl in clauses_set:
            has_pos = set()
            has_neg = set()
            for (v, s) in cl:
                if s:
                    has_pos.add(v)
                else:
                    has_neg.add(v)
            if has_pos & has_neg:
                changed = True  # tautology removed, it's always satisfied
                continue
            new_clauses.append(cl)
        clauses_set = new_clauses
        
        # Pure literal elimination
        # For each variable, check if it appears only with one sign
        var_pos = [set() for _ in range(n)]
        var_neg = [set() for _ in range(n)]
        for i, cl in enumerate(clauses_set):
            for (v, s) in cl:
                if s:
                    var_pos[v].add(i)
                else:
                    var_neg[v].add(i)
        
        for v in range(n):
            if fixed[v] is not None:
                continue
            if len(var_pos[v]) == 0 and len(var_neg[v]) > 0:
                # Only appears negatively -> set to True to satisfy all
                fixed[v] = True
                changed = True
            elif len(var_neg[v]) == 0 and len(var_pos[v]) > 0:
                # Only appears positively -> set to False to satisfy all
                fixed[v] = False
                changed = True
        
        # Unit propagation
        for i, cl in enumerate(clauses_set):
            if len(cl) == 1:
                v, s = next(iter(cl))
                if fixed[v] is None:
                    fixed[v] = s
                    changed = True
                elif fixed[v] != s:
                    # Conflict: unit clause unsatisfied
                    # This means the instance is unsatisfiable given current assignments
                    # For planted SAT, this shouldn't happen if we're correct
                    # But we'll handle it by not fixing further
                    pass
        
        # Remove satisfied clauses and simplify
        new_clauses = []
        for cl in clauses_set:
            # Check if clause is satisfied by current fixed assignments
            sat = False
            new_cl = set()
            for (v, s) in cl:
                if fixed[v] is not None:
                    if fixed[v] == s:
                        sat = True
                        break
                    else:
                        # This literal is false, remove it
                        continue
                new_cl.add((v, s))
            if sat:
                changed = True
                continue
            if len(new_cl) == 0:
                # Empty clause -> unsatisfiable (shouldn't happen for planted SAT)
                continue
            if new_cl != cl:
                changed = True
            new_clauses.append(new_cl)
        clauses_set = new_clauses
    
    # Build mapping for remaining variables
    remaining_vars = [v for v in range(n) if fixed[v] is None]
    var_map = {}
    for new_idx, old_idx in enumerate(remaining_vars):
        var_map[old_idx] = new_idx
    
    # Remap clauses to new indices
    reduced_clauses = []
    for cl in clauses_set:
        new_cl = []
        for (v, s) in cl:
            if v in var_map:
                new_cl.append((var_map[v], s))
        if new_cl:
            reduced_clauses.append(new_cl)
    
    return reduced_clauses, fixed, remaining_vars, var_map


def _walksat_vectorized(n_vars, clauses, rng, deadline, max_restarts=50):
    """
    Clause-weighted WalkSAT with dynamic noise and restarts.
    clauses: list of lists of (var_idx, sign) tuples, 0-based
    Returns assignment as list of bools for n_vars variables.
    """
    if not clauses:
        return [True] * n_vars
    
    m = len(clauses)
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        if restart == 0:
            assignment = np.array([rng.random() < 0.5 for _ in range(n_vars)], dtype=bool)
        else:
            # Perturb from best known or random
            if best_assignment is not None:
                assignment = best_assignment.copy()
                # Flip some random variables
                n_flip = max(1, n_vars // 10)
                flip_vars = rng.sample(range(n_vars), min(n_flip, n_vars))
                assignment[flip_vars] = ~assignment[flip_vars]
            else:
                assignment = np.array([rng.random() < 0.5 for _ in range(n_vars)], dtype=bool)
        
        # Precompute clause structure for fast evaluation
        # For each clause, store vars and signs as arrays
        clause_vars = np.array([np.array([v for v, s in cl]) for cl in clauses], dtype=object)
        clause_signs = np.array([np.array([s for v, s in cl]) for cl in clauses], dtype=object)
        
        # Clause weights
        weights = np.ones(m, dtype=float)
        
        # Track unsatisfied clause count
        def count_unsat(assignment):
            unsat_count = 0
            for i in range(m):
                cl_vars = clause_vars[i]
                cl_signs = clause_signs[i]
                # Check if any literal is satisfied
                vals = assignment[cl_vars]
                # A literal (v, s) is satisfied if assignment[v] == s
                if np.any(vals == cl_signs):
                    continue
                unsat_count += 1
            return unsat_count
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        
        best_assignment = assignment.copy()
        best_unsat_count = unsat_count
        
        max_steps = 5000 + restart * 1000
        step = 0
        stuck_count = 0
        max_stuck = 500
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            # Recompute unsat clauses (vectorized-ish)
            unsat_indices = []
            for i in range(m):
                cl_vars = clause_vars[i]
                cl_signs = clause_signs[i]
                vals = assignment[cl_vars]
                if not np.any(vals == cl_signs):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment.tolist()
            
            unsat_count = len(unsat_indices)
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
                stuck_count = 0
            elif unsat_count == best_unsat_count:
                stuck_count += 1
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                break  # Restart
            
            # Dynamic noise: start high, decrease
            progress = step / max_steps
            noise = 0.4 * (1.0 - progress) + 0.1  # From 0.5 to 0.1
            
            # Select unsatisfied clause (weighted by clause weight)
            # Use weighted selection
            sel_weights = np.array([weights[i] for i in unsat_indices])
            sel_weights = sel_weights / sel_weights.sum()
            chosen_idx = rng.choice(len(unsat_indices), p=sel_weights)
            clause_idx = unsat_indices[chosen_idx]
            
            # Select variable to flip
            cl = clauses[clause_idx]
            if rng.random() < noise:
                # Random flip from clause
                v, s = rng.choice(cl)
            else:
                # Greedy: flip variable that minimizes unsat count
                best_var = cl[0][0]
                best_delta = float('inf')
                for v, s in cl:
                    # Count how many unsat clauses would be affected
                    # Simplified: count clauses containing this variable
                    delta = 0
                    for i in unsat_indices:
                        cl_i = clauses[i]
                        if any(vv == v for vv, ss in cl_i):
                            # Check if flipping v would satisfy or unsatisfy this clause
                            # Currently unsat: no literal satisfied
                            # If we flip v, check if (v, s_i) becomes satisfied
                            for vv, ss in cl_i:
                                if vv == v:
                                    # After flip, assignment[v] = not assignment[v]
                                    new_val = not assignment[v]
                                    if new_val == ss:
                                        delta -= 1  # This clause becomes satisfied
                                    break
                    # Also check clauses that are currently satisfied but contain v
                    # This is expensive; approximate by just considering unsat clauses
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                v = best_var
            
            assignment[v] = not assignment[v]
            
            # Update weights: increase weight of unsatisfied clauses
            # Simplified: just increment weights of currently unsat clauses
            # This is O(m) per step which is expensive; do it periodically
            if step % 100 == 0:
                for i in unsat_indices:
                    weights[i] += 0.1
                # Normalize occasionally
                if step % 1000 == 0:
                    weights = weights / (weights.sum() / m)
            
            step += 1
        
        # After exhausting steps for this restart, update best
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n_vars


def solve(instance):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit
    
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    # Preprocessing
    reduced_clauses, fixed, remaining_vars, var_map = _preprocess(n, clauses)
    
    if not remaining_vars:
        # All variables fixed
        assignment = [False] * n
        for v, val in fixed.items():
            if val is not None:
                assignment[v] = val
        return assignment
    
    # Check if reduced instance is empty (all clauses satisfied)
    if not reduced_clauses:
        assignment = [False] * n
        for v, val in fixed.items():
            if val is not None:
                assignment[v] = val
        return assignment
    
    # Run WalkSAT on reduced instance
    reduced_assignment = _walksat_vectorized(
        len(remaining_vars), reduced_clauses, rng, deadline
    )
    
    # Map back to original
    assignment = [False] * n
    for v, val in fixed.items():
        if val is not None:
            assignment[v] = val
    for new_idx, old_idx in enumerate(remaining_vars):
        assignment[old_idx] = reduced_assignment[new_idx]
    
    # Verify and if not satisfied, try a few more quick local search steps
    # to see if we can fix it
    def is_satisfied(assign):
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                s = (lit > 0)
                if assign[v] == s:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    if not is_satisfied(assignment) and time.time() < deadline:
        # Try quick local search on full instance
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Find unsatisfied clause
            unsat_clause = None
            for c in clauses:
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    s = (lit > 0)
                    if assignment[v] == s:
                        sat = True
                        break
                if not sat:
                    unsat_clause = c
                    break
            if unsat_clause is None:
                break
            # Flip a variable in the clause
            lit = rng.choice(unsat_clause)
            v = abs(lit) - 1
            assignment[v] = not assignment[v]
    
    return assignment
# EVOLVE-BLOCK-END
