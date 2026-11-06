# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based internal representation
    # literals: list of lists of (var_idx, sign) where sign is +1 or -1
    # var_idx is 0-based
    internal_clauses = []
    for clause in clauses:
        ic = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            ic.append((var, sign))
        internal_clauses.append(ic)
    
    m = len(internal_clauses)
    if m == 0:
        return [True] * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # --- Preprocessing: Pure Literal Elimination and Unit Propagation ---
    # We'll work with a mutable structure
    # For efficiency, let's use arrays
    # clause_vars[i] = list of var indices in clause i
    # clause_signs[i] = list of signs in clause i
    # But we need to track which clauses are still active
    
    # Let's do a simple preprocessing pass
    # Track literal occurrences
    # pos_occurs[var] = set of clause indices where var appears positively
    # neg_occurs[var] = set of clause indices where var appears negatively
    
    # Actually, for a 150-variable problem, let's just do a few rounds of
    # pure literal elimination and unit propagation on the original clauses
    
    assignment = [None] * n  # None means unassigned
    active_clauses = list(range(m))
    
    # Build occurrence maps
    # For each variable, track which clauses contain it positively and negatively
    var_pos = [set() for _ in range(n)]
    var_neg = [set() for _ in range(n)]
    for ci, clause in enumerate(internal_clauses):
        for var, sign in clause:
            if sign > 0:
                var_pos[var].add(ci)
            else:
                var_neg[var].add(ci)
    
    # Unit propagation and pure literal elimination
    changed = True
    while changed:
        changed = False
        
        # Pure literal elimination
        for var in range(n):
            if assignment[var] is not None:
                continue
            if len(var_pos[var]) == 0 and len(var_neg[var]) > 0:
                # Only appears negatively, set to False
                assignment[var] = False
                changed = True
            elif len(var_neg[var]) == 0 and len(var_pos[var]) > 0:
                # Only appears positively, set to True
                assignment[var] = True
                changed = True
        
        if time.time() > deadline:
            break
        
        # Unit propagation: find clauses with only one unassigned literal
        for ci in active_clauses:
            if ci >= m:
                continue
            unassigned = [(var, sign) for var, sign in internal_clauses[ci] 
                         if assignment[var] is None]
            if len(unassigned) == 1:
                var, sign = unassigned[0]
                assignment[var] = (sign > 0)
                changed = True
            elif len(unassigned) == 0:
                # Clause is already satisfied or falsified
                # Check if satisfied
                is_sat = any(assignment[var] == (sign > 0) 
                            for var, sign in internal_clauses[ci] 
                            if assignment[var] is not None)
                if not is_sat:
                    # Contradiction - shouldn't happen for SAT instances
                    pass
    
    # Build reduced instance
    # Assignments for unassigned variables will be determined by WalkSAT
    # Create a list of variables that are still unassigned
    unassigned_vars = [i for i in range(n) if assignment[i] is None]
    var_to_idx = {v: i for i, v in enumerate(unassigned_vars)}
    num_unassigned = len(unassigned_vars)
    
    # Build reduced clauses (only contain unassigned variables)
    reduced_clauses = []
    for ci in range(m):
        rc = []
        is_sat = False
        for var, sign in internal_clauses[ci]:
            if assignment[var] is not None:
                if assignment[var] == (sign > 0):
                    is_sat = True
                    break
            else:
                rc.append((var_to_idx[var], sign))
        if not is_sat:
            # Remove duplicates (shouldn't happen but just in case)
            reduced_clauses.append(rc)
    
    # If all variables are assigned, check if solution is valid
    if num_unassigned == 0:
        # Verify
        for clause in internal_clauses:
            if not any(assignment[var] == (sign > 0) for var, sign in clause):
                # Not satisfied, but shouldn't happen
                pass
        return [assignment[i] if assignment[i] is not None else True for i in range(n)]
    
    # If no reduced clauses, any assignment works
    if len(reduced_clauses) == 0:
        for v in unassigned_vars:
            assignment[v] = True
        return assignment
    
    # --- WalkSAT on reduced instance ---
    # We'll use a clause-weighted WalkSAT with dynamic noise scheduling
    
    # Map back to original variable indices for the reduced problem
    # reduced_var_idx -> original_var_idx
    idx_to_var = unassigned_vars
    
    # Initialize assignment for reduced variables
    rng = random.Random(42)
    red_assignment = [rng.random() < 0.5 for _ in range(num_unassigned)]
    
    # Precompute for each variable in reduced instance, which clauses it appears in
    # var_clause_map[v] = list of clause indices in reduced_clauses
    var_clause_map = [[] for _ in range(num_unassigned)]
    for ci, clause in enumerate(reduced_clauses):
        for v, sign in clause:
            var_clause_map[v].append(ci)
    
    def is_clause_satisfied(ci):
        clause = reduced_clauses[ci]
        for v, sign in clause:
            if red_assignment[v] == (sign > 0):
                return True
        return False
    
    def clause_falsity(ci):
        """Return number of falsified literals in clause ci"""
        clause = reduced_clauses[ci]
        count = 0
        for v, sign in clause:
            if red_assignment[v] != (sign > 0):
                count += 1
        return count
    
    # Initial unsat clauses
    unsat_clauses = [ci for ci in range(len(reduced_clauses)) if not is_clause_satisfied(ci)]
    
    if not unsat_clauses:
        # Already satisfied
        for i, v in enumerate(unassigned_vars):
            assignment[v] = red_assignment[i]
        return assignment
    
    # WalkSAT with clause weighting and dynamic noise
    # Weight for each clause
    clause_weights = np.ones(len(reduced_clauses))
    
    # Noise schedule: start with high noise, decrease over time
    max_steps = 50000
    step = 0
    
    while unsat_clauses and step < max_steps:
        if time.time() > deadline:
            break
        
        # Select an unsatisfied clause (weighted)
        # Use weighted random selection
        weights = clause_weights[unsat_clauses]
        total_w = weights.sum()
        if total_w == 0:
            # Uniform selection
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
        else:
            probs = weights / total_w
            # Cumulative sum for selection
            cumsum = np.cumsum(probs)
            r = rng.random()
            idx = np.searchsorted(cumsum, r)
            idx = min(idx, len(unsat_clauses) - 1)
            ci = unsat_clauses[idx]
        
        clause = reduced_clauses[ci]
        
        # Dynamic noise: p decreases as we progress
        # p = p_start * (1 - step/max_steps) + p_end
        p_start = 0.5
        p_end = 0.1
        p = p_start * (1.0 - step / max_steps) + p_end
        
        if rng.random() < p:
            # Random flip
            v, sign = clause[rng.randrange(len(clause))]
        else:
            # Choose literal that maximizes gain (minimizes falsified literals in other clauses)
            best_v = -1
            best_score = float('inf')
            for v, sign in clause:
                # Calculate score: number of currently satisfied clauses that would become unsatisfied
                # minus number of currently unsatisfied clauses that would become satisfied
                score = 0
                for cj in var_clause_map[v]:
                    if cj == ci:
                        continue
                    # Check if cj is currently satisfied
                    if is_clause_satisfied(cj):
                        # If we flip v, does cj become unsatisfied?
                        # cj is satisfied by v currently
                        clause_cj = reduced_clauses[cj]
                        # Check if v is the only satisfying literal in cj
                        only_sat = True
                        for v2, sign2 in clause_cj:
                            if v2 != v and red_assignment[v2] == (sign2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            score += 1
                    else:
                        # cj is unsatisfied. If we flip v, does cj become satisfied?
                        # v is currently falsifying cj
                        # After flip, v will satisfy cj
                        # But cj might still be unsatisfied if other literals are also falsified
                        # Actually, if v satisfies cj after flip, cj becomes satisfied
                        # unless another literal in cj was already satisfying it (but cj is unsatisfied, so none are)
                        # So flipping v will satisfy cj
                        score -= 1
                if score < best_score:
                    best_score = score
                    best_v = v
            v = best_v
        
        # Flip variable v
        red_assignment[v] = not red_assignment[v]
        
        # Update unsat clauses
        # Clauses containing v may have changed status
        # Rebuild unsat list (could be optimized but for 150 vars it's fine)
        new_unsat = []
        for cj in range(len(reduced_clauses)):
            if not is_clause_satisfied(cj):
                new_unsat.append(cj)
                # Increase weight for unsatisfied clauses
                clause_weights[cj] *= 1.01
            else:
                # Decay weight for satisfied clauses
                clause_weights[cj] *= 0.99
        unsat_clauses = new_unsat
        
        step += 1
    
    # If we found a solution, great. Otherwise, return best effort
    for i, v in enumerate(unassigned_vars):
        assignment[v] = red_assignment[i]
    
    return assignment
# EVOLVE-BLOCK-END
