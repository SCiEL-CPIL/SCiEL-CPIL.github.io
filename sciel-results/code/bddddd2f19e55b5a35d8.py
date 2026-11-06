"""Rewritten planted 3-SAT solver with preprocessing + clause-weighted WalkSAT + restarts."""
import random
import time


def _preprocess(n, clauses):
    """
    Apply unit propagation and pure literal elimination.
    Returns (assignment_list_or_None, remaining_clauses, fixed_values_dict).
    If conflict, returns (None, [], {}).
    """
    # assignment: list of -1 (unassigned), 0 (False), 1 (True)
    assign = [-1] * n
    # For each variable, track which clauses contain it (positive and negative)
    # We'll use a simpler approach: maintain clause list and variable->clauses map
    # Actually, for efficiency with 150 vars, let's do iterative simplification
    
    # Build var -> set of clause indices
    var_clauses = [[] for _ in range(n)]  # var_clauses[v] = list of (clause_idx, sign)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_clauses[v].append((ci, sign))
    
    # Work on a copy of clauses; mark clauses as satisfied/removed
    clause_satisfied = [False] * len(clauses)
    clause_removed = [False] * len(clauses)  # removed by unit propagation
    
    # Pure literal tracking: for each var, count positive and negative occurrences in unsatisfied, non-removed clauses
    def is_clause_active(ci):
        return not clause_satisfied[ci] and not clause_removed[ci]
    
    def is_lit_satisfied(lit):
        v = abs(lit) - 1
        return assign[v] != -1 and ((lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == 0))
    
    def is_clause_satisfied(ci):
        return any(is_lit_satisfied(lit) for lit in clauses[ci])
    
    def is_clause_unit(ci):
        """Returns the unassigned literal if clause is unit, None otherwise."""
        unassigned_lits = []
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if assign[v] == -1:
                unassigned_lits.append(lit)
        if len(unassigned_lits) == 1:
            return unassigned_lits[0]
        return None
    
    changed = True
    while changed:
        changed = False
        # Unit propagation
        for ci in range(len(clauses)):
            if clause_satisfied[ci] or clause_removed[ci]:
                continue
            # Check if satisfied
            if is_clause_satisfied(ci):
                clause_satisfied[ci] = True
                changed = True
                continue
            # Check if unit
            unit_lit = is_clause_unit(ci)
            if unit_lit is not None:
                v = abs(unit_lit) - 1
                val = 1 if unit_lit > 0 else 0
                if assign[v] != -1 and assign[v] != val:
                    return None, [], {}  # Conflict
                if assign[v] == -1:
                    assign[v] = val
                    clause_removed[ci] = True  # This clause is now satisfied by the assignment
                    clause_satisfied[ci] = True
                    changed = True
                    # Propagate: check other clauses containing this variable
                    # Actually, let's just loop again
                    break
        if changed:
            continue
        
        # Pure literal elimination
        # For each unassigned variable, check if all occurrences have the same sign
        for v in range(n):
            if assign[v] != -1:
                continue
            pos_found = False
            neg_found = False
            for ci, sign in var_clauses[v]:
                if clause_satisfied[ci] or clause_removed[ci]:
                    continue
                if sign == 1:
                    pos_found = True
                else:
                    neg_found = True
                if pos_found and neg_found:
                    break
            if pos_found and not neg_found:
                assign[v] = 1
                changed = True
            elif neg_found and not pos_found:
                assign[v] = 0
                changed = True
            # If neither, variable doesn't appear in any active clause - leave unassigned
    
    # Build remaining clauses (those not satisfied)
    remaining_clauses = []
    for ci in range(len(clauses)):
        if clause_satisfied[ci]:
            continue
        # Filter out assigned literals
        new_clause = []
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if assign[v] == -1:
                new_clause.append(lit)
        if len(new_clause) > 0:
            remaining_clauses.append(new_clause)
    
    return assign, remaining_clauses, {}


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin
    
    # Preprocessing
    assign, remaining_clauses, _ = _preprocess(n, clauses)
    
    # If all clauses are satisfied, return the assignment
    if not remaining_clauses:
        result = [False] * n
        for i in range(n):
            if assign[i] == 1:
                result[i] = True
            # If unassigned, default to False
        return result
    
    # Get list of free variables
    free_vars = [i for i in range(n) if assign[i] == -1]
    if not free_vars:
        # All assigned but clauses remain - should not happen if preprocessing is correct
        result = [False] * n
        for i in range(n):
            if assign[i] == 1:
                result[i] = True
        return result
    
    # Map free var index to position in free_vars
    var_to_free = {v: i for i, v in enumerate(free_vars)}
    num_free = len(free_vars)
    
    # Convert remaining clauses to use free variable indices (0-based)
    # Each clause becomes a list of (free_var_idx, sign) where sign is 1 for positive, 0 for negative
    # But we need to keep original literal signs for evaluation
    # Actually, let's just work with the original assignment array and track which vars are free
    
    # WalkSAT on free variables
    # assignment: dict from var index to bool
    # We'll use a list of length n with None for unassigned, but we already have assign from preprocessing
    
    # Let's use a separate array for the SAT search
    # sat_assign: list of bool for free vars, indexed by position in free_vars
    # We'll maintain a full assignment list of length n
    
    def build_full_assignment(sat_assign):
        """Build full assignment from preprocessing + sat_assign."""
        full = [False] * n
        for i in range(n):
            if assign[i] == 1:
                full[i] = True
            elif assign[i] == 0:
                full[i] = False
            else:
                # Free variable
                idx = var_to_free[i]
                full[i] = sat_assign[idx]
        return full
    
    def is_clause_satisfied_full(clause, full_assign):
        return any(full_assign[abs(l) - 1] == (l > 0) for l in clause)
    
    def count_unsat_full(full_assign):
        return sum(1 for c in remaining_clauses if not is_clause_satisfied_full(c, full_assign))
    
    # Initialize with random assignment for free vars
    rng = random.Random(0)
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 50
    max_flips_per_restart = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        sat_assign = [rng.random() < 0.5 for _ in range(num_free)]
        full_assign = build_full_assignment(sat_assign)
        cur_unsat = count_unsat_full(full_assign)
        
        if cur_unsat == 0:
            return full_assign
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = full_assign[:]
        
        # Clause weights for weighted WalkSAT
        # Initialize weights based on clause frequency in unsat clauses
        clause_weights = [1.0] * len(remaining_clauses)
        
        # Compute initial weights
        unsat_clauses = [i for i, c in enumerate(remaining_clauses) if not is_clause_satisfied_full(c, full_assign)]
        for ci in unsat_clauses:
            clause_weights[ci] = 2.0
        
        noise = 0.25
        for flip in range(max_flips_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsat clauses
            unsat_clauses = [i for i, c in enumerate(remaining_clauses) if not is_clause_satisfied_full(c, full_assign)]
            if not unsat_clauses:
                return full_assign
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assign = full_assign[:]
            
            # Weighted selection of clause
            total_weight = sum(clause_weights[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_clause_idx = unsat_clauses[0]
            for ci in unsat_clauses:
                cumulative += clause_weights[ci]
                if r <= cumulative:
                    chosen_clause_idx = ci
                    break
            
            clause = remaining_clauses[chosen_clause_idx]
            
            # Find free variables in this clause
            free_lits = []
            for lit in clause:
                v = abs(lit) - 1
                if v in var_to_free:
                    free_lits.append((lit, var_to_free[v]))
            
            if not free_lits:
                # No free variables in this clause - shouldn't happen
                continue
            
            # Evaluate damage for each free literal
            best_damage = float('inf')
            best_lit = free_lits[0][0]
            best_free_idx = free_lits[0][1]
            
            for lit, free_idx in free_lits:
                # Compute damage: how many currently satisfied clauses would become unsatisfied
                damage = 0
                v = abs(lit) - 1
                current_val = full_assign[v]
                new_val = not current_val
                
                for ci, c in enumerate(remaining_clauses):
                    if is_clause_satisfied_full(c, full_assign):
                        # Check if this clause contains the variable
                        # And if flipping would make it unsatisfied
                        contains_var = False
                        all_other_satisfied = True
                        for l in c:
                            if abs(l) - 1 == v:
                                contains_var = True
                                break
                        if contains_var:
                            # Check if all other literals in clause are satisfied
                            for l in c:
                                if abs(l) - 1 == v:
                                    continue
                                if not (full_assign[abs(l) - 1] == (l > 0)):
                                    all_other_satisfied = False
                                    break
                            if all_other_satisfied:
                                damage += clause_weights[ci]
                
                # Add noise
                damage += rng.random() * noise * 10
                
                if damage < best_damage:
                    best_damage = damage
                    best_lit = lit
                    best_free_idx = free_idx
            
            # Flip the variable
            v = abs(best_lit) - 1
            full_assign[v] = not full_assign[v]
            sat_assign[best_free_idx] = full_assign[v]
            
            # Update clause weights (simple: increase weight of unsat clauses)
            # For efficiency, just periodically recompute
            if flip % 100 == 0:
                for ci in range(len(remaining_clauses)):
                    if is_clause_satisfied_full(remaining_clauses[ci], full_assign):
                        clause_weights[ci] = 1.0
                    else:
                        clause_weights[ci] = 2.0
        
        # After restart, if no improvement, increase noise
        noise = min(noise * 1.2, 0.5)
    
    if best_assign is not None:
        return best_assign
    
    # Fallback: return current or default
    return build_full_assignment([False] * num_free)
