"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0  # 15s margin before 300s limit
    
    # Convert to 0-based internal representation
    # Each clause is a list of (var, sign) where sign is +1 or -1
    # var is 0-based
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    def is_satisfied(assignment):
        for clause in clauses_internal:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # DPLL with VSIDS and unit propagation
    # State: assignment (list of None/True/False), trail for backtracking
    
    def dpll(assignment, decision_level, trail):
        """
        assignment: list of length n, each element is None (unassigned), True, or False
        decision_level: current decision level
        trail: list of (var, value, level) for backtracking
        Returns: assignment if SAT, None if UNSAT
        """
        # Unit propagation
        # Find all unit clauses
        while True:
            propagated = False
            # Check for conflicts and unit clauses
            # For efficiency, we track clause satisfaction counts
            # But for simplicity in this implementation, we scan clauses
            
            # Actually, let's use a more efficient approach with clause state tracking
            # For now, simple approach
            unit_vars = []
            conflict = False
            
            for ci, clause in enumerate(clauses_internal):
                unassigned = []
                for var, sign in clause:
                    if assignment[var] is None:
                        unassigned.append((var, sign))
                    elif assignment[var] == (sign == 1):
                        break  # clause satisfied
                else:
                    # Clause not satisfied
                    if len(unassigned) == 0:
                        return None  # conflict
                    elif len(unassigned) == 1:
                        unit_vars.append(unassigned[0])
            
            if not unit_vars:
                break
            
            for var, sign in unit_vars:
                val = (sign == 1)
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return None  # conflict
                    continue
                assignment[var] = val
                trail.append((var, val, decision_level))
                propagated = True
            
            if not propagated:
                break
        
        # Check if all assigned
        all_assigned = True
        for var in range(n):
            if assignment[var] is None:
                all_assigned = False
                break
        
        if all_assigned:
            return assignment
        
        # VSIDS variable selection
        # Score variables by how many unsatisfied clauses they appear in
        # We need to track scores. For simplicity, compute on the fly
        # but this is expensive. Let's use a simpler heuristic: pick unassigned var
        # that appears in the most unsatisfied clauses
        
        # Count appearances in unsatisfied clauses
        var_scores = [0] * n
        for clause in clauses_internal:
            # Check if clause is currently unsatisfied (all literals false or unassigned)
            clause_unsat = True
            for var, sign in clause:
                if assignment[var] is not None and assignment[var] == (sign == 1):
                    clause_unsat = False
                    break
            if clause_unsat:
                for var, sign in clause:
                    if assignment[var] is None:
                        var_scores[var] += 1
        
        # Pick variable with highest score
        best_var = -1
        best_score = -1
        for var in range(n):
            if assignment[var] is None and var_scores[var] > best_score:
                best_score = var_scores[var]
                best_var = var
        
        if best_var == -1:
            # All assigned but not satisfied? Shouldn't happen
            return None
        
        # Decide: try True first
        assignment[best_var] = True
        trail.append((best_var, True, decision_level))
        
        result = dpll(assignment, decision_level + 1, trail)
        if result is not None:
            return result
        
        # Backtrack
        while trail and trail[-1][2] > decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        if trail and trail[-1][2] == decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        
        # Try False
        assignment[best_var] = False
        trail.append((best_var, False, decision_level))
        
        result = dpll(assignment, decision_level + 1, trail)
        if result is not None:
            return result
        
        # Backtrack again
        while trail and trail[-1][2] > decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        if trail and trail[-1][2] == decision_level:
            v, val, lvl = trail.pop()
            assignment[v] = None
        
        return None
    
    # Try DPLL first
    assignment = [None] * n
    trail = []
    result = dpll(assignment, 0, trail)
    
    if result is not None:
        return [bool(x) for x in result]
    
    # Check deadline
    if time.time() > deadline - 5:
        # Return random assignment as fallback
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
    
    # Fallback: WalkSAT
    rng = random.Random(42)
    w_assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied_clause(clause, assign):
        for var, sign in clause:
            if assign[var] == (sign == 1):
                return True
        return False
    
    steps = 0
    max_steps = 50000
    while steps < max_steps and time.time() < deadline - 2:
        unsat_clauses = []
        for clause in clauses_internal:
            if not satisfied_clause(clause, w_assignment):
                unsat_clauses.append(clause)
        
        if not unsat_clauses:
            return w_assignment
        
        # Pick clause with minimum unsatisfied literals
        best_clause = None
        best_count = float('inf')
        for clause in unsat_clauses:
            count = 0
            for var, sign in clause:
                if w_assignment[var] != (sign == 1):
                    count += 1
            if count < best_count:
                best_count = count
                best_clause = clause
        
        # Flip a literal in the clause
        if best_clause is not None:
            # Try to find a flip that reduces total unsatisfied count
            best_var = None
            best_delta = float('inf')
            for var, sign in best_clause:
                if w_assignment[var] == (sign == 1):
                    continue  # already satisfied
                # Count impact of flipping var
                delta = 0
                for ci, s in var_to_clauses[var]:
                    clause = clauses_internal[ci]
                    # Was this clause satisfied before?
                    was_sat = False
                    for v, sg in clause:
                        if w_assignment[v] == (sg == 1):
                            was_sat = True
                            break
                    # Will it be satisfied after?
                    would_be_sat = (s == 1)  # flipping var to satisfy this literal
                    # Wait, we're flipping var, so the literal (var, s) becomes true
                    # if we set var = (s == 1)
                    new_val = (s == 1)
                    would_be_sat = False
                    for v, sg in clause:
                        if v == var:
                            if new_val == (sg == 1):
                                would_be_sat = True
                                break
                        elif w_assignment[v] == (sg == 1):
                            would_be_sat = True
                            break
                    if was_sat and not would_be_sat:
                        delta += 1
                    elif not was_sat and would_be_sat:
                        delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_var = var
            
            if best_var is not None:
                w_assignment[best_var] = not w_assignment[best_var]
        
        steps += 1
    
    # Return best found
    return w_assignment


# EVOLVE-BLOCK-END
