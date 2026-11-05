"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def solve(instance):
    """
    Solve a 3-SAT instance using a hybrid approach:
    1. DPLL with watched literals, unit propagation, and pure literal elimination
    2. If DPLL hits a timeout or fails, fall back to WalkSAT
    """
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 20s safety margin before 300s limit
    
    # Try DPLL first
    result = _dpll_solve(n, clauses, deadline)
    if result is not None:
        return result
    
    # Fall back to WalkSAT if DPLL didn't find a solution in time
    return _walksat_solve(n, clauses, deadline)


def _dpll_solve(n, clauses, deadline):
    """
    DPLL with watched literals, unit propagation, and pure literal elimination.
    Returns a satisfying assignment or None if timeout/no solution found.
    """
    # Build clause list with watched literals
    # Each clause is a list of signed literals (1-based)
    # Watched literals: for each variable, track which clauses watch its True and False values
    
    # Initialize assignment: -1 means unassigned, 0 means False, 1 means True
    assignment = [-1] * n
    
    # For each variable, maintain lists of clause indices that watch True and False
    # watch_true[var] = list of clause indices where the var appears as a positive literal
    # watch_false[var] = list of clause indices where the var appears as a negative literal
    # But we need to track which specific literal in the clause is being watched
    
    # Better approach: for each clause, track two watched literals
    # watched[i] = (lit1, lit2) for clause i
    # For each variable, maintain a set of (clause_idx, lit_position) that watch it
    
    # Let's use a simpler but effective approach:
    # Maintain for each variable a count of how many clauses it appears in (for pure literal)
    # And for unit propagation, we'll scan clauses
    
    # Actually, let's implement a clean DPLL with a queue for unit propagation
    
    # Preprocess: remove tautologies and duplicate literals in clauses
    processed_clauses = []
    for clause in clauses:
        # Remove duplicate literals (keep one)
        seen = set()
        new_clause = []
        for lit in clause:
            abs_lit = abs(lit)
            if abs_lit not in seen:
                seen.add(abs_lit)
                new_clause.append(lit)
        # Check for tautology (both x and -x present)
        pos_vars = set()
        neg_vars = set()
        for lit in new_clause:
            if lit > 0:
                pos_vars.add(lit)
            else:
                neg_vars.add(-lit)
        if pos_vars & neg_vars:
            continue  # Tautology, skip
        if not new_clause:
            return None  # Empty clause, unsatisfiable
        processed_clauses.append(new_clause)
    
    # If no clauses, any assignment works
    if not processed_clauses:
        return [False] * n
    
    # DPLL recursive solver with iterative deepening on decision limit
    # Use a stack-based approach to avoid recursion depth issues
    
    # State: assignment, list of active clauses (indices into processed_clauses)
    # We'll use a worklist for unit propagation
    
    def _dpll_recursive(assign, active_clauses, depth):
        """
        Recursive DPLL. assign is a list of -1/0/1.
        active_clauses is a list of clause indices that are still active.
        Returns assignment list if SAT, None if UNSAT or timeout.
        """
        if time.time() > deadline:
            return None
        
        # Unit propagation and pure literal elimination
        assign, active_clauses, success = _propagate(assign, active_clauses, processed_clauses)
        if not success:
            return None  # Conflict
        
        if not active_clauses:
            return assign  # All clauses satisfied
        
        # Pure literal elimination
        assign, active_clauses = _pure_literal_eliminate(assign, active_clauses, processed_clauses, n)
        
        # Check if all clauses are satisfied after pure literal elimination
        remaining = []
        for ci in active_clauses:
            if not _is_satisfied(processed_clauses[ci], assign):
                remaining.append(ci)
        active_clauses = remaining
        
        if not active_clauses:
            return assign
        
        # Variable selection: pick variable appearing in most active clauses
        var_counts = [0] * n
        for ci in active_clauses:
            for lit in processed_clauses[ci]:
                var_counts[abs(lit) - 1] += 1
        
        # Pick the variable with highest count among unassigned
        best_var = -1
        best_count = 0
        for v in range(n):
            if assign[v] == -1 and var_counts[v] > best_count:
                best_count = var_counts[v]
                best_var = v
        
        if best_var == -1:
            # All variables assigned but some clauses unsatisfied? Shouldn't happen
            return None
        
        # Try True first, then False
        for value in [1, 0]:
            if time.time() > deadline:
                return None
            
            new_assign = assign[:]
            new_assign[best_var] = value
            
            # Filter clauses: keep those that are not falsified
            new_active = []
            for ci in active_clauses:
                clause = processed_clauses[ci]
                # Check if clause is falsified by this assignment
                falsified = True
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    if new_assign[v] == -1:
                        falsified = False
                        break
                    if new_assign[v] == (1 if lit > 0 else 0):
                        satisfied = True
                        break
                if satisfied:
                    continue  # Clause is satisfied, remove it
                if not falsified:
                    new_active.append(ci)
            
            if not new_active:
                return new_assign
            
            result = _dpll_recursive(new_assign, new_active, depth + 1)
            if result is not None:
                return result
        
        return None  # Both branches failed
    
    def _propagate(assign, active_clauses, clauses_list):
        """
        Unit propagation: repeatedly find unit clauses and assign their literals.
        Returns (assign, active_clauses, success).
        """
        changed = True
        while changed:
            changed = False
            new_active = []
            for ci in active_clauses:
                clause = clauses_list[ci]
                # Count unassigned literals and check if satisfied
                unassigned_lits = []
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    if assign[v] == -1:
                        unassigned_lits.append(lit)
                    elif assign[v] == (1 if lit > 0 else 0):
                        satisfied = True
                        break
                
                if satisfied:
                    continue  # Remove satisfied clause
                
                if len(unassigned_lits) == 0:
                    return assign, active_clauses, False  # Conflict: empty clause
                
                if len(unassigned_lits) == 1:
                    # Unit clause: assign the literal
                    lit = unassigned_lits[0]
                    v = abs(lit) - 1
                    assign[v] = 1 if lit > 0 else 0
                    changed = True
                    new_active.append(ci)  # Keep for further propagation check
                else:
                    new_active.append(ci)
            
            active_clauses = new_active
        
        return assign, active_clauses, True
    
    def _pure_literal_eliminate(assign, active_clauses, clauses_list, n):
        """
        Pure literal elimination: if a variable only appears with one polarity in active clauses,
        assign it to satisfy all those clauses.
        """
        # Count polarities for each variable
        pos_count = [0] * n
        neg_count = [0] * n
        for ci in active_clauses:
            clause = clauses_list[ci]
            for lit in clause:
                v = abs(lit) - 1
                if lit > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        new_assign = assign[:]
        new_active = []
        for ci in active_clauses:
            clause = clauses_list[ci]
            satisfied = False
            for lit in clause:
                v = abs(lit) - 1
                if new_assign[v] == -1:
                    # Check if this is a pure literal
                    if lit > 0 and neg_count[v] == 0 and pos_count[v] > 0:
                        new_assign[v] = 1
                    elif lit < 0 and pos_count[v] == 0 and neg_count[v] > 0:
                        new_assign[v] = 0
                if new_assign[v] != -1 and new_assign[v] == (1 if lit > 0 else 0):
                    satisfied = True
                    break
            
            if not satisfied:
                new_active.append(ci)
        
        return new_assign, new_active
    
    def _is_satisfied(clause, assign):
        for lit in clause:
            v = abs(lit) - 1
            if assign[v] == (1 if lit > 0 else 0):
                return True
        return False
    
    # Run DPLL
    initial_active = list(range(len(processed_clauses)))
    result = _dpll_recursive(assignment, initial_active, 0)
    
    if result is not None:
        # Convert to boolean list
        return [bool(a) for a in result]
    
    return None


def _walksat_solve(n, clauses, deadline):
    """
    WalkSAT with improved heuristics:
    - Weight-based flipping
    - Multiple restarts with different seeds
    """
    rng = random.Random(42)
    
    # Try multiple restarts
    best_assignment = [rng.random() < 0.5 for _ in range(n)]
    best_unsat_count = float('inf')
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Weight for each variable
        weights = [1.0] * n
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if assignment[v] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(c)
            
            if not unsat:
                return assignment
            
            # Track best
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment[:]
            
            # Pick an unsatisfied clause
            clause = rng.choice(unsat)
            
            # Choose literal to flip using weight-based heuristic
            # Score = (number of satisfied literals in clause if we flip) * w - (number of unsatisfied literals broken) * (1-w)
            # Actually, standard WalkSAT: with probability p, flip a literal that satisfies the clause
            # Otherwise, flip a random literal in the clause
            
            # Let's use a weighted approach
            best_lit = None
            best_score = -float('inf')
            for lit in clause:
                v = abs(lit) - 1
                current_val = assignment[v]
                desired_val = lit > 0
                
                # If we flip this literal, does it satisfy the clause?
                if current_val != desired_val:
                    # Flipping will satisfy this clause
                    # But may break other clauses where this variable appears
                    # Count how many clauses would be broken
                    break_count = 0
                    for c2 in clauses:
                        if c2 == clause:
                            continue
                        # Check if this clause is currently satisfied and would be broken
                        sat_before = False
                        for lit2 in c2:
                            v2 = abs(lit2) - 1
                            if assignment[v2] == (lit2 > 0):
                                sat_before = True
                                break
                        if not sat_before:
                            continue
                        # Check if flipping this variable breaks it
                        # Only if this clause's only satisfying literal is this variable
                        only_sat = True
                        for lit2 in c2:
                            v2 = abs(lit2) - 1
                            if v2 == v:
                                continue
                            if assignment[v2] == (lit2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            break_count += 1
                    
                    score = 1.0 - 0.5 * break_count * weights[v]
                else:
                    # Flipping will not satisfy this clause (it's already satisfied by this lit)
                    # So flipping breaks the clause
                    score = -1.0 * weights[v]
                
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            if best_lit is None:
                best_lit = rng.choice(clause)
            
            # With some probability, flip randomly instead
            if rng.random() < 0.3:
                best_lit = rng.choice(clause)
            
            v = abs(best_lit) - 1
            assignment[v] = not assignment[v]
    
    # Return best assignment found
    # Check if best is actually satisfying
    for c in clauses:
        sat = False
        for lit in c:
            v = abs(lit) - 1
            if best_assignment[v] == (lit > 0):
                sat = True
                break
        if not sat:
            return best_assignment  # Not satisfying, but return best effort
    
    return best_assignment
# EVOLVE-BLOCK-END
