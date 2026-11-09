"""CDCL SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Convert to internal representation: (var, positive) where var is 0-based
    def to_internal(clause):
        return [(abs(lit) - 1, lit > 0) for lit in clause]
    
    iclauses = [to_internal(c) for c in clauses]
    
    # Data structures
    assignment = [None] * n
    reason = [None] * n  # clause index that caused assignment, or None for decision
    trail = []
    trail_lim = []
    decision_level = 0
    learnt_clauses = []
    
    # VSIDS
    activity = [0.0] * n
    var_inc = 1.0
    var_decay = 0.95
    phase = [True] * n
    
    # All clauses (original + learned)
    all_clauses = [list(c) for c in iclauses]
    
    # Watch structure
    watch_lits = []
    watch_map = {}
    
    for ci, clause in enumerate(all_clauses):
        if len(clause) > 2:
            wl = clause[:2]
        else:
            wl = clause[:]
        watch_lits.append(wl)
        for slot, lit in enumerate(wl):
            if lit not in watch_map:
                watch_map[lit] = []
            watch_map[lit].append((ci, slot))
    
    def is_true(lit):
        var, pos = lit
        if assignment[var] is None:
            return False
        return assignment[var] == pos
    
    def is_false(lit):
        var, pos = lit
        if assignment[var] is None:
            return False
        return assignment[var] != pos
    
    def enqueue(lit, ci=None):
        var, pos = lit
        if assignment[var] is not None:
            return assignment[var] == pos
        assignment[var] = pos
        phase[var] = pos
        reason[var] = ci
        trail.append(lit)
        return True
    
    def propagate():
        """Returns None if no conflict, or conflicting clause index."""
        start = len(trail) - 1
        while start >= 0:
            lit = trail[start]
            start -= 1
            var, pos = lit
            false_lit = (var, not pos)
            
            if false_lit not in watch_map:
                continue
            
            watchers = watch_map[false_lit][:]
            for wci, slot in watchers:
                wl = watch_lits[wci]
                other_slot = 1 - slot
                other_lit = wl[other_slot]
                
                if is_true(other_lit):
                    continue
                
                clause = all_clauses[wci]
                found = False
                for i, lit2 in enumerate(clause):
                    if i == slot or i == other_slot:
                        continue
                    if not is_false(lit2):
                        # Found new watch
                        watch_map[false_lit].remove((wci, slot))
                        watch_lits[wci][slot] = lit2
                        if lit2 not in watch_map:
                            watch_map[lit2] = []
                        watch_map[lit2].append((wci, slot))
                        found = True
                        break
                
                if not found:
                    unassigned = [l for l in clause if assignment[l[0]] is None]
                    if len(unassigned) == 0:
                        return wci
                    elif len(unassigned) == 1:
                        ulit = unassigned[0]
                        if not enqueue(ulit, wci):
                            return wci
                        watch_map[false_lit].remove((wci, slot))
                        watch_lits[wci][slot] = ulit
                        if ulit not in watch_map:
                            watch_map[ulit] = []
                        watch_map[ulit].append((wci, slot))
        
        return None
    
    def choose_var():
        best_var = -1
        best_score = -1
        for v in range(n):
            if assignment[v] is None:
                if activity[v] > best_score:
                    best_score = activity[v]
                    best_var = v
        return best_var if best_var >= 0 else None
    
    def decide():
        global decision_level
        var = choose_var()
        if var is None:
            return False
        pos = phase[var]
        trail_lim.append(len(trail))
        decision_level += 1
        enqueue((var, pos), None)
        return True
    
    def backtrack(level):
        global decision_level
        while len(trail_lim) > level:
            lim = trail_lim.pop()
            while len(trail) > lim:
                lit = trail.pop()
                var, pos = lit
                assignment[var] = None
                reason[var] = None
        decision_level = level
    
    def bump_activity(var):
        global var_inc
        activity[var] += var_inc
        if activity[var] > 1e100:
            for v in range(n):
                activity[v] *= 1e-100
            var_inc *= 1e-100
    
    def decay_activities():
        global var_inc
        for v in range(n):
            activity[v] *= var_decay
        var_inc /= var_decay
    
    def analyze(ci):
        """Simplified conflict analysis. Returns learnt clause or None."""
        # For simplicity, just boost activities of variables in the conflict
        for lit in all_clauses[ci]:
            var, pos = lit
            bump_activity(var)
        decay_activities()
        return None
    
    # Main loop
    max_iterations = 100000
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # Propagate
        conflict = propagate()
        
        if conflict is not None:
            # Conflict occurred
            analyze(conflict)
            
            if decision_level == 0:
                break  # Inconsistent
            
            # Backtrack one level
            backtrack(decision_level - 1)
        else:
            # Check if solved
            if all(a is not None for a in assignment):
                if all(any(is_true(lit) for lit in clause) for clause in all_clauses):
                    break
                else:
                    # Should not happen, but break to avoid infinite loop
                    break
            
            # Make a decision
            if not decide():
                # No more variables to decide
                if all(any(is_true(lit) for lit in clause) for clause in all_clauses):
                    break
                else:
                    break
    
    # Build result
    result = []
    for v in range(n):
        if assignment[v] is not None:
            result.append(assignment[v])
        else:
            result.append(phase[v])
    
    return result

# EVOLVE-BLOCK-END
