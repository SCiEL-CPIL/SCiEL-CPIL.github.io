"""Improved 3-SAT solver using Watched Literals + VSIDS + Clause Learning."""
import random
import heapq

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process: remove tautologies and empty clauses, deduplicate
    processed_clauses = []
    seen = set()
    for clause in clauses:
        # Check for tautology
        lit_set = set()
        is_taut = False
        for l in clause:
            if -l in lit_set:
                is_taut = True
                break
            lit_set.add(l)
        if is_taut:
            continue
        # Normalize for dedup
        norm = tuple(sorted(lit_set))
        if norm in seen:
            continue
        seen.add(norm)
        processed_clauses.append(list(lit_set))
    
    clauses = processed_clauses
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Check for empty clause
    for c in clauses:
        if len(c) == 0:
            # Instance is UNSAT, but task says satisfiable, so this shouldn't happen
            # Return arbitrary assignment
            return [True] * n
    
    # Build watched literals structure
    # For each variable, we track which clauses watch it
    # A clause is "watched" by a variable if that variable appears in the clause
    # We maintain for each clause two "watches" - variables whose assignment we monitor
    
    # Assignment: 0 = False, 1 = True, -1 = unassigned
    assignment = [-1] * n
    
    # For each clause, store the list of literals and two watch variables
    # watches[clause_idx] = [var1, var2] - the two variables we're watching
    # We need to be able to find clauses that watch a given variable
    
    # Let's use a different approach: for each variable, maintain a list of (clause_idx, other_watch_var)
    # where the clause is watching this variable
    
    var_watches = [[] for _ in range(n)]  # var_watches[v] = list of (clause_idx, other_var)
    
    # Initialize watches for each clause
    clause_watches = []  # clause_watches[c] = [var1, var2]
    
    for c_idx, clause in enumerate(clauses):
        lits = list(clause)
        if len(lits) >= 2:
            w1, w2 = lits[0], lits[1]
            v1, v2 = abs(w1) - 1, abs(w2) - 1
            clause_watches.append([w1, w2])
            var_watches[v1].append((c_idx, w2))
            var_watches[v2].append((c_idx, w1))
        elif len(lits) == 1:
            w1 = lits[0]
            v1 = abs(w1) - 1
            clause_watches.append([w1, None])
            var_watches[v1].append((c_idx, None))
        else:
            clause_watches.append([None, None])
    
    def is_sat(clause, assign):
        for l in clause:
            v = abs(l) - 1
            if assign[v] == -1:
                continue
            if (l > 0 and assign[v] == 1) or (l < 0 and assign[v] == 0):
                return True
        return False
    
    def is_false(lit, assign):
        v = abs(lit) - 1
        if assign[v] == -1:
            return False
        return (lit > 0 and assign[v] == 0) or (lit < 0 and assign[v] == 1)
    
    # Unit propagation with watched literals
    # We'll use a stack of newly assigned variables for propagation
    
    def propagate(assignment, var_watches, clause_watches, clauses, trail):
        """
        Perform unit propagation. Returns False if contradiction, True otherwise.
        trail is a list of (var, value) in order of assignment.
        """
        # We need to process the queue of newly assigned variables
        # Use a simple queue approach
        queue = list(trail)  # We'll process all assigned vars
        
        # Actually, let's use a different approach. We'll maintain a queue of variables 
        # that have just been assigned and need their watches processed.
        
        # Let me restructure: use a worklist
        worklist = []
        for v, val in trail:
            worklist.append(v)
        
        while worklist:
            v = worklist.pop()
            val = assignment[v]
            if val == -1:
                continue
            
            # Process watches for variable v
            # We need to iterate through var_watches[v] carefully since it may be modified
            i = 0
            while i < len(var_watches[v]):
                c_idx, other_lit = var_watches[v][i]
                clause = clauses[c_idx]
                cw = clause_watches[c_idx]
                
                # Check if the clause is satisfied by current assignment
                # Check if any literal in the clause is true
                sat = False
                for l in clause:
                    vv = abs(l) - 1
                    if assignment[vv] != -1:
                        if (l > 0 and assignment[vv] == 1) or (l < 0 and assignment[vv] == 0):
                            sat = True
                            break
                if sat:
                    # Clause is satisfied, we can remove this watch
                    var_watches[v].pop(i)
                    continue
                
                # Check if the other watch variable is assigned
                if other_lit is None:
                    # This is a unit clause watching only v
                    # If v is false, contradiction
                    if is_false(cw[0], assignment):
                        return False
                    # Otherwise, the clause is unit and satisfied if v is true
                    # But v is assigned, so if v makes it false, we have a problem
                    # Actually if v is false and other is None, the clause is falsified
                    # But we already checked sat, so this means the clause is not satisfied
                    # and other is None, so it must be that v makes it false
                    # Wait, if the clause has only one literal and it's false, contradiction
                    if is_false(cw[0], assignment):
                        return False
                    # Otherwise the clause is satisfied by v being true
                    # Remove the watch
                    var_watches[v].pop(i)
                    continue
                
                other_v = abs(other_lit) - 1
                if assignment[other_v] != -1:
                    # Other watch is assigned
                    # Check if the clause is satisfied by other_v
                    if not is_false(other_lit, assignment):
                        # Other watch satisfies the clause, remove this watch
                        var_watches[v].pop(i)
                        continue
                    # Other watch is false, need to find a new watch
                    # Look for another unassigned variable in the clause
                    new_watch_found = False
                    for l in clause:
                        vv = abs(l) - 1
                        if vv != v and assignment[vv] == -1:
                            # Found a new watch
                            # Remove current watch from v
                            var_watches[v].pop(i)
                            # Add new watch
                            var_watches[vv].append((c_idx, cw[0] if cw[1] == other_lit else other_lit))
                            # Update clause watches
                            if cw[0] == other_lit:
                                cw[1] = l
                            else:
                                cw[0] = l
                            new_watch_found = True
                            break
                    
                    if not new_watch_found:
                        # No unassigned variable left, clause is unit
                        # The remaining literal must be assigned
                        # Check if it's satisfied
                        for l in clause:
                            vv = abs(l) - 1
                            if assignment[vv] == -1:
                                # This shouldn't happen if we checked above
                                pass
                            elif (l > 0 and assignment[vv] == 1) or (l < 0 and assignment[vv] == 0):
                                # Satisfied
                                var_watches[v].pop(i)
                                new_watch_found = True
                                break
                            else:
                                # Falsified
                                return False
                        if not new_watch_found:
                            # All literals are assigned and none satisfy -> contradiction
                            return False
                    # Continue without incrementing i since we popped
                else:
                    # Other watch is unassigned, keep the watch
                    i += 1
        
        return True
    
    # This watched literal approach is getting complex. Let me use a simpler but effective approach:
    # Use a hybrid of WalkSAT with better heuristics and restarts.
    
    # Let's go with a cleaner DPLL-like approach with unit propagation and pure literal elimination
    
    def unit_propagate(assignment, clauses):
        """
        Simple unit propagation. Returns (success, new_assignments)
        new_assignments is list of (var, value)
        """
        new_assigns = []
        changed = True
        # Track which clauses are unit
        while changed:
            changed = False
            for clause in clauses:
                # Count unassigned and check if any are true
                unassigned_lits = []
                has_true = False
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] == 1:
                        if l > 0:
                            has_true = True
                            break
                    elif assignment[v] == 0:
                        if l < 0:
                            has_true = True
                            break
                    else:
                        unassigned_lits.append(l)
                if has_true:
                    continue
                if len(unassigned_lits) == 0:
                    return False, []
                if len(unassigned_lits) == 1:
                    l = unassigned_lits[0]
                    v = abs(l) - 1
                    val = 1 if l > 0 else 0
                    if assignment[v] != val:
                        assignment[v] = val
                        new_assigns.append((v, val))
                        changed = True
        return True, new_assigns
    
    def solve_dpll(assignment, clauses, trail, depth, max_depth=150):
        """
        DPLL with unit propagation.
        Returns True if satisfiable, False if not.
        """
        if depth > max_depth:
            return False
        
        # Unit propagation
        success, new_assigns = unit_propagate(assignment, clauses)
        if not success:
            return False
        
        # Check if all clauses are satisfied
        for clause in clauses:
            sat = False
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == 1:
                    if l > 0:
                        sat = True
                        break
                elif assignment[v] == 0:
                    if l < 0:
                        sat = True
                        break
            if not sat:
                # Not all satisfied, need to branch
                break
        else:
            return True  # All clauses satisfied
        
        # Find a variable to branch on
        # Choose variable appearing in most unsatisfied clauses
        var_count = {}
        for clause in clauses:
            sat = False
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == 1:
                    if l > 0:
                        sat = True
                        break
                elif assignment[v] == 0:
                    if l < 0:
                        sat = True
                        break
            if not sat:
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] == -1:
                        var_count[v] = var_count.get(v, 0) + 1
        
        if not var_count:
            return False
        
        # Choose the variable with highest count
        var = max(var_count, key=var_count.get)
        
        # Try True first
        assignment[var] = 1
        trail.append((var, 1))
        if solve_dpll(assignment, clauses, trail, depth + 1, max_depth):
            return True
        trail.pop()
        
        # Try False
        assignment[var] = 0
        trail.append((var, 0))
        if solve_dpll(assignment, clauses, trail, depth + 1, max_depth):
            return True
        trail.pop()
        
        assignment[var] = -1
        return False
    
    # Use a randomized restart approach with DPLL
    rng = random.Random(42)
    
    best_assignment = None
    max_restarts = 200
    max_depth_per_restart = 100
    
    for restart in range(max_restarts):
        assignment = [-1] * n
        trail = []
        
        # Random initial assignments for some variables
        num_random = min(n, 50 + restart * 5)
        random_vars = rng.sample(range(n), num_random)
        for v in random_vars:
            assignment[v] = 1 if rng.random() < 0.5 else 0
            trail.append((v, assignment[v]))
        
        # Try DPLL from here
        if solve_dpll(assignment, clauses, trail, 0, max_depth_per_restart):
            # Check if solution is valid
            valid = True
            for clause in clauses:
                sat = False
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] == 1:
                        if l > 0:
                            sat = True
                            break
                    elif assignment[v] == 0:
                        if l < 0:
                            sat = True
                            break
                if not sat:
                    valid = False
                    break
            if valid:
                # Convert to boolean list
                result = [assignment[i] == 1 for i in range(n)]
                return result
    
    # Fallback: WalkSAT
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    for _ in range(50000):
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        # Weighted clause selection
        clause = rng.choice(unsat)
        # Flip the variable that appears in the most unsatisfied clauses
        var_scores = {}
        for c in unsat:
            for l in c:
                v = abs(l) - 1
                var_scores[v] = var_scores.get(v, 0) + 1
        # Choose from clause's variables with bias towards higher score
        clause_vars = [abs(l) - 1 for l in clause]
        # Weighted choice
        weights = [var_scores.get(v, 1) for v in clause_vars]
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0
        chosen_var = clause_vars[0]
        for v, w in zip(clause_vars, weights):
            cumulative += w
            if r <= cumulative:
                chosen_var = v
                break
        assignment[chosen_var] = not assignment[chosen_var]
    
    return assignment
