import time
import random
from collections import deque

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # Safety margin before 330s wall-clock kill
    
    # CDCL SAT Solver
    # Variables are 0-indexed internally
    # We use a watched literals scheme with 2 watches per clause
    
    # Build occurrence lists for variable indexing
    # var_occurs[v] = list of clause indices where variable v appears
    var_occurs = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            var_occurs[v].append(ci)
    
    # CDCL state
    # assignment: -1 unassigned, 0 false, 1 true
    assignment = [-1] * n
    # trail: order of decisions
    trail = []
    # trail_lim: decision levels
    trail_lim = []
    # reasons: for implied assignments, the clause index that forced them
    reasons = [-1] * n
    # watches: for each variable, list of (clause_idx, watch_lit_index_in_clause)
    # We store watches as: watches[v] = list of (clause_idx, lit_in_clause)
    # where lit_in_clause is the literal that "watches" this variable
    watches = [[] for _ in range(n)]
    
    # Initialize watches: each clause watches its first two literals
    for ci, clause in enumerate(clauses):
        if len(clause) >= 2:
            v0 = abs(clause[0]) - 1
            v1 = abs(clause[1]) - 1
            watches[v0].append(ci)
            watches[v1].append(ci)
        elif len(clause) == 1:
            v0 = abs(clause[0]) - 1
            watches[v0].append(ci)
    
    # Conflict analysis will add learned clauses
    learned_clauses = []
    # All clauses: original + learned. We'll use a single list for simplicity.
    # Actually, let's keep them separate for clarity but handle uniformly.
    
    # For simplicity in this implementation, let's use a different approach:
    # We'll implement a simpler but effective CDCL with explicit clause management.
    
    # Let me restructure: use a flat list of all clauses (original + learned)
    all_clauses = [list(c) for c in clauses]
    num_orig = m
    
    # Rebuild watches from all_clauses
    watches = [[] for _ in range(n)]
    for ci, clause in enumerate(all_clauses):
        if len(clause) >= 2:
            v0 = abs(clause[0]) - 1
            v1 = abs(clause[1]) - 1
            watches[v0].append(ci)
            watches[v1].append(ci)
        elif len(clause) == 1:
            v0 = abs(clause[0]) - 1
            watches[v0].append(ci)
    
    # Decision level counter
    decision_level = 0
    
    # Phase saving
    phase = [False] * n  # heuristic for which value to try first
    
    # Stats
    conflicts = 0
    clauses_processed = 0
    
    def lit_value(lit):
        """Return truth value of literal given current assignment. -1 if unassigned."""
        v = abs(lit) - 1
        if assignment[v] == -1:
            return -1
        return 1 if (lit > 0 == (assignment[v] == 1)) else 0
    
    def assign(v, val, reason=-1):
        """Assign variable v to val. Append to trail."""
        assignment[v] = val
        trail.append(v)
        reasons[v] = reason
        if val == 1:
            phase[v] = True
        else:
            phase[v] = False
        return True
    
    def unassign_all():
        """Unassign all variables on trail."""
        global trail
        while trail:
            v = trail.pop()
            assignment[v] = -1
            reasons[v] = -1
        trail_lim.clear()
    
    def get_watch_clause(ci, widx):
        """Get the watched literal for clause ci at watch index widx (0 or 1)."""
        # This is a simplification - in proper CDCL, watches are maintained differently
        # For this implementation, we'll use a simpler propagation approach
        return all_clauses[ci]
    
    def propagate():
        """Unit propagation. Returns (ok, conflict_clause_idx)."""
        global all_clauses, watches, assignment, trail, reasons
        queue = deque()
        
        # Initialize queue with all unassigned variables that have watched clauses
        # that might become unit
        # Actually, we need a proper queue-based propagation
        
        # Let's use a different, cleaner approach for propagation
        # We'll check all clauses for unit status
        
        while True:
            found_unit = False
            for ci, clause in enumerate(all_clauses):
                if len(clause) == 0:
                    return False, ci
                # Count assigned literals
                num_unassigned = 0
                num_false = 0
                last_unassigned = None
                for lit in clause:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        num_unassigned += 1
                        last_unassigned = lit
                    elif lit_value(lit) == 0:
                        num_false += 1
                
                if num_unassigned == 0:
                    if num_false == len(clause):
                        return False, ci
                    # Clause is satisfied
                elif num_unassigned == 1:
                    # Unit clause - assign the remaining literal
                    v = abs(last_unassigned) - 1
                    val = 1 if last_unassigned > 0 else 0
                    if assignment[v] == -1:
                        assign(v, val, ci)
                        found_unit = True
                        # After assigning, we need to re-scan
                        break
                    elif assignment[v] != val:
                        return False, ci
            
            if not found_unit:
                return True, -1
        
    def pick_branch_variable():
        """Pick a variable to branch on using VSIDS-like heuristic."""
        # Simple: pick the first unassigned variable, or one that appears most
        # For better performance, use a heuristic based on occurrence in unsatisfied clauses
        best_v = -1
        best_score = -1
        for v in range(n):
            if assignment[v] != -1:
                continue
            # Count occurrences in potentially unsatisfied clauses
            score = 0
            for ci in var_occurs[v]:
                clause = all_clauses[ci]
                # Check if clause is potentially satisfiable
                num_unassigned = 0
                for lit in clause:
                    if abs(lit) - 1 == v:
                        continue
                    lv = lit_value(lit)
                    if lv == -1:
                        num_unassigned += 1
                    elif lv == 1:
                        num_unassigned = -1
                        break
                if num_unassigned >= 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        if best_v == -1:
            # All assigned
            return -1
        return best_v
    
    def conflict_analysis(conflict_ci):
        """Analyze conflict and return a learned clause and backtrack level."""
        global all_clauses, assignment, trail, reasons, decision_level
        # Simplified 1-UIP conflict analysis
        # This is a simplified version - for proper CDCL we need more sophistication
        
        # For planted SAT instances, a simpler approach may work:
        # Just backtrack and try the other value
        # Find the highest decision level involved in the conflict
        # and backtrack there
        
        # Get the variables involved in the conflict clause
        conflict_vars = set()
        for lit in all_clauses[conflict_ci]:
            conflict_vars.add(abs(lit) - 1)
        
        # Find the second-highest decision level
        # We need to know the decision level of each variable
        # Let's track decision levels
        # Actually, let's simplify: just backtrack to the most recent decision
        # and flip it
        
        # Find the last decision variable on the trail
        # Decisions are marked by trail_lim
        if not trail_lim:
            return None, -1
        
        # The last decision level
        last_dec_level = trail_lim[-1]
        # Find the variable that was decided at that level
        # It's the variable at index trail_lim[-1] in the trail
        # Actually, trail_lim stores indices into the trail where decisions were made
        # So the decision variable is trail[trail_lim[-1]]
        
        if len(trail) > 0 and trail_lim and trail_lim[-1] < len(trail):
            dec_var = trail[trail_lim[-1]]
            return dec_var, decision_level - 1
        return None, -1
    
    def learn_clause(lit_list, backtrack_level):
        """Add a learned clause and return the clause index."""
        global all_clauses
        # Reduce the clause if possible (simplify)
        # Remove satisfied literals
        simplified = []
        for lit in lit_list:
            lv = lit_value(lit)
            if lv != 1:  # Not already true
                simplified.append(lit)
        
        if len(simplified) == 0:
            # Empty clause - UNSAT
            return -1
        if len(simplified) == 1:
            # Unit clause
            pass
        
        all_clauses.append(simplified)
        ci = len(all_clauses) - 1
        
        # Add watches for the new clause
        if len(simplified) >= 2:
            v0 = abs(simplified[0]) - 1
            v1 = abs(simplified[1]) - 1
            watches[v0].append(ci)
            watches[v1].append(ci)
        elif len(simplified) == 1:
            v0 = abs(simplified[0]) - 1
            watches[v0].append(ci)
        
        return ci
    
    def backtrack_to_level(level):
        """Backtrack to a specific decision level."""
        global trail, trail_lim, decision_level, assignment
        # Unassign variables until we reach the target level
        while trail_lim and trail_lim[-1] > level:
            # Unassign all variables after the last decision at level+1
            target = trail_lim[-1]
            while len(trail) > target:
                v = trail.pop()
                assignment[v] = -1
                reasons[v] = -1
            trail_lim.pop()
        decision_level = level
    
    def is_satisfied():
        """Check if all clauses are satisfied."""
        for clause in all_clauses:
            if not any(lit_value(lit) == 1 for lit in clause):
                return False
        return True
    
    def solve_cdcl():
        """Main CDCL solving loop."""
        global decision_level, conflicts, trail, trail_lim, assignment, all_clauses, watches
        start_time = time.time()
        
        while time.time() < deadline:
            # Propagate
            ok, conflict_ci = propagate()
            if not ok:
                # Conflict!
                conflicts += 1
                
                if all_clauses[conflict_ci] == []:
                    return False, None  # UNSAT
                
                # Analyze conflict
                dec_var, bt_level = conflict_analysis(conflict_ci)
                
                if dec_var is None:
                    # Can't analyze - just random restart
                    unassign_all()
                    decision_level = 0
                    continue
                
                # Backtrack
                backtrack_to_level(bt_level)
                
                # Flip the decision variable
                if assignment[dec_var] == 1:
                    assign(dec_var, 0, -1)
                    trail_lim.append(len(trail) - 1)
                    decision_level += 1
                else:
                    assign(dec_var, 1, -1)
                    trail_lim.append(len(trail) - 1)
                    decision_level += 1
                
                # Learn a simple clause: the negation of the conflict clause's unassigned literals
                # Simplified learning
                conflict_clause = all_clauses[conflict_ci]
                learned_lits = []
                for lit in conflict_clause:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        learned_lits.append(-lit)  # Negate
                    elif lit_value(lit) == 0:
                        learned_lits.append(lit)
                
                if learned_lits:
                    ci = learn_clause(learned_lits, bt_level)
                
                continue
            
            # Check if solved
            if is_satisfied():
                result = []
                for v in range(n):
                    if assignment[v] == 1:
                        result.append(True)
                    elif assignment[v] == 0:
                        result.append(False)
                    else:
                        result.append(phase[v])
                return True, result
            
            # Pick a branch variable
            v = pick_branch_variable()
            if v == -1:
                # All variables assigned
                if is_satisfied():
                    result = []
                    for var in range(n):
                        result.append(assignment[var] == 1)
                    return True, result
                else:
                    return False, None
            
            # Branch: try phase-preferred value first
            val = 1 if phase[v] else 0
            assign(v, val, -1)
            trail_lim.append(len(trail) - 1)
            decision_level += 1
        
        # Time's up - return best effort
        # Check if current assignment satisfies
        if is_satisfied():
            result = []
            for v in range(n):
                if assignment[v] == 1:
                    result.append(True)
                elif assignment[v] == 0:
                    result.append(False)
                else:
                    result.append(phase[v])
            return True, result
        return False, None
    
    # Run CDCL
    success, result = solve_cdcl()
    if success and result is not None:
        return result
    
    # Fallback: WalkSAT if CDCL didn't find a solution
    rng = random.Random(42)
    
    def walksat():
        """WalkSAT as fallback."""
        assignment_ws = [rng.random() < 0.5 for _ in range(n)]
        
        # Count satisfied literals per clause
        def count_true(ci):
            count = 0
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if assignment_ws[v] == (lit > 0):
                    count += 1
            return count
        
        clause_true = [count_true(i) for i in range(m)]
        
        def is_solved_ws():
            return all(c >= 1 for c in clause_true)
        
        restart_threshold = 5000
        steps = 0
        p_walk = 0.25
        
        ws_deadline = deadline  # Use remaining time
        
        while time.time() < ws_deadline:
            if is_solved_ws():
                return assignment_ws
            
            if steps >= restart_threshold:
                assignment_ws = [rng.random() < 0.5 for _ in range(n)]
                clause_true = [count_true(i) for i in range(m)]
                steps = 0
                continue
            
            # Find unsatisfied clauses
            unsat = [i for i in range(m) if clause_true[i] == 0]
            if not unsat:
                return assignment_ws
            
            ci = rng.choice(unsat)
            
            # Pick variable to flip
            best_var = -1
            best_gain = -1000
            clause_vars = [abs(lit) - 1 for lit in clauses[ci]]
            
            for v in clause_vars:
                gain = 0
                for other_ci in var_occurs[v]:
                    # Current state
                    cur_true = 0
                    for lit in clauses[other_ci]:
                        vv = abs(lit) - 1
                        if assignment_ws[vv] == (lit > 0):
                            cur_true += 1
                    # New state if we flip v
                    new_true = 0
                    for lit in clauses[other_ci]:
                        vv = abs(lit) - 1
                        if vv == v:
                            new_val = not assignment_ws[v]
                            if new_val == (lit > 0):
                                new_true += 1
                        else:
                            if assignment_ws[vv] == (lit > 0):
                                new_true += 1
                    if new_true > cur_true:
                        gain += 1
                    elif new_true < cur_true:
                        gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            
            if rng.random() < p_walk:
                v = rng.choice(clause_vars)
            else:
                v = best_var
            
            assignment_ws[v] = not assignment_ws[v]
            
            # Update clause_true for affected clauses
            for affected_ci in var_occurs[v]:
                clause_true[affected_ci] = count_true(affected_ci)
            
            steps += 1
        
        return assignment_ws
    
    # Run WalkSAT with remaining time
    if time.time() < deadline:
        result = walksat()
        if result is not None:
            # Verify
            for clause in clauses:
                if any(result[abs(lit)-1] == (lit > 0) for lit in clause):
                    continue
                break
            else:
                return result
    
    # Last resort: return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
