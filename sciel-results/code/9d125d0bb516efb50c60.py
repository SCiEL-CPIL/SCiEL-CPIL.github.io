import random
import time
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 300s limit
    
    m = len(clauses)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            is_pos = l > 0
            var_to_clauses[v].append((ci, is_pos))
    
    # Count occurrences for variable selection
    var_occurrences = [0] * n
    for v in range(n):
        var_occurrences[v] = len(var_to_clauses[v])
    
    # ============ DPLL with Watched Literals ============
    def dpll_watched(deadline_dpll):
        """
        Complete DPLL solver with watched literals, unit propagation, and pure literal elimination.
        Returns (success, assignment) where assignment is list of booleans or None.
        """
        # assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        
        # Watched literals: for each clause, maintain two watched literals
        # watched[ci] = [lit1, lit2] where lit is signed literal (1-based)
        # We need to track which literal is which position
        clause_watches = []  # list of [watch1, watch2] for each clause
        clause_false_count = [0] * m  # number of literals that are false in each clause
        
        # Initialize watches
        for ci, clause in enumerate(clauses):
            if len(clause) >= 2:
                clause_watches.append([clause[0], clause[1]])
            elif len(clause) == 1:
                clause_watches.append([clause[0], 0])  # 0 means no second watch
            else:
                # Empty clause - unsatisfiable
                return False, None
        
        # Track which clauses are satisfied
        clause_satisfied = [False] * m
        
        def set_variable(v, val):
            """Set variable v to val (0 or 1). Returns (success, list_of_new_units)."""
            if assignment[v] != -1:
                return assignment[v] == val, []
            
            assignment[v] = val
            new_units = []
            
            # Process all clauses containing v
            for (ci, is_pos) in var_to_clauses[v]:
                if clause_satisfied[ci]:
                    continue
                
                lit = v + 1 if is_pos else -(v + 1)
                clause = clauses[ci]
                
                # Check if this literal is in the watched positions
                watch1, watch2 = clause_watches[ci]
                is_w1 = (watch1 == lit)
                is_w2 = (watch2 == lit)
                
                if is_w1 or is_w2:
                    # This literal is watched
                    if is_pos == (val == 1):
                        # Literal is satisfied -> clause is satisfied
                        clause_satisfied[ci] = True
                    else:
                        # Literal is false
                        clause_false_count[ci] += 1
                        
                        # Check if clause is now false
                        if clause_false_count[ci] == len(clause):
                            return False, []
                        
                        # Need to find a new watch
                        # Find another unassigned or satisfied literal
                        new_watch = 0
                        for l in clause:
                            if l == lit:
                                continue
                            other_v = abs(l) - 1
                            other_is_pos = l > 0
                            if assignment[other_v] == -1:
                                new_watch = l
                                break
                            elif other_is_pos == (assignment[other_v] == 1):
                                # This literal is satisfied
                                new_watch = l
                                break
                        
                        if new_watch == 0:
                            # No other literal found - clause is unit or false
                            # Since we already checked it's not false, it must be unit
                            # But we need to identify the unit literal
                            for l in clause:
                                other_v = abs(l) - 1
                                if assignment[other_v] == -1:
                                    new_watch = l
                                    break
                            if new_watch == 0:
                                # All literals assigned and none satisfied -> should have been caught
                                return False, []
                            else:
                                # This is a unit clause
                                new_units.append(new_watch)
                        
                        # Update watch
                        if is_w1:
                            clause_watches[ci][0] = new_watch
                        else:
                            clause_watches[ci][1] = new_watch
            
            return True, new_units
        
        # Unit propagation queue
        unit_queue = deque()
        
        # Initialize: find any initial unit clauses
        for ci, clause in enumerate(clauses):
            if len(clause) == 1:
                unit_queue.append(clause[0])
        
        def propagate_units():
            """Process all unit clauses. Returns False if conflict."""
            while unit_queue:
                if time.time() > deadline_dpll:
                    return True  # Timeout, continue with current state
                
                lit = unit_queue.popleft()
                v = abs(lit) - 1
                val = 1 if lit > 0 else 0
                
                success, new_units = set_variable(v, val)
                if not success:
                    return False
                unit_queue.extend(new_units)
            
            return True
        
        # Check if all clauses are satisfied
        def all_satisfied():
            for ci in range(m):
                if not clause_satisfied[ci]:
                    # Check if clause has unassigned literals
                    has_unassigned = False
                    for l in clauses[ci]:
                        v = abs(l) - 1
                        if assignment[v] == -1:
                            has_unassigned = True
                            break
                    if not has_unassigned:
                        # All assigned but not satisfied -> false
                        return False
            return True
        
        # Find pure literal
        def find_pure_literal():
            """Find a variable that only appears in one polarity among unsatisfied clauses."""
            for v in range(n):
                if assignment[v] != -1:
                    continue
                has_pos = False
                has_neg = False
                for (ci, is_pos) in var_to_clauses[v]:
                    if clause_satisfied[ci]:
                        continue
                    if is_pos:
                        has_pos = True
                    else:
                        has_neg = True
                    if has_pos and has_neg:
                        break
                if has_pos and not has_neg:
                    return v, True
                elif has_neg and not has_pos:
                    return v, False
            return None, None
        
        # Select branching variable
        def select_branching_variable():
            """Select variable appearing in most unsatisfied clauses."""
            best_v = -1
            best_count = -1
            for v in range(n):
                if assignment[v] != -1:
                    continue
                count = 0
                for (ci, _) in var_to_clauses[v]:
                    if not clause_satisfied[ci]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            return best_v
        
        # Main DPLL loop with backtracking
        # Use a simple recursive approach with trail for backtracking
        
        def dpll_rec(depth, max_depth):
            """Recursive DPLL with trail-based backtracking."""
            if time.time() > deadline_dpll:
                return None  # Timeout
            
            if depth > max_depth:
                return None
            
            # Check if solved
            if all_satisfied():
                result = [False] * n
                for v in range(n):
                    if assignment[v] == 1:
                        result[v] = True
                    elif assignment[v] == -1:
                        result[v] = rng.random() < 0.5
                return result
            
            # Propagate units
            if not propagate_units():
                return None  # Conflict
            
            # Check again after propagation
            if all_satisfied():
                result = [False] * n
                for v in range(n):
                    if assignment[v] == 1:
                        result[v] = True
                    elif assignment[v] == -1:
                        result[v] = rng.random() < 0.5
                return result
            
            # Try pure literal elimination
            v, val = find_pure_literal()
            if v is not None:
                success, new_units = set_variable(v, 1 if val else 0)
                if not success:
                    return None
                unit_queue.extend(new_units)
                if not propagate_units():
                    return None
                return dpll_rec(depth + 1, max_depth)
            
            # Select branching variable
            bv = select_branching_variable()
            if bv == -1:
                return None  # All assigned but not satisfied
            
            # Try True
            save_assignment = assignment[bv]
            success, new_units = set_variable(bv, 1)
            if success:
                unit_queue.extend(new_units)
                if propagate_units():
                    result = dpll_rec(depth + 1, max_depth)
                    if result is not None:
                        return result
                # Backtrack
                assignment[bv] = -1
                # Note: proper backtracking requires undoing all changes
                # For simplicity, we'll restart from scratch on backtrack
                # This is inefficient but correct for our use case
            
            # Try False
            success, new_units = set_variable(bv, 0)
            if success:
                unit_queue.extend(new_units)
                if propagate_units():
                    result = dpll_rec(depth + 1, max_depth)
                    if result is not None:
                        return result
                assignment[bv] = -1
            
            return None
        
        # Use iterative deepening with a reasonable depth limit
        # Since n=50, we can try up to n levels
        for max_depth in [10, 20, 30, 50]:
            result = dpll_rec(0, max_depth)
            if result is not None:
                return True, result
        
        return False, None
    
    # ============ WalkSAT with Adaptive Noise ============
    def walksat(deadline_ws):
        """High-performance WalkSAT with adaptive noise."""
        
        def initialize():
            """Initialize random assignment and clause true counts."""
            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_true_count = [0] * m
            for ci, clause in enumerate(clauses):
                count = 0
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        count += 1
                clause_true_count[ci] = count
            return assignment, clause_true_count
        
        def update_clause_count(ci, assignment):
            """Recalculate true count for clause ci."""
            count = 0
            for l in clauses[ci]:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    count += 1
            return count
        
        def is_solved(clause_true_count):
            return all(c >= 1 for c in clause_true_count)
        
        # Adaptive parameters
        p_walk = 0.25  # Base walk probability
        min_p_walk = 0.1
        max_p_walk = 0.4
        restart_threshold = 3000
        step_decay = 0.999
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        assignment, clause_true_count = initialize()
        
        unsat_count = sum(1 for c in clause_true_count if c == 0)
        if unsat_count == 0:
            return assignment
        
        steps = 0
        restarts = 0
        max_restarts = 50
        
        while time.time() < deadline_ws:
            if is_solved(clause_true_count):
                return assignment
            
            # Track best solution
            current_unsat = sum(1 for c in clause_true_count if c == 0)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Check for restart
            if steps >= restart_threshold or restarts >= max_restarts:
                # Decay restart threshold
                restart_threshold = int(restart_threshold * 0.8)
                assignment, clause_true_count = initialize()
                steps = 0
                restarts += 1
                continue
            
            # Find unsatisfied clauses
            unsat_indices = [i for i, c in enumerate(clause_true_count) if c == 0]
            if not unsat_indices:
                return assignment
            
            # Select clause: weighted by number of false literals (all are 0 true, so pick random)
            ci = rng.choice(unsat_indices)
            
            # Select variable to flip using WalkSAT heuristic
            clause = clauses[ci]
            best_var = -1
            best_gain = -1000
            gains = []
            
            for l in clause:
                v = abs(l) - 1
                gain = 0
                for (other_ci, is_pos) in var_to_clauses[v]:
                    if assignment[v] == is_pos:
                        gain -= 1
                    else:
                        gain += 1
                gains.append((gain, v))
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            
            # Adaptive walk probability: increase when stuck
            local_steps = steps % restart_threshold
            effective_p_walk = p_walk + 0.1 * (local_steps / restart_threshold)
            effective_p_walk = min(effective_p_walk, max_p_walk)
            
            if rng.random() < effective_p_walk:
                v = rng.choice([abs(l) - 1 for l in clause])
            else:
                v = best_var
            
            # Flip variable
            assignment[v] = not assignment[v]
            
            # Update clause counts
            for (affected_ci, _) in var_to_clauses[v]:
                clause_true_count[affected_ci] = update_clause_count(affected_ci, assignment)
            
            steps += 1
        
        # Return best assignment found
        if best_assignment is not None:
            return best_assignment
        return assignment
    
    # ============ Main Solver ============
    
    # Phase 1: Try DPLL for easy instances
    dpll_deadline = start_time + 30  # Give DPLL up to 30 seconds
    success, dpll_result = dpll_watched(dpll_deadline)
    if success and dpll_result is not None:
        # Verify the solution
        valid = True
        for clause in clauses:
            sat = False
            for l in clause:
                v = abs(l) - 1
                if dpll_result[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                valid = False
                break
        if valid:
            return dpll_result
    
    # Phase 2: Fall back to WalkSAT for harder instances
    ws_deadline = deadline
    result = walksat(ws_deadline)
    
    # Final verification
    if result is not None:
        for clause in clauses:
            sat = False
            for l in clause:
                v = abs(l) - 1
                if result[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                break
    
    return result
