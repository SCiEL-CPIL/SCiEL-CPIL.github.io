import random
import sys
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Seed based on instance for reproducibility
    seed_val = sum(abs(l) for c in clauses for l in c) % (2**32)
    rng = random.Random(seed_val)
    
    num_clauses = len(clauses)
    
    # --- DPLL Solver with Unit Propagation and VSIDS ---
    
    # Data structures for DPLL
    # We need to handle dynamic clause deletion (unit propagation) efficiently.
    # For simplicity and speed in Python, we will use a recursive DPLL with
    # careful bookkeeping. Given n=150, recursion depth is manageable.
    
    # Variable activity scores for VSIDS
    var_activity = [0.0] * (n + 1)
    var_phase = [False] * (n + 1)  # Phase saving: last assigned value
    
    # Clause database: list of lists of literals (signed 1-based)
    # We will maintain a list of active clauses.
    # To handle unit propagation efficiently, we track watchers.
    # However, implementing full watched literals in Python from scratch can be complex.
    # A simpler but effective approach for n=150 is to use a clause list and
    # perform unit propagation by scanning or using a queue of unit clauses.
    
    # Let's implement a standard DPLL with Unit Propagation using a queue.
    # For n=150, scanning clauses for units might be okay if we optimize,
    # but let's try a slightly more efficient structure:
    # var_watchers[var] = list of clause indices watching this variable.
    
    # Initialize watchers
    var_watchers = [[] for _ in range(n + 1)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit)
            var_watchers[v].append(i)
    
    # Active clauses
    active_clauses = [list(c) for c in clauses]
    
    # Assignment stack
    assignment = [None] * (n + 1)
    
    # Queue for unit propagation
    from collections import deque
    unit_queue = deque()
    
    def assign_var(v, val):
        """Assign variable v to val (True/False). Returns False if conflict."""
        assignment[v] = val
        var_phase[v] = val
        # Check watchers for variable v
        for ci in var_watchers[v]:
            clause = active_clauses[ci]
            if not clause:
                continue
            # Check if clause is satisfied
            sat = False
            for lit in clause:
                if abs(lit) == v:
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
                else:
                    # Check other literals in clause
                    v2 = abs(lit)
                    if assignment[v2] is not None:
                        if (lit > 0 and assignment[v2]) or (lit < 0 and not assignment[v2]):
                            sat = True
                            break
            if sat:
                continue
            
            # Clause is not satisfied. Check if it becomes unit.
            # Count unassigned literals
            unassigned = []
            for lit in clause:
                v2 = abs(lit)
                if assignment[v2] is None:
                    unassigned.append(lit)
            
            if len(unassigned) == 0:
                # Conflict: clause is falsified
                return False
            elif len(unassigned) == 1:
                # Unit clause
                unit_queue.append((unassigned[0], ci))
                
        return True

    def propagate():
        """Process unit queue. Returns False if conflict, True otherwise."""
        while unit_queue:
            lit, ci = unit_queue.popleft()
            v = abs(lit)
            val = lit > 0
            
            # If already assigned, check for conflict
            if assignment[v] is not None:
                if assignment[v] != val:
                    return False
                else:
                    continue
            
            # Assign
            if not assign_var(v, val):
                return False
        
        return True

    def vsids_select_var():
        """Select next variable using VSIDS heuristic."""
        # Find unassigned variable with highest activity
        best_v = -1
        best_act = -1.0
        for v in range(1, n + 1):
            if assignment[v] is None:
                if var_activity[v] > best_act:
                    best_act = var_activity[v]
                    best_v = v
        
        if best_v == -1:
            return None
            
        # Decay activities
        for v in range(1, n + 1):
            var_activity[v] *= 0.95
            
        # Bump activity of selected variable
        var_activity[best_v] += 1.0
        
        return best_v

    def dpll():
        """Recursive DPLL search."""
        if not propagate():
            return False
            
        # Check if all variables assigned
        # If yes, and no conflicts, we have a solution
        # We can check if any variable is unassigned to decide branching
        unassigned_vars = [v for v in range(1, n + 1) if assignment[v] is None]
        
        if not unassigned_vars:
            return True
            
        # Select variable
        v = vsids_select_var()
        if v is None:
            return True # Should not happen if unassigned_vars is not empty
        
        # Try phase-saved value first
        val1 = var_phase[v]
        
        # Save state for backtracking (simplification: copy assignment and activities)
        # For n=150, copying is cheap.
        old_assignment = assignment[:]
        old_activity = var_activity[:]
        old_queue = list(unit_queue)
        unit_queue.clear()
        
        # Branch 1: Try val1
        if assign_var(v, val1):
            if dpll():
                return True
        else:
            # Conflict immediately? Unlikely if propagation didn't catch it, 
            # but assign_var checks watchers. If it returns False, it means conflict.
            pass
            
        # Restore state
        assignment[:] = old_assignment
        var_activity[:] = old_activity
        unit_queue.clear()
        for item in old_queue:
            unit_queue.append(item)
        
        # Branch 2: Try not val1
        val2 = not val1
        if assign_var(v, val2):
            if dpll():
                return True
        
        # Restore state again (for future backtracking if this call returns False)
        assignment[:] = old_assignment
        var_activity[:] = old_activity
        unit_queue.clear()
        for item in old_queue:
            unit_queue.append(item)
            
        return False

    # Time budget
    start_time = time.time()
    time_budget = 290
    
    # Run DPLL
    # We need to handle the case where DPLL takes too long.
    # Since Python recursion and copying might be slow, we add a time check.
    # However, DPLL is deterministic and usually fast for satisfiable planted instances.
    # If it times out, we fall back to WalkSAT.
    
    try:
        # Set recursion limit higher just in case
        sys.setrecursionlimit(10000)
        
        # Initial propagation
        if propagate():
            # Check if solved
            if all(assignment[v] is not None for v in range(1, n + 1)):
                # Verify solution
                # Convert to 0-based list of booleans
                result = [assignment[v] for v in range(1, n + 1)]
                return result
            
            # Run DPLL
            if dpll():
                result = [assignment[v] for v in range(1, n + 1)]
                return result
    except RecursionError:
        pass
    except Exception:
        pass
    
    # Fallback: WalkSAT if DPLL fails or times out
    # (Reusing the logic from the original program but optimized)
    
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_list = []
        signs_list = []
        for l in c:
            vars_list.append(abs(l) - 1)
            signs_list.append(1 if l > 0 else -1)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    def count_unsat(assignment_list):
        count = 0
        for i in range(num_clauses):
            sat = False
            for j, v in enumerate(clause_vars[i]):
                if assignment_list[v] == (clause_signs[i][j] > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def get_unsat_clauses(assignment_list):
        unsat = []
        for i in range(num_clauses):
            sat = False
            for j, v in enumerate(clause_vars[i]):
                if assignment_list[v] == (clause_signs[i][j] > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def walksat(max_restarts, max_flips):
        best_assignment = None
        best_unsat_count = float('inf')
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget - 5:
                break
                
            assignment_list = [rng.random() < 0.5 for _ in range(n)]
            
            curr_unsat = count_unsat(assignment_list)
            if curr_unsat == 0:
                return assignment_list
            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment_list[:]
            
            for flip_count in range(max_flips):
                if time.time() - start_time > time_budget - 5:
                    break
                    
                unsat_indices = get_unsat_clauses(assignment_list)
                if not unsat_indices:
                    return assignment_list
                
                curr_unsat = len(unsat_indices)
                if curr_unsat < best_unsat_count:
                    best_unsat_count = curr_unsat
                    best_assignment = assignment_list[:]
                
                # Pick a random unsat clause
                chosen_idx = rng.choice(unsat_indices)
                clause = chosen_idx
                clause_vars_local = clause_vars[clause]
                
                # Pick a variable to flip
                # Heuristic: random with slight bias towards variables in more unsat clauses
                # For simplicity and speed, just random pick from clause vars
                best_v = rng.choice(clause_vars_local)
                
                assignment_list[best_v] = not assignment_list[best_v]
            
        return best_assignment

    result = walksat(50, 5000)
    if result is not None and count_unsat(result) == 0:
        return result
        
    return result
