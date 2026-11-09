import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Parameters
    max_steps_per_run = 10000
    base_noise = 0.25
    max_noise = 0.8
    
    # Pre-process: for each variable, store list of (clause_idx, literal_value)
    var_occurrences = [[] for _ in range(n)]
    
    # Initialize
    rng = random.Random(42)
    
    # Build index
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # ==================== DPLL SOLVER ====================
    def dpll_solve():
        """Complete DPLL with watched literals and unit propagation."""
        # Watched literals: for each clause, track two literals
        # Assignment: -1 unassigned, 0 false, 1 true
        assignment = [-1] * n
        
        # Watched literals per clause: list of (clause_idx, lit_pos)
        # lit_pos is 0 or 1 indicating which position in the clause is watched
        # We store: watches[clause_idx] = [pos1, pos2]
        watches = []
        for c in clauses:
            watches.append([0, 1])
        
        # For efficient lookup: for each variable, list of (clause_idx, watch_pos)
        var_watches = [[] for _ in range(n)]
        for ci in range(len(clauses)):
            for wp in [0, 1]:
                lit = clauses[ci][watches[ci][wp]]
                v = abs(lit) - 1
                var_watches[v].append((ci, wp))
        
        def lit_value(lit):
            v = abs(lit) - 1
            if assignment[v] == -1:
                return -1  # unassigned
            if lit > 0:
                return 1 if assignment[v] == 1 else 0
            else:
                return 1 if assignment[v] == 0 else 0
        
        def is_unsat(lit):
            v = abs(lit) - 1
            if assignment[v] == -1:
                return False
            if lit > 0:
                return assignment[v] == 0
            else:
                return assignment[v] == 1
        
        def is_sat(lit):
            v = abs(lit) - 1
            if assignment[v] == -1:
                return False
            if lit > 0:
                return assignment[v] == 1
            else:
                return assignment[v] == 0
        
        def propagate():
            """Unit propagation. Returns False on contradiction."""
            queue = []
            
            # Find all unit clauses to start
            for ci in range(len(clauses)):
                c = clauses[ci]
                sat_count = 0
                unassigned_lits = []
                for l in c:
                    if is_sat(l):
                        sat_count += 1
                    elif is_unsat(l):
                        continue
                    else:
                        unassigned_lits.append(l)
                
                if sat_count == 0:
                    if len(unassigned_lits) == 0:
                        return False  # empty clause
                    elif len(unassigned_lits) == 1:
                        queue.append(unassigned_lits[0])
            
            while queue:
                lit = queue.pop()
                v = abs(lit) - 1
                
                if assignment[v] != -1:
                    if is_sat(lit):
                        continue
                    else:
                        return False  # contradiction
                
                # Assign variable
                if lit > 0:
                    assignment[v] = 1
                else:
                    assignment[v] = 0
                
                # Update watches
                for ci, wp in list(var_watches[v]):
                    c = clauses[ci]
                    w1 = c[watches[ci][0]]
                    w2 = c[watches[ci][1]]
                    
                    # Check if this clause is satisfied
                    if is_sat(w1) or is_sat(w2):
                        continue
                    
                    # Both watched literals are false or unassigned
                    # Need to find new watches
                    if is_unsat(w1) and is_unsat(w2):
                        return False  # contradiction
                    
                    # Find new watch positions
                    new_pos = -1
                    for i, l in enumerate(c):
                        if i != watches[ci][0] and i != watches[ci][1] and not is_unsat(l):
                            new_pos = i
                            break
                    
                    if new_pos == -1:
                        return False  # contradiction
                    
                    # Update watch
                    if wp == 0:
                        watches[ci][0] = new_pos
                    else:
                        watches[ci][1] = new_pos
                    
                    # Update var_watches
                    new_lit = c[new_pos]
                    new_v = abs(new_lit) - 1
                    var_watches[new_v].append((ci, 1 if wp == 0 else 0))
                    
                    # If new watch is unassigned, it might be unit
                    if assignment[new_v] == -1:
                        # Check if clause is now unit
                        unassigned_count = 0
                        sat_found = False
                        for l in c:
                            if is_sat(l):
                                sat_found = True
                                break
                            elif not is_unsat(l):
                                unassigned_count += 1
                        if not sat_found and unassigned_count == 1:
                            # Find the unassigned literal
                            for l in c:
                                if not is_unsat(l) and not is_sat(l):
                                    queue.append(l)
                                    break
            
            return True
        
        def count_unassigned():
            return sum(1 for a in assignment if a == -1)
        
        def dpll_recursion(depth=0):
            if depth > 1000:
                return None  # Too deep, fallback to WalkSAT
            
            if time.time() > deadline - 5:
                return None  # Time's up
            
            # Check if all assigned and satisfied
            if count_unassigned() == 0:
                # Verify all clauses are satisfied
                for ci in range(len(clauses)):
                    c = clauses[ci]
                    if not any(is_sat(l) for l in c):
                        return None
                return assignment[:]
            
            # Try unit propagation
            if not propagate():
                return None
            
            # Check if all clauses satisfied
            all_sat = True
            for ci in range(len(clauses)):
                c = clauses[ci]
                if not any(is_sat(l) for l in c):
                    all_sat = False
                    break
            
            if all_sat:
                return assignment[:]
            
            # Pure literal elimination
            var_polarity = [0] * n  # 0: both, 1: only positive, 2: only negative
            for ci in range(len(clauses)):
                c = clauses[ci]
                if any(is_sat(l) for l in c):
                    continue
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] != -1:
                        continue
                    if l > 0:
                        var_polarity[v] |= 1
                    else:
                        var_polarity[v] |= 2
            
            for v in range(n):
                if assignment[v] == -1 and var_polarity[v] == 1:
                    assignment[v] = 1
                elif assignment[v] == -1 and var_polarity[v] == 2:
                    assignment[v] = 0
            
            # Re-propagate after pure literal elimination
            if not propagate():
                return None
            
            # Check if all clauses satisfied
            all_sat = True
            for ci in range(len(clauses)):
                c = clauses[ci]
                if not any(is_sat(l) for l in c):
                    all_sat = False
                    break
            
            if all_sat:
                return assignment[:]
            
            # Variable selection: pick variable with highest occurrence in unsat clauses
            var_unsat_count = [0] * n
            for ci in range(len(clauses)):
                c = clauses[ci]
                if any(is_sat(l) for l in c):
                    continue
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == -1:
                        var_unsat_count[v] += 1
            
            best_var = -1
            best_count = -1
            for v in range(n):
                if assignment[v] == -1 and var_unsat_count[v] > best_count:
                    best_count = var_unsat_count[v]
                    best_var = v
            
            if best_var == -1:
                # All variables assigned but not all satisfied
                return None
            
            # Try both values
            for val in [1, 0]:
                assignment[best_var] = val
                # Save state
                old_assignment = assignment[:]
                
                if propagate():
                    # Check if all satisfied
                    all_sat = True
                    for ci in range(len(clauses)):
                        c = clauses[ci]
                        if not any(is_sat(l) for l in c):
                            all_sat = False
                            break
                    
                    if all_sat:
                        return assignment[:]
                    
                    result = dpll_recursion(depth + 1)
                    if result is not None:
                        return result
                
                # Backtrack
                assignment[:] = old_assignment
            
            return None
        
        return dpll_recursion()
    
    # ==================== WALKSAT HELPERS ====================
    def compute_clause_sat_count(assignment):
        counts = [0] * len(clauses)
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    counts[i] += 1
        return counts
    
    def is_satisfied(clause_sat_count):
        for count in clause_sat_count:
            if count == 0:
                return False
        return True
    
    def get_unsat_indices(clause_sat_count):
        return [i for i, count in enumerate(clause_sat_count) if count == 0]
    
    def is_literal_satisfied(l, assignment):
        v = abs(l) - 1
        return (l > 0 and assignment[v]) or (l < 0 and not assignment[v])
    
    def apply_flip(v, assignment, clause_sat_count):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            if is_literal_satisfied(lit, assignment):
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    def walksat_step(assignment, clause_sat_count, rng, noise):
        unsat_indices = get_unsat_indices(clause_sat_count)
        if not unsat_indices:
            return True
        
        clause_idx = rng.choice(unsat_indices)
        clause = clauses[clause_idx]
        
        if rng.random() < noise:
            l = rng.choice(clause)
            v = abs(l) - 1
            apply_flip(v, assignment, clause_sat_count)
            return False
        
        best_var_idx = None
        best_newly_unsat = float('inf')
        best_newly_sat = 0
        
        for l in clause:
            v = abs(l) - 1
            
            newly_unsat = 0
            newly_sat = 0
            
            for ci, lit in var_occurrences[v]:
                old_count = clause_sat_count[ci]
                currently_sat = is_literal_satisfied(lit, assignment)
                
                if currently_sat:
                    if old_count == 1:
                        newly_unsat += 1
                else:
                    if old_count == 0:
                        newly_sat += 1
            
            if newly_unsat < best_newly_unsat or (newly_unsat == best_newly_unsat and newly_sat > best_newly_sat):
                best_newly_unsat = newly_unsat
                best_newly_sat = newly_sat
                best_var_idx = v
        
        if best_var_idx is not None:
            apply_flip(best_var_idx, assignment, clause_sat_count)
        
        return False
    
    # ==================== MAIN SOLVER ====================
    # First try DPLL
    dpll_result = dpll_solve()
    if dpll_result is not None:
        # Convert DPLL result (-1, 0, 1) to boolean list
        result = []
        for a in dpll_result:
            if a == 1:
                result.append(True)
            elif a == 0:
                result.append(False)
            else:
                result.append(rng.random() < 0.5)
        return result
    
    # Fallback to WalkSAT
    while time.time() < deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = compute_clause_sat_count(assignment)
        
        if is_satisfied(clause_sat_count):
            return assignment
        
        steps = 0
        current_noise = base_noise
        
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                return assignment
            
            if is_satisfied(clause_sat_count):
                return assignment
            
            if steps > 0 and steps % 200 == 0:
                current_noise = min(max_noise, current_noise + 0.05)
            
            walksat_step(assignment, clause_sat_count, rng, current_noise)
            steps += 1
    
    return assignment
