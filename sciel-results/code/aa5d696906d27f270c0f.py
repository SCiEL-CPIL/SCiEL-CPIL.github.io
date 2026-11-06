import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit (use 110 to be safe)
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Build SAT instance
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # For each variable i (triple i), var i is True if blue (1), False if red (0)
        # Constraint 1: No red K4. For each 4-set {a,b,c,d}, not all 4 triples red.
        #   i.e., at least one of the 4 triples is blue.
        #   Clause: (x_{abc} OR x_{abd} OR x_{acd} OR x_{bcd})
        # Constraint 2: No blue K5. For each 5-set, not all 10 triples blue.
        #   i.e., at least one of the 10 triples is red.
        #   Clause: (NOT x_{abc} OR NOT x_{abd} OR ... OR NOT x_{ijk})
        
        clauses = []
        
        # Red K4 constraints: for each 4-set, OR of the 4 triples (must be blue)
        for quad in combinations(range(n), 4):
            triple_list = list(combinations(quad, 3))
            clause = []
            for t in triple_list:
                clause.append(triple_index[t] + 1)  # positive literal
            clauses.append(clause)
        
        # Blue K5 constraints: for each 5-set, OR of negations of the 10 triples (must be red)
        for quint in combinations(range(n), 5):
            triple_list = list(combinations(quint, 3))
            clause = []
            for t in triple_list:
                clause.append(-(triple_index[t] + 1))  # negative literal
            clauses.append(clause)
        
        num_clauses = len(clauses)
        
        # Solve using DPLL with unit propagation and pure literal heuristic
        # with randomized restarts
        best_result = None
        
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            assignment = solve_sat_dpll(num_triples, clauses, deadline, restart)
            if assignment is not None:
                # Convert assignment to colors
                # assignment[i] = 1 if blue, 0 if red
                colors = []
                for i in range(num_triples):
                    if i < len(assignment) and assignment[i]:
                        colors.append(1)
                    else:
                        colors.append(0)
                
                # Verify validity
                if verify_coloring(n, colors, triples, triple_index):
                    return {'n': n, 'colors': colors}
        
        # If SAT didn't find a solution for this n, try smaller n
    
    # Fallback: return a valid small coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}


def verify_coloring(n, colors, triples, triple_index):
    """Verify that the coloring has no red K4 and no blue K5."""
    # Check red K4s
    for quad in combinations(range(n), 4):
        all_red = True
        for t in combinations(quad, 3):
            if colors[triple_index[t]] != 0:
                all_red = False
                break
        if all_red:
            return False
    
    # Check blue K5s
    for quint in combinations(range(n), 5):
        all_blue = True
        for t in combinations(quint, 3):
            if colors[triple_index[t]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True


def solve_sat_dpll(num_vars, clauses, deadline, restart_seed):
    """
    Solve SAT using DPLL with unit propagation and pure literal heuristic.
    Returns a list of booleans (assignment) or None if timeout.
    Variables: 0..num_vars-1
    Clauses: list of lists of integers (positive = variable True, negative = variable False)
    """
    random.seed(restart_seed * 12345 + 67890)
    
    # Build clause index: for each variable, which clauses it appears in
    var_to_clauses = [[] for _ in range(num_vars)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            var_to_clauses[v].append(ci)
    
    # Initial state
    assignment = [None] * num_vars  # None = unassigned, True/False = assigned
    clause_status = [0] * len(clauses)  # 0 = active, 1 = satisfied, -1 = falsified
    
    # Unit propagation queue
    def unit_propagate(assignment, clause_status, clauses, var_to_clauses):
        """Perform unit propagation. Returns (assignment, clause_status, success)"""
        changed = True
        while changed:
            changed = False
            for ci in range(len(clauses)):
                if clause_status[ci] != 0:
                    continue
                clause = clauses[ci]
                # Check if unit
                unassigned_count = 0
                unit_var = None
                falsified = False
                for lit in clause:
                    v = abs(lit) - 1
                    if assignment[v] is None:
                        unassigned_count += 1
                        unit_var = v
                    elif (lit > 0 and assignment[v] == False) or (lit < 0 and assignment[v] == True):
                        falsified = True
                        break
                if falsified:
                    return assignment, clause_status, False
                if unassigned_count == 1:
                    # Assign unit_var
                    lit = None
                    for l in clause:
                        if abs(l) - 1 == unit_var:
                            lit = l
                            break
                    if lit > 0:
                        assignment[unit_var] = True
                    else:
                        assignment[unit_var] = False
                    changed = True
                    # Update clause statuses
                    for cj in var_to_clauses[unit_var]:
                        if clause_status[cj] != 0:
                            continue
                        c = clauses[cj]
                        for l in c:
                            v = abs(l) - 1
                            if v == unit_var:
                                if (l > 0 and assignment[v] == True) or (l < 0 and assignment[v] == False):
                                    clause_status[cj] = 1  # satisfied
                                else:
                                    # Check if falsified
                                    all_false = True
                                    for l2 in c:
                                        v2 = abs(l2) - 1
                                        if assignment[v2] is None:
                                            all_false = False
                                            break
                                        elif (l2 > 0 and assignment[v2] == False) or (l2 < 0 and assignment[v2] == True):
                                            all_false = False
                                            break
                                    if all_false:
                                        clause_status[cj] = -1
                    break  # Restart scan after assignment
        
        # Final check for falsified clauses
        for ci in range(len(clauses)):
            if clause_status[ci] != 0:
                continue
            clause = clauses[ci]
            all_false = True
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] is None:
                    all_false = False
                    break
                elif (lit > 0 and assignment[v] == False) or (lit < 0 and assignment[v] == True):
                    all_false = False
                    break
            if all_false:
                return assignment, clause_status, False
        
        return assignment, clause_status, True
    
    # Pure literal heuristic
    def pure_literal_heuristic(assignment, clause_status, clauses, var_to_clauses):
        """Find pure literals and assign them."""
        changed = True
        while changed:
            changed = False
            for v in range(num_vars):
                if assignment[v] is not None:
                    continue
                # Check if pure
                has_pos = False
                has_neg = False
                for ci in var_to_clauses[v]:
                    if clause_status[ci] != 0:
                        continue
                    for lit in clauses[ci]:
                        if abs(lit) - 1 == v:
                            if lit > 0:
                                has_pos = True
                            else:
                                has_neg = True
                            break
                    if has_pos and has_neg:
                        break
                if has_pos and not has_neg:
                    assignment[v] = True
                    changed = True
                    update_clause_status(v, True, clause_status, clauses, var_to_clauses)
                    break
                elif has_neg and not has_pos:
                    assignment[v] = False
                    changed = True
                    update_clause_status(v, False, clause_status, clauses, var_to_clauses)
                    break
    
    def update_clause_status(var, val, clause_status, clauses, var_to_clauses):
        """Update clause statuses after assigning a variable."""
        for ci in var_to_clauses[var]:
            if clause_status[ci] != 0:
                continue
            clause = clauses[ci]
            for lit in clause:
                v = abs(lit) - 1
                if v == var:
                    if (lit > 0 and val == True) or (lit < 0 and val == False):
                        clause_status[ci] = 1  # satisfied
                    else:
                        # Check if all other literals are false
                        all_false = True
                        for l2 in clause:
                            v2 = abs(l2) - 1
                            if v2 == var:
                                continue
                            if assignment[v2] is None:
                                all_false = False
                                break
                            elif (l2 > 0 and assignment[v2] == False) or (l2 < 0 and assignment[v2] == True):
                                all_false = False
                                break
                        if all_false:
                            clause_status[ci] = -1
                    break
    
    # DPLL with backtracking
    def dpll(assignment, clause_status):
        """Recursive DPLL. Returns assignment or None."""
        if time.time() > deadline - 2:
            return None
        
        # Unit propagation
        assignment, clause_status, success = unit_propagate(assignment, clause_status, clauses, var_to_clauses)
        if not success:
            return None
        
        # Check if all clauses satisfied
        for ci in range(len(clauses)):
            if clause_status[ci] == -1:
                return None
        
        # Check if all variables assigned
        all_assigned = True
        for v in range(num_vars):
            if assignment[v] is None:
                all_assigned = False
                break
        if all_assigned:
            return assignment[:]
        
        # Pure literal heuristic
        pure_literal_heuristic(assignment, clause_status, clauses, var_to_clauses)
        
        # After pure literal, check again
        assignment, clause_status, success = unit_propagate(assignment, clause_status, clauses, var_to_clauses)
        if not success:
            return None
        
        all_assigned = True
        for v in range(num_vars):
            if assignment[v] is None:
                all_assigned = False
                break
        if all_assigned:
            return assignment[:]
        
        # Choose a variable to branch on
        unassigned = [v for v in range(num_vars) if assignment[v] is None]
        if not unassigned:
            return assignment[:]
        
        # Heuristic: choose variable with highest degree in active clauses
        best_var = unassigned[0]
        best_score = -1
        for v in unassigned:
            score = 0
            for ci in var_to_clauses[v]:
                if clause_status[ci] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_var = v
        
        # Try True first (or randomize)
        if random.random() < 0.5:
            assignment[best_var] = True
            result = dpll(assignment, clause_status)
            if result is not None:
                return result
            assignment[best_var] = False
            result = dpll(assignment, clause_status)
            if result is not None:
                return result
        else:
            assignment[best_var] = False
            result = dpll(assignment, clause_status)
            if result is not None:
                return result
            assignment[best_var] = True
            result = dpll(assignment, clause_status)
            if result is not None:
                return result
        
        # Backtrack
        assignment[best_var] = None
        return None
    
    # Run DPLL
    result = dpll(assignment, clause_status)
    if result is not None:
        # Verify all clauses are satisfied
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if (lit > 0 and result[v]) or (lit < 0 and not result[v]):
                    sat = True
                    break
            if not sat:
                return None
        return result
    
    return None
