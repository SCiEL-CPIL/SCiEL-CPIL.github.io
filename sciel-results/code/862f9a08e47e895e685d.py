# EVOLVE-BLOCK-START
import random
import sys
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess: convert clauses to lists of signed literals (1-based)
    # We'll use a DPLL solver with VSIDS, unit propagation, and failed literal probing.
    
    # Since we have a time limit of 300s and need to solve multiple instances,
    # we'll implement a robust DPLL with activity-based variable ordering.
    
    # Data structures for DPLL
    # We'll represent the formula as a list of clauses, where each clause is a list of literals.
    # Literals are 1-based signed integers.
    
    # For efficiency, we'll maintain:
    # - A list of clauses (active clauses)
    # - Watched literals scheme for unit propagation
    # - Variable activity for VSIDS
    # - Decision stack for backtracking
    
    # Given the constraints (150 variables), a well-implemented DPLL should handle
    # planted SAT instances very quickly, especially with unit propagation.
    
    # Let's implement a simplified but effective DPLL:
    
    # 1. Convert to 0-based indexing internally
    # 2. Implement unit propagation with watched literals
    # 3. VSIDS for variable selection
    # 4. Clause learning (basic form)
    # 5. Failed literal probing for initial variable ordering
    
    def lit_to_var(l):
        return abs(l) - 1
    
    def lit_to_val(l):
        return l > 0
    
    # Build watched literals structure
    # watches[var] = list of (clause_idx, other_lit) where var is watched in clause_idx
    
    m = len(clauses)
    
    # Convert clauses to 0-based
    clauses0 = [[lit for lit in clause] for clause in clauses]
    
    # Initialize watches
    watches = [[] for _ in range(n)]
    
    # For each clause, pick two literals to watch
    for ci, clause in enumerate(clauses0):
        if len(clause) == 0:
            return None  # Empty clause, unsat
        if len(clause) == 1:
            watches[lit_to_var(clause[0])].append((ci, None))
        else:
            l1, l2 = clause[0], clause[1]
            watches[lit_to_var(l1)].append((ci, l2))
            watches[lit_to_var(l2)].append((ci, l1))
    
    # Assignment: None, True, False
    assignment = [None] * n
    trail = []  # Stack of (var, value) for backtracking
    trail_lim = []  # Decision levels
    
    # VSIDS activities
    activity = [1.0 / (i + 1) for i in range(n)]
    var_inc = 1.0
    var_decr = 1.0 / 2.0
    max_var = 0
    
    # Clause learning
    learnt_clauses = []
    learnt_watches = [[] for _ in range(n)]
    learnt_count = 0
    
    # Decision level
    dl = 0
    
    def propagate():
        """Unit propagation. Returns False if conflict, True otherwise."""
        while len(trail) > (trail_lim[-1] if trail_lim else 0):
            var = trail[-1][0]
            val = trail[-1][1]
            trail.pop()
            
            # Process watches for var
            i = 0
            wlist = watches[var]
            while i < len(wlist):
                ci, other = wlist[i]
                clause = clauses0[ci]
                
                # Check if clause is satisfied
                sat = False
                for l in clause:
                    v = lit_to_var(l)
                    if assignment[v] is not None and assignment[v] == lit_to_val(l):
                        sat = True
                        break
                
                if sat:
                    # Clause is satisfied, keep watching
                    i += 1
                    continue
                
                # Clause is not satisfied. Check if other literal is assigned
                if other is not None:
                    ov = lit_to_var(other)
                    if assignment[ov] is not None and assignment[ov] == lit_to_val(other):
                        # Other literal is satisfied, keep watching
                        i += 1
                        continue
                
                # Clause is not satisfied and other literal is not assigned (or is False)
                # This is a unit clause or conflict
                
                if other is not None and assignment[lit_to_var(other)] is None:
                    # Unit clause: assign other
                    assignment[lit_to_var(other)] = lit_to_val(other)
                    trail.append((lit_to_var(other), lit_to_val(other)))
                    # Move watch to new unassigned literal
                    # Find a new literal in the clause that is unassigned
                    new_other = None
                    for l in clause:
                        if l != other and assignment[lit_to_var(l)] is None:
                            new_other = l
                            break
                    if new_other is None:
                        # All other literals are assigned False, this is a conflict
                        # But we should have caught this above
                        return False
                    # Update watch
                    wlist[i] = (ci, new_other)
                    watches[lit_to_var(new_other)].append((ci, other))
                else:
                    # Conflict: all literals in clause are False
                    return False
            # Remove processed watches from var's list
            # We need to be careful here; let's use a different approach
            
        return True
    
    # The above propagate is getting complicated. Let me use a simpler but still effective approach.
    # Given only 150 variables, we can use a straightforward DPLL with unit propagation
    # that scans all clauses each time (O(n*m) per propagation step, which is fine for 150 vars).
    
    def solve_simple():
        """Simple DPLL with unit propagation and VSIDS."""
        # State: assignment is a dict from var to bool, or None
        # We'll use a recursive approach with backtracking
        
        # For simplicity and reliability, let's use a different strategy:
        # Since these are planted SAT instances, they should be relatively easy.
        # Let's use WalkSAT with a better heuristic, or a simple DPLL.
        
        # Actually, let's implement a clean DPLL:
        
        # assignment: list of None/True/False
        # We'll do unit propagation by scanning all clauses
        
        def unit_propagate(assignment):
            """Perform unit propagation. Returns (assignment, conflict, unit_vars)."""
            changed = True
            while changed:
                changed = False
                for clause in clauses0:
                    # Check clause status
                    unassigned = []
                    false_count = 0
                    for l in clause:
                        v = lit_to_var(l)
                        if assignment[v] is None:
                            unassigned.append(l)
                        elif assignment[v] != lit_to_val(l):
                            false_count += 1
                    
                    if false_count == len(clause):
                        return assignment, True, []  # Conflict
                    if len(unassigned) == 1 and false_count == len(clause) - 1:
                        # Unit clause
                        l = unassigned[0]
                        v = lit_to_var(l)
                        assignment[v] = lit_to_val(l)
                        changed = True
            return assignment, False, []
        
        def pick_variable(assignment):
            """Pick a variable using VSIDS-like heuristic."""
            # Count occurrences of unassigned variables in unsatisfied clauses
            scores = [0] * n
            for clause in clauses0:
                # Check if clause is already satisfied
                sat = False
                for l in clause:
                    v = lit_to_var(l)
                    if assignment[v] is not None and assignment[v] == lit_to_val(l):
                        sat = True
                        break
                if sat:
                    continue
                # Count unassigned literals
                for l in clause:
                    v = lit_to_var(l)
                    if assignment[v] is None:
                        scores[v] += 1
            
            # Pick variable with highest score
            best_var = -1
            best_score = -1
            for v in range(n):
                if assignment[v] is None and scores[v] > best_score:
                    best_score = scores[v]
                    best_var = v
            return best_var
        
        def dpll(assignment, decision_level):
            """DPLL recursive solver."""
            # Unit propagation
            assignment, conflict, _ = unit_propagate(assignment)
            if conflict:
                return None
            
            # Check if all clauses are satisfied
            for clause in clauses0:
                sat = False
                for l in clause:
                    v = lit_to_var(l)
                    if assignment[v] is not None and assignment[v] == lit_to_val(l):
                        sat = True
                        break
                if not sat:
                    break
            else:
                return assignment  # All satisfied
            
            # Pick variable
            var = pick_variable(assignment)
            if var == -1:
                return None  # No unassigned variable but not all satisfied?
            
            # Try True
            assignment[var] = True
            result = dpll(assignment, decision_level + 1)
            if result is not None:
                return result
            
            # Try False
            assignment[var] = False
            result = dpll(assignment, decision_level + 1)
            if result is not None:
                return result
            
            # Backtrack
            assignment[var] = None
            return None
        
        # Initial assignment
        assignment = [None] * n
        result = dpll(assignment, 0)
        if result is not None:
            return [bool(x) for x in result]
        
        return None
    
    # Try the simple DPLL first
    result = solve_simple()
    if result is not None:
        return result
    
    # Fallback: WalkSAT with more iterations
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    for _ in range(500000):
        unsat = [c for c in clauses0 if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        # Weighted random literal selection
        weights = []
        for l in clause:
            v = lit_to_var(l)
            # Count how many clauses this literal appears in
            count = sum(1 for c in clauses0 if l in c)
            weights.append(count)
        total = sum(weights)
        r = rng.random() * total
        cum = 0
        chosen = clause[0]
        for l, w in zip(clause, weights):
            cum += w
            if r <= cum:
                chosen = l
                break
        v = lit_to_var(chosen)
        assignment[v] = not assignment[v]
    
    # Check if solved
    if all(satisfied(c) for c in clauses0):
        return assignment
    
    return assignment
# EVOLVE-BLOCK-END
