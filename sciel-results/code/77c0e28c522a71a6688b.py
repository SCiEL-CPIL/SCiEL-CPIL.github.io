import time
import random
import sys

# Set recursion limit high just in case, though n=50 is small
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 300s limit, use 290s for safety
    deadline = time.time() + 290.0
    
    # Internal representation:
    # Variables are 0..n-1
    # Literals are signed: positive means True, negative means False
    # We maintain:
    # 1. assignment: list of 0 (unassigned), 1 (True), -1 (False)
    # 2. watches: dict mapping literal -> list of clause indices watching it
    # 3. clause_watches: list of sets of literals watched by each clause
    
    # Preprocessing: Normalize clauses and setup watches
    # We assume input literals are 1-based.
    # Convert to 0-based internal representation for logic, but keep mapping simple.
    # Actually, let's stick to 1-based literals in the logic to avoid conversion errors, 
    # but indices in arrays are 0-based.
    
    # watches: key is literal (int), value is list of clause indices
    watches = {}
    clause_watches = []
    
    for i, clause in enumerate(clauses):
        # Ensure we have at least 2 watches per clause
        # If clause has 1 literal, watch it twice? Or just handle unit propagation separately.
        # Standard watched literals: watch 2 literals per clause.
        if len(clause) < 2:
            # Unit clause or empty
            # For simplicity in the main loop, we handle unit propagation explicitly.
            # If empty, UNSAT. If unit, we must propagate.
            pass
            
        if len(clause) >= 2:
            # Pick first two distinct literals if possible, or just first two
            w1 = clause[0]
            w2 = clause[1] if len(clause) > 1 else clause[0]
            
            if w1 not in watches:
                watches[w1] = []
            if w2 not in watches:
                watches[w2] = []
                
            watches[w1].append(i)
            watches[w2].append(i)
            clause_watches.append({w1, w2})
        else:
            # Unit clause: watch the single literal twice? 
            # Or just keep it in a unit queue.
            # Let's add it to watches for uniformity, handling the "re-watching" logic carefully.
            # Actually, standard implementation: if clause size 1, it's a unit clause.
            # We'll handle unit clauses in a separate list/queue for efficiency.
            clause_watches.append(set()) # No watches for unit clauses in the main watch list initially
            # We will process all unit clauses at the start.

    assignment = [0] * (n + 1) # 1-based index for assignment
    
    # Queue for unit propagation
    # We will maintain a list of clauses that need checking.
    # A more robust approach for small N:
    # Use a standard DPLL with efficient clause evaluation.
    # Given N=50, a simpler but optimized recursive DPLL without complex watched literals
    # might be sufficient and less prone to bugs, but watched literals are better.
    
    # Let's implement a robust DPLL with:
    # 1. Unit Propagation
    # 2. Pure Literal Elimination
    # 3. Variable selection based on frequency
    
    # Helper to check if clause is satisfied
    def is_satisfied(clause, assign):
        for l in clause:
            var = abs(l)
            val = assign[var]
            if val == 0:
                continue
            if (l > 0 and val == 1) or (l < 0 and val == -1):
                return True
        return False

    # Helper to check if clause is falsified
    def is_falsified(clause, assign):
        for l in clause:
            var = abs(l)
            val = assign[var]
            if val == 0:
                return False
            if (l > 0 and val == -1) or (l < 0 and val == 1):
                return False # This literal is true, so clause is not falsified
        return True

    # Helper to find unit clauses
    def get_unit_clauses(clauses, assign):
        units = []
        for c in clauses:
            unassigned = [l for l in c if assign[abs(l)] == 0]
            if len(unassigned) == 1:
                units.append(unassigned[0])
        return units

    # Helper to find pure literals
    def find_pure_literals(clauses, assign, n):
        # pos[var] = count of positive occurrences in unassigned clauses
        # neg[var] = count of negative occurrences in unassigned clauses
        pos = [0] * (n + 1)
        neg = [0] * (n + 1)
        
        active_clauses = []
        for c in clauses:
            # If clause is already satisfied, ignore
            if is_satisfied(c, assign):
                continue
            # If clause is falsified, we have a conflict (shouldn't happen if UP is correct)
            if is_falsified(c, assign):
                return None, active_clauses # Conflict detected
            active_clauses.append(c)
            
            for l in c:
                var = abs(l)
                if assign[var] != 0:
                    continue
                if l > 0:
                    pos[var] += 1
                else:
                    neg[var] += 1
        
        pure = []
        for var in range(1, n + 1):
            if assign[var] != 0:
                continue
            if pos[var] == 0 and neg[var] > 0:
                pure.append(-var)
            elif neg[var] == 0 and pos[var] > 0:
                pure.append(var)
        return pure, active_clauses

    # Variable selection heuristic: 
    # Pick variable with highest frequency in the smallest unsatisfied clauses.
    def select_variable(active_clauses, assign, n):
        freq = [0] * (n + 1)
        min_clause_size = float('inf')
        
        # First pass: find min clause size among unsatisfied active clauses
        for c in active_clauses:
            size = 0
            for l in c:
                if assign[abs(l)] == 0:
                    size += 1
            if size < min_clause_size:
                min_clause_size = size
            if size == 1:
                break # Unit clause, should be handled by UP, but just in case
        
        if min_clause_size == float('inf'):
            return None
            
        # Second pass: count frequencies in clauses of min size
        for c in active_clauses:
            size = 0
            for l in c:
                if assign[abs(l)] == 0:
                    size += 1
            if size == min_clause_size:
                for l in c:
                    var = abs(l)
                    if assign[var] == 0:
                        freq[var] += 1
        
        best_var = 0
        max_freq = 0
        for var in range(1, n + 1):
            if assign[var] == 0 and freq[var] > max_freq:
                max_freq = freq[var]
                best_var = var
                
        return best_var

    def dpll(clauses, assign, n, depth=0):
        if time.time() > deadline:
            return None # Timeout, return partial assignment (handled by caller)
            
        # 1. Check for empty clause (conflict)
        # 2. Check for all satisfied (solution)
        # 3. Unit Propagation
        # 4. Pure Literal Elimination
        # 5. Branch
        
        # We'll manage the state by modifying assign in place and backtracking.
        
        # Step 1: Unit Propagation & Pure Literal Elimination
        # We need to iterate until fixed point.
        changed = True
        while changed:
            changed = False
            
            # Check for conflict: empty clause
            for c in clauses:
                if is_falsified(c, assign):
                    return False # Conflict
            
            # Check for solution
            all_assigned = True
            for var in range(1, n + 1):
                if assign[var] == 0:
                    all_assigned = False
                    break
            if all_assigned:
                return True # All satisfied (since no conflict)

            # Unit Propagation
            units = get_unit_clauses(clauses, assign)
            for lit in units:
                var = abs(lit)
                val = 1 if lit > 0 else -1
                if assign[var] == 0:
                    assign[var] = val
                    changed = True
                elif assign[var] != val:
                    return False # Conflict
                
            # Pure Literal Elimination
            pure, active_clauses = find_pure_literals(clauses, assign, n)
            if pure is None:
                return False # Conflict found during PLE check
            
            for lit in pure:
                var = abs(lit)
                val = 1 if lit > 0 else -1
                if assign[var] == 0:
                    assign[var] = val
                    changed = True
        
        # If we get here, no more UP/PLE changes.
        # If all assigned, we already returned True above.
        # So there must be unassigned variables.
        
        # Step 2: Variable Selection
        # We need the list of active (unsatisfied) clauses for heuristic
        active_clauses = []
        for c in clauses:
            if not is_satisfied(c, assign):
                active_clauses.append(c)
        
        var = select_variable(active_clauses, assign, n)
        if var == 0:
            # No variable to choose? All assigned?
            return True
            
        # Try True
        assign[var] = 1
        if dpll(clauses, assign, n, depth + 1):
            return True
        assign[var] = 0 # Backtrack
        
        # Try False
        assign[var] = -1
        if dpll(clauses, assign, n, depth + 1):
            return True
        assign[var] = 0 # Backtrack
        
        return False # Conflict

    # Initialize
    initial_assign = [0] * (n + 1)
    
    # Run DPLL
    success = dpll(clauses, initial_assign, n)
    
    # If success, return the assignment
    # If failure (UNSAT or timeout), return the current best partial assignment
    # The problem states "planted satisfiable", so we expect SAT.
    # If timeout, we return the current state of initial_assign.
    
    result = [False] * n
    for i in range(1, n + 1):
        if initial_assign[i] == 1:
            result[i-1] = True
        else:
            # If unassigned (0), we can guess randomly or default to False.
            # Since we expect SAT, if we timed out, the partial assignment might not satisfy all.
            # But we return the best effort.
            result[i-1] = False
            
    return result
