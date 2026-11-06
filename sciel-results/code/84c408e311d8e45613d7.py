import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 290  # Safety margin

    # 1. Preprocessing: Unit Propagation and Pure Literal Elimination
    # We work with a list of clauses, where each clause is a list of literals.
    # We maintain a set of assignments: var -> 1 or -1
    assignments = {}
    clauses = [list(c) for c in clauses]
    
    def unit_propagate():
        """
        Applies unit propagation. Returns False if contradiction, True if consistent.
        Modifies 'clauses' and 'assignments' in place.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                unit_lit = None
                unassigned_lits = []
                
                for lit in clause:
                    var = abs(lit)
                    sign = 1 if lit > 0 else -1
                    if var in assignments:
                        if assignments[var] == sign:
                            sat = True
                            break
                    else:
                        unassigned_lits.append(lit)
                
                if sat:
                    continue  # Clause is satisfied, remove it
                
                if len(unassigned_lits) == 0:
                    return False  # Contradiction: clause is false
                
                if len(unassigned_lits) == 1:
                    unit_lit = unassigned_lits[0]
                    var = abs(unit_lit)
                    sign = 1 if unit_lit > 0 else -1
                    if var in assignments:
                        if assignments[var] != sign:
                            return False
                    else:
                        assignments[var] = sign
                        changed = True
                    # Clause becomes empty (satisfied by the unit literal assignment), so we don't add it to new_clauses
                    continue
                
                new_clauses.append(unassigned_lits)
            
            clauses[:] = new_clauses
        
        return True

    # Apply initial unit propagation
    if not unit_propagate():
        # If contradiction, the instance is UNSAT (but task says satisfiable, so this shouldn't happen unless bug)
        # Or we made a mistake. Return random.
        return [rng.random() < 0.5 for _ in range(n)]

    # Pure Literal Elimination
    # A literal is pure if its variable appears only with that sign in the remaining clauses.
    # If we assign a pure literal to True, we satisfy all clauses containing it and don't falsify any.
    def pure_literal_elimination():
        changed = True
        while changed:
            changed = False
            # Count occurrences of each literal
            pos_count = [0] * (n + 1)
            neg_count = [0] * (n + 1)
            
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
            
            new_clauses = []
            for clause in clauses:
                # Check if any literal in this clause is pure
                pure_lit = None
                for lit in clause:
                    var = abs(lit)
                    if lit > 0:
                        if neg_count[var] == 0 and var not in assignments:
                            pure_lit = lit
                            break
                    else:
                        if pos_count[var] == 0 and var not in assignments:
                            pure_lit = lit
                            break
                
                if pure_lit is not None:
                    var = abs(pure_lit)
                    sign = 1 if pure_lit > 0 else -1
                    assignments[var] = sign
                    changed = True
                    # Clause is satisfied, so we don't add it
                    continue
                else:
                    # Remove any assigned literals that are false? 
                    # Actually, if a literal is assigned False, it's already removed in UP.
                    # If assigned True, it would have been removed in UP.
                    # So all literals in 'clauses' are unassigned.
                    new_clauses.append(clause)
            
            clauses[:] = new_clauses

    pure_literal_elimination()
    
    # If all variables assigned, we are done
    if len(assignments) == n:
        return [assignments.get(i+1, 1) == 1 for i in range(n)]

    # If no clauses left, we are done (all remaining vars are free)
    if len(clauses) == 0:
        return [assignments.get(i+1, 1) == 1 for i in range(n)]

    # 2. WalkSAT on the remaining sub-problem
    # Identify unassigned variables
    unassigned_vars = [i for i in range(1, n+1) if i not in assignments]
    # Map original var indices to 0..k-1 for the sub-problem?
    # It's easier to stick to original indices but only consider unassigned ones.
    
    m = len(clauses)
    if m == 0:
        return [assignments.get(i+1, 1) == 1 for i in range(n)]

    # Precompute matrices for the remaining clauses
    # We only care about the unassigned variables.
    # Let's create a mapping from original var to index in the sub-problem array for speed?
    # Or just use the full n array and ignore assigned ones.
    # Since n=50, full array is fine.
    
    Pos = np.zeros((m, n), dtype=np.int8)
    Neg = np.zeros((m, n), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                Pos[i, var_idx] = 1
            else:
                Neg[i, var_idx] = 1

    def count_violated(A):
        """A is a numpy boolean array of size n."""
        A_int = A.astype(np.int8)
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        return np.sum(SatCount == 0)

    def solve_walksat(A, max_steps):
        """
        A: numpy boolean array of size n
        max_steps: int
        Returns True if solved, False otherwise.
        """
        A_int = A.astype(np.int8)
        # Initial SatCount
        SatCount = np.sum(Pos * A_int, axis=1) + np.sum(Neg * (1 - A_int), axis=1)
        
        violated_indices = np.where(SatCount == 0)[0]
        
        if len(violated_indices) == 0:
            return True

        p_greedy = 0.5
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if len(violated_indices) == 0:
                return True

            # Pick a random violated clause
            ci = rng.choice(violated_indices.tolist())
            clause = clauses[ci]
            
            # Candidates are the variables in this clause
            # Compute gain for each
            gains = []
            
            for lit in clause:
                var_idx = abs(lit) - 1
                # We should only flip unassigned variables? 
                # In the sub-problem, all vars in 'clauses' are unassigned.
                
                if A_int[var_idx] == 0:
                    Delta = Pos[:, var_idx] - Neg[:, var_idx]
                else:
                    Delta = Neg[:, var_idx] - Pos[:, var_idx]
                
                become_sat = np.sum((SatCount == 0) & (Delta == 1))
                become_violated = np.sum((SatCount == 1) & (Delta == -1))
                gain = become_sat - become_violated
                
                gains.append((gain, var_idx))
            
            # Selection
            if rng.random() < p_greedy:
                # Greedy
                gains.sort(key=lambda x: -x[0])
                best_var = gains[0][1]
            else:
                # Random from clause
                best_var = rng.choice([abs(lit)-1 for lit in clause])
            
            # Update A and A_int
            A_int[best_var] = 1 - A_int[best_var]
            A[best_var] = bool(A_int[best_var])
            
            # Update SatCount
            if A_int[best_var] == 1:
                Delta = Pos[:, best_var] - Neg[:, best_var]
            else:
                Delta = Neg[:, best_var] - Pos[:, best_var]
                
            SatCount = SatCount + Delta
            violated_indices = np.where(SatCount == 0)[0]

        return False

    # Main loop with restarts
    best_assignment = None
    best_violated = m + 1
    
    max_restarts = 5000
    base_steps = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment for unassigned vars
        A = np.zeros(n, dtype=bool)
        for var in unassigned_vars:
            A[var-1] = rng.random() < 0.5
        
        # Try WalkSAT
        # Geometric distribution for steps?
        # steps = int(rng.expovariate(1.0 / base_steps))
        steps = base_steps + rng.randint(0, 5000)
        
        if solve_walksat(A, steps):
            # Combine with initial assignments
            final_A = A.copy()
            for var, val in assignments.items():
                final_A[var-1] = (val == 1)
            return final_A.tolist()
            
        # Track best
        v = count_violated(A)
        if v < best_violated:
            best_violated = v
            best_assignment = A.copy()

    if best_assignment is not None:
        final_A = best_assignment.copy()
        for var, val in assignments.items():
            final_A[var-1] = (val == 1)
        return final_A.tolist()
        
    return [assignments.get(i+1, 1) == 1 for i in range(n)]
