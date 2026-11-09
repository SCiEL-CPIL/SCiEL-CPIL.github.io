import random
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 280.0  # Leave 20s margin for safety
    
    # Pre-process clauses: convert to 0-based indices and signs
    # clause is list of signed 1-based literals
    # e.g., [3, -2, 1] -> var 2, 1, 0 with signs +, -, +
    # We store (var_idx, sign) where sign is 1 for positive, -1 for negative
    # assignment[i] is True/False. 
    # Literal (v, s) is satisfied if assignment[v] == (s == 1)
    
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pc.append((var, sign))
        processed_clauses.append(pc)
    
    # Calculate literal frequencies for phase-biased initialization
    # Count how often each variable appears with each sign
    var_count_pos = [0] * n
    var_count_neg = [0] * n
    for c in processed_clauses:
        for var, sign in c:
            if sign == 1:
                var_count_pos[var] += 1
            else:
                var_count_neg[var] += 1
    
    def generate_initial_assignment(rng):
        """
        Phase-biased initialization.
        If a variable appears mostly positive, set it to True.
        If mostly negative, set it to False.
        If equal, random.
        Add some noise to avoid getting stuck in the same local minima on restarts.
        """
        assignment = [False] * n
        for i in range(n):
            pos = var_count_pos[i]
            neg = var_count_neg[i]
            if pos > neg:
                # Bias towards True
                assignment[i] = rng.random() > 0.1  # 90% True
            elif neg > pos:
                # Bias towards False
                assignment[i] = rng.random() < 0.1  # 10% True
            else:
                assignment[i] = rng.random() < 0.5
        return assignment

    def count_unsat(assignment):
        """
        Returns the number of unsatisfied clauses and a list of unsatisfied clause indices.
        """
        unsat_clauses = []
        count = 0
        for idx, c in enumerate(processed_clauses):
            sat = False
            for var, sign in c:
                # Satisfied if assignment[var] matches the sign
                if (sign == 1 and assignment[var]) or (sign == -1 and not assignment[var]):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(idx)
                count += 1
        return count, unsat_clauses

    def get_conflict_counts(assignment, unsat_indices):
        """
        For each variable, count how many unsatisfied clauses it appears in.
        """
        conflicts = [0] * n
        for idx in unsat_indices:
            c = processed_clauses[idx]
            for var, sign in c:
                conflicts[var] += 1
        return conflicts

    def min_conflicts_step(assignment, unsat_indices, rng):
        """
        Pick an unsatisfied clause.
        Among the variables in that clause, pick the one with the highest conflict count (global heuristic) 
        or just random from the clause (local heuristic).
        Weighted min-conflicts often uses global conflict counts to guide the flip.
        We will use the variable in the chosen clause that has the highest total conflicts.
        """
        if not unsat_indices:
            return assignment, True
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices)
        clause = processed_clauses[clause_idx]
        
        # Calculate conflicts for variables in this clause
        # We need conflict counts. To be efficient, we can recompute for just these vars 
        # or maintain a global array. Given n=150 and potentially many clauses, 
        # recomputing global conflicts every step is O(M*N) which might be slow if M is large.
        # However, for 150 vars, even O(N^2) per step is small.
        # Let's maintain a global conflict array updated incrementally? 
        # That's complex. Let's try the simpler approach first: 
        # Pick a variable from the clause that is currently False in a way that makes it True?
        # No, Min-Conflicts flips a variable to see if it reduces total unsat.
        
        # Heuristic: Flip the variable in the clause that appears in the most unsatisfied clauses.
        # We need conflict counts.
        # Let's compute conflicts for all vars once per step? 
        # If we have 1000 unsat clauses, and 150 vars, scanning all unsat clauses to build conflict array is O(M_unsat * 3).
        # Then picking best var is O(3).
        # This is efficient.
        
        conflicts = [0] * n
        for idx in unsat_indices:
            c = processed_clauses[idx]
            for var, sign in c:
                conflicts[var] += 1
        
        # Find the variable in the current clause with the max conflict count
        best_var = None
        max_conf = -1
        for var, sign in clause:
            if conflicts[var] > max_conf:
                max_conf = conflicts[var]
                best_var = var
        
        if best_var is None:
            best_var = clause[0][0]
            
        # Flip it
        assignment[best_var] = not assignment[best_var]
        
        return assignment, False

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 1000
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = generate_initial_assignment(rng)
        unsat_count, unsat_indices = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Local search
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # If we found a better assignment, update best
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            # Perform a min-conflicts step
            assignment, solved = min_conflicts_step(assignment, unsat_indices, rng)
            
            if solved:
                return assignment
            
            # Update unsat count and indices
            # Instead of recounting all, we can just recount. 
            # For n=150, recounting is fast.
            unsat_count, unsat_indices = count_unsat(assignment)
            
            if unsat_count == 0:
                return assignment

            # Stagnation check: if no improvement for a long time, break early to restart
            # Simple heuristic: if we've done many steps without reducing unsat_count significantly, restart
            # This is handled by the fixed max_steps_per_restart.
            
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return random
    return [random.random() < 0.5 for _ in range(n)]
