import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin

    rng = random.Random(42)

    # Preprocess: 0-based literals
    # clause_vars[c] = list of (var_idx, sign)
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    for c, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((var, sign))
            var_to_clauses[var].append(c)
        clause_vars.append(cv)

    def is_clause_satisfied(c_idx, assignment):
        cv = clause_vars[c_idx]
        for var, sign in cv:
            if (assignment[var] and sign == 1) or (not assignment[var] and sign == -1):
                return True
        return False

    best_assignment = None
    best_unsat_count = m + 1

    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize satisfaction status
        clause_satisfied = [False] * m
        unsat_indices = []
        
        for c in range(m):
            if is_clause_satisfied(c, assignment):
                clause_satisfied[c] = True
            else:
                unsat_indices.append(c)
        
        if not unsat_indices:
            return assignment

        if len(unsat_indices) < best_unsat_count:
            best_unsat_count = len(unsat_indices)
            best_assignment = assignment[:]

        # Local search loop
        steps = 0
        max_steps = 100000  # Cap steps per restart to avoid infinite loops on hard instances
        
        while unsat_indices and steps < max_steps and time.time() < deadline:
            # Pick a random unsatisfied clause
            # Lazy deletion: pick index, if it's actually sat, remove it
            idx = rng.randrange(len(unsat_indices))
            c = unsat_indices[idx]
            
            # Check if c is still unsat (it might have become sat and not removed yet)
            if clause_satisfied[c]:
                # Remove from list (swap with last)
                unsat_indices[idx] = unsat_indices[-1]
                unsat_indices.pop()
                continue
            
            # It is unsat. Pick a random variable in this clause to flip
            cv = clause_vars[c]
            var_idx, _ = rng.choice(cv)
            
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update satisfaction status for all clauses containing var_idx
            for c2 in var_to_clauses[var_idx]:
                new_sat = is_clause_satisfied(c2, assignment)
                old_sat = clause_satisfied[c2]
                
                if new_sat != old_sat:
                    clause_satisfied[c2] = new_sat
                    if new_sat:
                        # Clause became satisfied. 
                        # We don't remove it from unsat_indices immediately (lazy deletion)
                        # It will be filtered out when picked.
                        pass
                    else:
                        # Clause became unsat.
                        # Add to unsat_indices
                        unsat_indices.append(c2)
            
            steps += 1
            
            # Check for solution
            if not unsat_indices:
                return assignment
            
            # Track best
            # Note: len(unsat_indices) is an overestimate due to lazy deletion.
            # For best tracking, we should calculate actual unsat count occasionally or just rely on finding 0.
            # To be safe, we can calculate actual count every 1000 steps or if we suspect improvement.
            # However, calculating actual count is O(m). 
            # Let's just track if we hit 0.
            # For "best assignment" fallback, we can compute actual count less frequently.
            if steps % 1000 == 0:
                actual_unsat = sum(1 for c in range(m) if not clause_satisfied[c])
                if actual_unsat < best_unsat_count:
                    best_unsat_count = actual_unsat
                    best_assignment = assignment[:]

        # End of restart
        # Compute actual unsat count for this final state
        actual_unsat = sum(1 for c in range(m) if not clause_satisfied[c])
        if actual_unsat < best_unsat_count:
            best_unsat_count = actual_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
