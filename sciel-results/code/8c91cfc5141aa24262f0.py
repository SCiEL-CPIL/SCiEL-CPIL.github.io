"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocessing
    # Convert to 0-based indices
    # Store literal as (var_idx, is_positive)
    # But for speed, let's stick to the raw literals and precompute var_to_clauses
    
    # var_to_clauses[i] = list of clause indices where variable i+1 appears
    var_to_clauses = [[] for _ in range(n)]
    # clause_lits[c] = list of literals (signed)
    # We need to know for a clause c, which literals are satisfied.
    
    # Let's use the incremental approach.
    # assignment: list of bools, size n
    # clause_sat: list of ints, size len(clauses), count of satisfied literals
    
    m = len(clauses)
    
    # Precompute variable occurrences
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(c_idx)
            
    # Helper to check if a literal is satisfied given assignment
    # lit > 0: assignment[var] is True
    # lit < 0: assignment[var] is False
    
    # Helper to update clause status when a variable flips
    # We need to know the current status of the clause before and after.
    
    # Let's maintain:
    # assignment: list of bools
    # clause_sat: list of ints (0, 1, 2, 3)
    # unsat_clauses: set of indices where clause_sat[c] == 0
    
    # Initialize
    rng = random.Random(0)
    
    def init_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
        
    def compute_initial_state(assignment):
        clause_sat = [0] * m
        unsat_clauses = set()
        for c_idx, clause in enumerate(clauses):
            sat_count = 0
            for lit in clause:
                var_idx = abs(lit) - 1
                if (lit > 0 and assignment[var_idx]) or (lit < 0 and not assignment[var_idx]):
                    sat_count += 1
            clause_sat[c_idx] = sat_count
            if sat_count == 0:
                unsat_clauses.add(c_idx)
        return clause_sat, unsat_clauses

    def flip_var(v_idx, assignment, clause_sat, unsat_clauses):
        """Flip variable v_idx and update state. Returns new state components."""
        old_val = assignment[v_idx]
        new_val = not old_val
        assignment[v_idx] = new_val
        
        # Update all clauses containing v_idx
        for c_idx in var_to_clauses[v_idx]:
            clause = clauses[c_idx]
            old_sat = clause_sat[c_idx]
            
            # Determine if the literal in this clause for v_idx was satisfied before
            lit = None
            for l in clause:
                if abs(l) - 1 == v_idx:
                    lit = l
                    break
            
            # Since it's 3-SAT, we can find it faster, but this is fine for n=150
            # Actually, precomputing the literal position is better.
            # Let's precompute clause_lit_info: for each clause, for each var in it, the literal sign.
            # Or just re-evaluate.
            
            # Check if the literal was satisfied before
            if lit is not None:
                was_sat = (lit > 0 and old_val) or (lit < 0 and not old_val)
                is_sat_now = (lit > 0 and new_val) or (lit < 0 and not new_val)
                
                if was_sat and not is_sat_now:
                    new_sat = old_sat - 1
                elif not was_sat and is_sat_now:
                    new_sat = old_sat + 1
                else:
                    new_sat = old_sat
            else:
                new_sat = old_sat # Should not happen if var_to_clauses is correct
                
            clause_sat[c_idx] = new_sat
            
            if old_sat == 0 and new_sat > 0:
                unsat_clauses.discard(c_idx)
            elif old_sat > 0 and new_sat == 0:
                unsat_clauses.add(c_idx)
                
        return clause_sat, unsat_clauses

    # Precompute literal in clause for faster lookup?
    # For n=150, m~600, degree ~12. The inner loop in flip_var iterates 3 times to find lit.
    # That's 3*12=36 ops per flip. Acceptable.
    
    deadline = time.time() + 290.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 10000
    steps_per_run = 5000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        if best_assignment is None or rng.random() < 0.3:
            assignment = init_assignment()
        else:
            # Perturb best assignment
            assignment = best_assignment[:]
            # Flip ~10% of variables
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                v = rng.randint(0, n-1)
                assignment[v] = not assignment[v]
                
        clause_sat, unsat_clauses = compute_initial_state(assignment)
        
        if not unsat_clauses:
            return assignment
            
        # Track best in this run
        run_best_assignment = assignment[:]
        run_best_unsat = len(unsat_clauses)
        
        for step in range(steps_per_run):
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                # Solved
                return assignment
                
            # Update global best
            if len(unsat_clauses) < run_best_unsat:
                run_best_unsat = len(unsat_clauses)
                run_best_assignment = assignment[:]
                
            # Pick random unsat clause
            c_idx = rng.choice(list(unsat_clauses))
            clause = clauses[c_idx]
            
            # Pick random literal from clause
            # Heuristic: pick literal that satisfies most other clauses?
            # Simple: random
            lit = rng.choice(clause)
            v_idx = abs(lit) - 1
            
            flip_var(v_idx, assignment, clause_sat, unsat_clauses)
            
        # End of run
        if run_best_unsat < best_unsat_count:
            best_unsat_count = run_best_unsat
            best_assignment = run_best_assignment[:]
            
    return best_assignment if best_assignment else init_assignment()

# EVOLVE-BLOCK-END
