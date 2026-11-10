"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit

    # Precompute clause literal indices and signs for fast access
    clause_data = []
    for c in clauses:
        clause_data.append([(abs(l) - 1, l > 0) for l in c])

    def is_satisfied(assignment, ci):
        for var_idx, positive in clause_data[ci]:
            if assignment[var_idx] == positive:
                return True
        return False

    def count_unsat(assignment):
        cnt = 0
        for ci in range(len(clause_data)):
            if not is_satisfied(assignment, ci):
                cnt += 1
        return cnt

    def walk_sat_single(assignment, max_steps, rng):
        """Run WalkSAT from a given assignment. Returns (assignment, unsat_count)."""
        m = len(clause_data)
        # Track unsatisfied clauses
        unsat_clauses = set()
        for ci in range(m):
            if not is_satisfied(assignment, ci):
                unsat_clauses.add(ci)
        
        if not unsat_clauses:
            return assignment, 0

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_clauses:
                return assignment, 0

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clause_data[ci]

            # Find the variable whose flip satisfies the most unsatisfied clauses
            best_var = None
            best_score = -1
            for var_idx, positive in clause:
                # Count how many unsatisfied clauses contain this variable
                score = 0
                for uc in unsat_clauses:
                    for v, p in clause_data[uc]:
                        if v == var_idx:
                            score += 1
                            break
                # Small random tie-breaker
                score += rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var is None:
                break

            # Flip the variable
            old_val = assignment[best_var]
            assignment[best_var] = not old_val

            # Update unsat_clauses: check all clauses containing best_var
            # For efficiency, we recompute affected clauses
            # Clauses that contain best_var may change status
            affected = []
            for uc in unsat_clauses:
                for v, p in clause_data[uc]:
                    if v == best_var:
                        affected.append(uc)
                        break
            # Also, clauses that were satisfied and contain best_var may become unsat
            # We need to check those too, but for simplicity, let's just recompute
            # a subset. For 50 vars and moderate clauses, this is manageable.

            # Re-evaluate all clauses that contain best_var
            # First, remove from unsat those that are now satisfied
            new_unsat = set()
            for uc in unsat_clauses:
                sat = False
                for v, p in clause_data[uc]:
                    if assignment[v] == p:
                        sat = True
                        break
                if not sat:
                    new_unsat.add(uc)
            # Check clauses NOT in unsat that contain best_var (they may have become unsat)
            for ci in range(m):
                if ci in new_unsat:
                    continue
                contains_var = False
                for v, p in clause_data[ci]:
                    if v == best_var:
                        contains_var = True
                        break
                if contains_var:
                    sat = False
                    for v, p in clause_data[ci]:
                        if assignment[v] == p:
                            sat = True
                            break
                    if not sat:
                        new_unsat.add(ci)
            unsat_clauses = new_unsat

        # Final count
        final_unsat = 0
        for ci in range(m):
            if not is_satisfied(assignment, ci):
                final_unsat += 1
        return assignment, final_unsat

    rng = random.Random(0)
    
    # Try multiple restarts with increasing budgets
    best_assignment = None
    best_unsat = float('inf')
    
    restart_count = 0
    base_steps = 500
    max_restarts = 50
    
    while time.time() < deadline and restart_count < max_restarts:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        max_steps = base_steps + restart_count * 200
        assignment, unsat = walk_sat_single(assignment, max_steps, rng)
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        if unsat == 0:
            return assignment
        
        restart_count += 1
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
