"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin

    # Precompute clause structure for fast evaluation
    # For each clause, store (var_indices, signs)
    # var_indices: list of 0-based variable indices
    # signs: +1 for positive literal, -1 for negative
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_)
        clause_signs.append(signs)

    # Convert to numpy arrays for vectorized operations
    # clause_var_matrix: (m, 3) array of variable indices
    # clause_sign_matrix: (m, 3) array of +1/-1
    clause_var_matrix = np.array(clause_vars, dtype=np.int32)
    clause_sign_matrix = np.array(clause_signs, dtype=np.int32)

    # For each variable, precompute which clauses it appears in and with what sign
    # var_clause_indices[v] = list of (clause_idx, sign_in_clause)
    var_clause_indices = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_var_matrix[ci, j]
            s = clause_sign_matrix[ci, j]
            var_clause_indices[v].append((ci, s))

    # Convert to arrays for faster access
    var_clause_arr = []
    var_sign_arr = []
    for v in range(n):
        if var_clause_indices[v]:
            cis = [x[0] for x in var_clause_indices[v]]
            ss = [x[1] for x in var_clause_indices[v]]
            var_clause_arr.append(np.array(cis, dtype=np.int32))
            var_sign_arr.append(np.array(ss, dtype=np.int32))
        else:
            var_clause_arr.append(np.array([], dtype=np.int32))
            var_sign_arr.append(np.array([], dtype=np.int32))

    def check_satisfied(assignment):
        """Check if all clauses are satisfied. Returns True if satisfied."""
        # For each clause, check if any literal is true
        # assignment is a boolean numpy array of length n
        # A literal is satisfied if assignment[var] == (sign == +1)
        # i.e., assignment[var] * sign == +1
        # clause is satisfied if max over its 3 literals of (assignment[var] * sign) == 1
        # where assignment is +1/True or -1/False
        
        # Convert assignment to +1/-1
        a = np.where(assignment, 1, -1)  # shape (n,)
        # For each clause, compute satisfaction
        # sat[ci, j] = a[clause_var_matrix[ci, j]] * clause_sign_matrix[ci, j]
        # clause satisfied if max(sat[ci, :]) == 1
        sat = a[clause_var_matrix] * clause_sign_matrix  # shape (m, 3)
        clause_sat = np.max(sat, axis=1) == 1  # shape (m,)
        return np.all(clause_sat)

    def count_unsat(assignment):
        """Return count of unsatisfied clauses."""
        a = np.where(assignment, 1, -1)
        sat = a[clause_var_matrix] * clause_sign_matrix
        clause_sat = np.max(sat, axis=1) == 1
        return int(np.sum(~clause_sat))

    def get_unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        a = np.where(assignment, 1, -1)
        sat = a[clause_var_matrix] * clause_sign_matrix
        clause_sat = np.max(sat, axis=1) == 1
        return np.where(~clause_sat)[0]

    def walksat(assignment, max_steps, rng, clause_weights=None):
        """Run WalkSAT with min-conflicts-like heuristic. Returns (assignment, solved)."""
        if clause_weights is None:
            clause_weights = np.ones(m, dtype=np.float64)
        
        current_unsat = get_unsat_clauses(assignment)
        
        for step in range(max_steps):
            if len(current_unsat) == 0:
                return assignment, True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(current_unsat.tolist())
            
            # For each literal in this clause, compute the number of clauses that would be affected
            # We want to pick the variable that minimizes the damage (min-conflicts heuristic)
            # For each variable in the clause, compute:
            #   - number of currently unsatisfied clauses that contain this variable (would be satisfied)
            #   - number of currently satisfied clauses that contain this variable (would be unsatisfied)
            
            best_var = -1
            best_score = float('inf')
            
            for j in range(3):
                v = clause_var_matrix[ci, j]
                # Get all clauses containing variable v
                cidx = var_clause_arr[v]
                if len(cidx) == 0:
                    # Variable not in any clause? Shouldn't happen but handle gracefully
                    score = 0
                else:
                    # Determine current state of these clauses
                    # A clause is currently unsatisfied if it's in current_unsat
                    # We need to check: if we flip v, which clauses change?
                    # A clause containing v changes state iff v is the only satisfied literal in that clause
                    # For currently unsatisfied clauses: flipping v to satisfy its literal will satisfy the clause
                    #   (since all literals are false, making one true satisfies it)
                    # For currently satisfied clauses: flipping v to false its literal will unsatisfy the clause
                    #   only if v was the only satisfied literal
                    
                    # Simpler approach: just count how many unsat clauses contain v
                    # and how many sat clauses contain v where v is critical
                    unsat_in_v = np.isin(cidx, current_unsat)
                    num_unsat_with_v = int(np.sum(unsat_in_v))
                    
                    # For satisfied clauses with v, check if v is critical
                    # This is more expensive. Let's use a simpler heuristic:
                    # score = num_satisfied_with_v - num_unsat_with_v * 2
                    # (prioritize fixing unsat clauses)
                    
                    # Even simpler: just use the number of unsatisfied clauses containing v
                    # as a proxy for "how helpful" flipping v would be
                    # and assume the damage is roughly proportional to total clauses with v
                    score = len(cidx) - 2 * num_unsat_with_v
                
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                # Fallback: random variable from the clause
                j = rng.randint(0, 2)
                best_var = clause_var_matrix[ci, j]
            
            # Flip the variable
            assignment[best_var] = not assignment[best_var]
            
            # Update unsatisfied clauses
            current_unsat = get_unsat_clauses(assignment)
            
            # Periodic time check
            if (step % 500 == 0) and (time.time() > deadline):
                return assignment, False
        
        return assignment, False

    # Main search loop with restarts
    best_assignment = None
    best_unsat_count = m + 1
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts * 12345 + 67890)
        
        # Initialize assignment randomly
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Check if already satisfied
        if check_satisfied(assignment):
            return assignment.tolist()
        
        # Run WalkSAT
        # Adaptive step budget: start with more steps, reduce if not converging
        max_steps = 15000
        if num_restarts > 10:
            max_steps = 8000
        if num_restarts > 30:
            max_steps = 5000
            
        assignment, solved = walksat(assignment, max_steps, rng)
        
        if solved:
            return assignment.tolist()
        
        # Track best assignment found
        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # If we've done many restarts without success, try a more aggressive approach
        if num_restarts >= 50 and best_assignment is not None:
            # Try to repair the best assignment found so far
            if time.time() < deadline:
                rng2 = random.Random(99999)
                assignment = best_assignment.copy()
                assignment, solved = walksat(assignment, 10000, rng2)
                if solved:
                    return assignment.tolist()
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        # Last resort: return random assignment
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
