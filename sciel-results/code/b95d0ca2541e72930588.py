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

    def walksat(assignment, max_steps, rng):
        """Run efficient WalkSAT with incremental clause updates. Returns (assignment, solved)."""
        # Initial state of unsatisfied clauses
        # Use a set for O(1) lookup and removal
        unsat_clauses = set(get_unsat_clauses(assignment).tolist())
        
        # Pre-compute a boolean mask for each variable indicating which clauses it appears in?
        # No, we have var_clause_arr which is list of clause indices.
        # We need to quickly check if a clause is currently unsatisfied.
        # The `unsat_clauses` set serves this purpose.
        
        p_flip = 0.5  # Probability of random flip in WalkSAT
        
        for step in range(max_steps):
            if not unsat_clauses:
                return assignment, True
            
            # Time check
            if (step % 100 == 0) and (time.time() > deadline):
                return assignment, False
            
            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            
            # Select variable to flip
            # Standard WalkSAT: with probability p, pick random var from clause.
            # Otherwise, pick var that minimizes number of newly unsatisfied clauses.
            
            if rng.random() < p_flip:
                j = rng.randint(0, 2)
                v = clause_var_matrix[ci, j]
            else:
                best_v = -1
                best_cost = float('inf')
                
                for j in range(3):
                    v = clause_var_matrix[ci, j]
                    # Calculate cost: number of clauses that will become unsatisfied
                    # if we flip v.
                    # Clauses affected are those in var_clause_arr[v].
                    # A clause c becomes unsatisfied if:
                    # 1. It is currently satisfied.
                    # 2. The literal for v in c is the ONLY satisfied literal in c.
                    
                    cost = 0
                    for c_idx in var_clause_arr[v]:
                        # If c_idx is currently unsatisfied, flipping v might satisfy it (good), 
                        # but won't make it "more" unsatisfied. 
                        # Actually, if it's unsatisfied, flipping v to satisfy its literal 
                        # will satisfy the whole clause (since other literals are already false).
                        # So unsat clauses don't contribute to "newly unsatisfied" count.
                        if c_idx in unsat_clauses:
                            continue
                        
                        # c_idx is currently satisfied.
                        # We need to check if flipping v makes it unsatisfied.
                        # This happens if v is the only true literal in c_idx.
                        # Check other 2 literals in c_idx.
                        
                        # Find position of v in c_idx
                        # We can just check the 3 literals.
                        sat_count = 0
                        v_is_true = False
                        for k in range(3):
                            if clause_var_matrix[c_idx, k] == v:
                                # This is the literal for v
                                # Current value of v
                                val = assignment[v]
                                sign = clause_sign_matrix[c_idx, k]
                                lit_val = 1 if val else -1
                                if lit_val * sign == 1:
                                    v_is_true = True
                                    sat_count += 1
                            else:
                                var_idx = clause_var_matrix[c_idx, k]
                                val = assignment[var_idx]
                                sign = clause_sign_matrix[c_idx, k]
                                lit_val = 1 if val else -1
                                if lit_val * sign == 1:
                                    sat_count += 1
                                    
                        # If v is true in this clause, and it's the only true literal (sat_count == 1),
                        # then flipping v will make this clause unsatisfied.
                        if v_is_true and sat_count == 1:
                            cost += 1
                            
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                
                v = best_v
            
            # Flip variable v
            assignment[v] = not assignment[v]
            
            # Update unsat_clauses set
            # Only clauses containing v can change status.
            for c_idx in var_clause_arr[v]:
                # Check if c_idx is satisfied now
                is_sat = False
                for k in range(3):
                    var_idx = clause_var_matrix[c_idx, k]
                    val = assignment[var_idx]
                    sign = clause_sign_matrix[c_idx, k]
                    lit_val = 1 if val else -1
                    if lit_val * sign == 1:
                        is_sat = True
                        break
                
                if is_sat:
                    unsat_clauses.discard(c_idx)
                else:
                    unsat_clauses.add(c_idx)
            
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
        # Increased steps allowed due to efficiency, but bounded by time
        max_steps = 100000
        
        assignment, solved = walksat(assignment, max_steps, rng)
        
        if solved:
            return assignment.tolist()
        
        # Track best assignment found
        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        # Last resort: return random assignment
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
