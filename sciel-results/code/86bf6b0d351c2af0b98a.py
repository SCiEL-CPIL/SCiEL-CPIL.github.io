"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = 315.0  # safety margin before 330s kill
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    var_clauses_arr = [np.array(vc, dtype=np.int32) for vc in var_clauses]
    
    # For each variable, precompute the position of that variable in each clause it appears in
    # var_pos[v] = list of (clause_idx, pos_in_clause) pairs
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_pos[v].append((i, j))
    var_pos_arr = [np.array(vp, dtype=np.int32) for vp in var_pos]  # shape (k, 2)
    
    def full_evaluate(assignment):
        """Full vectorized evaluation. Returns (unsat_count, unsat_indices, num_true_per_clause)."""
        assign_int = (2 * assignment.astype(np.int8) - 1)
        lit_vals = clause_data * assign_int[clause_vars]
        clause_satisfied = np.max(lit_vals, axis=1) == 1
        unsat_count = int(np.sum(~clause_satisfied))
        unsat_indices = np.where(~clause_satisfied)[0]
        # Count true literals per clause for incremental updates
        num_true = np.sum(lit_vals == 1, axis=1)
        return unsat_count, unsat_indices, num_true
    
    def compute_flip_delta(v, assignment, num_true):
        """Compute the change in unsat count if we flip variable v.
        Returns the delta (new_unsat - old_unsat). Negative means improvement."""
        vp = var_pos_arr[v]
        if len(vp) == 0:
            return 0
        
        clause_idxs = vp[:, 0]
        positions = vp[:, 1]
        
        # Current sign of variable v in each clause
        signs = clause_data[clause_idxs, positions]
        
        # Current value of v: True -> 1, False -> -1
        v_val = 1 if assignment[v] else -1
        
        # Current literal value for v in each clause: sign * v_val
        # If v_val == 1: literal is true if sign == 1
        # If v_val == -1: literal is true if sign == -1
        # After flip: v_val -> -v_val, so literal flips
        
        # Clauses where v's literal is currently TRUE (sign * v_val == 1)
        v_lit_true = (signs * v_val) == 1
        
        # Clauses where v's literal is currently FALSE
        v_lit_false = ~v_lit_true
        
        # For clauses where v's lit is TRUE:
        #   If num_true[clause] == 1, flipping v makes it unsat (delta +1)
        #   If num_true[clause] >= 2, flipping v keeps it sat (delta 0)
        
        # For clauses where v's lit is FALSE:
        #   If num_true[clause] == 0, flipping v makes it sat (delta -1)
        #   If num_true[clause] >= 1, flipping v keeps it unsat (delta 0)
        
        nt = num_true[clause_idxs]
        
        delta = 0
        # Clauses that become unsat: v_lit_true AND nt == 1
        delta += int(np.sum(v_lit_true & (nt == 1)))
        # Clauses that become sat: v_lit_false AND nt == 0
        delta -= int(np.sum(v_lit_false & (nt == 0)))
        
        return delta
    
    def apply_flip(v, assignment, num_true):
        """Apply flip to variable v and update num_true incrementally."""
        vp = var_pos_arr[v]
        if len(vp) == 0:
            return
        
        clause_idxs = vp[:, 0]
        positions = vp[:, 1]
        signs = clause_data[clause_idxs, positions]
        
        v_val = 1 if assignment[v] else -1
        
        # v's literal is true where sign * v_val == 1
        v_lit_true = (signs * v_val) == 1
        
        # Where v's lit was true, after flip it's false: num_true decreases by 1
        num_true[clause_idxs[v_lit_true]] -= 1
        # Where v's lit was false, after flip it's true: num_true increases by 1
        num_true[clause_idxs[~v_lit_true]] += 1
        
        assignment[v] = not assignment[v]
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    # Adaptive parameters
    base_max_steps = 3000
    restart_count = 0
    
    while time.time() - start_time < deadline:
        restart_count += 1
        
        # Initialize: first restart random, later ones biased toward best
        if best_assignment is not None and restart_count > 1 and rng.random() < 0.15:
            assignment = best_assignment.copy()
        else:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count, unsat_indices, num_true = full_evaluate(assignment)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT loop with incremental updates
        # Adaptive step limit: increase as we get closer to deadline
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            max_steps = 500
        elif remaining < 30.0:
            max_steps = 1000
        else:
            max_steps = base_max_steps + restart_count * 200
        
        step = 0
        while step < max_steps and time.time() - start_time < deadline:
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            
            # For each of the 3 variables in this clause, compute flip delta
            best_var = -1
            best_delta = float('inf')
            
            # Use a small random perturbation to break ties and add exploration
            deltas = []
            for j in range(3):
                v = int(clause_vars[ci, j])
                d = compute_flip_delta(v, assignment, num_true)
                # Add small random noise for exploration
                d += rng.random() * 0.5
                deltas.append((d, v))
            
            # Pick the variable with minimum delta
            deltas.sort()
            best_var = deltas[0][1]
            
            # With some probability, pick a random variable for exploration
            if rng.random() < 0.1:
                best_var = int(clause_vars[ci, rng.randrange(3)])
            
            # Apply the flip
            old_unsat = unsat_count
            apply_flip(best_var, assignment, num_true)
            
            # Update unsat count incrementally
            # We need to recompute which clauses are unsat
            # A clause is unsat iff num_true[clause] == 0
            unsat_count = int(np.sum(num_true == 0))
            
            # Update unsat_indices
            if unsat_count != old_unsat:
                unsat_indices = np.where(num_true == 0)[0]
            
            step += 1
        
        # Check if we solved it
        if unsat_count == 0:
            return assignment.tolist()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n
# EVOLVE-BLOCK-END
