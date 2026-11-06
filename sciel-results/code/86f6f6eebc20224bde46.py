import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    # 0-based variable indices, signs: +1 for True, -1 for False
    # We will represent the matrix such that clause is satisfied if sum(A @ x) > 0?
    # Actually, let's stick to the "satisfied count" logic which is more robust for WalkSAT.
    # A[i, j] = 1 if var j is in clause i with positive sign
    # A[i, j] = -1 if var j is in clause i with negative sign
    # A[i, j] = 0 otherwise
    
    A = np.zeros((m, n), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            A[i, v] = sign
    
    # Helper function to calculate unsatisfied count and mask
    # x is a boolean array of shape (n,)
    # A is int8 (m, n)
    # For a clause i, satisfied if sum(A[i] * x_mapped) > 0 is not quite right because of 0s.
    # Let's compute the value of each clause.
    # If we map True -> 1, False -> -1.
    # Clause value = sum_{v in clause} sign(v) * x(v)
    # If value > 0, satisfied. If value == 0 (impossible for odd clauses? 3-SAT so sum is odd), or value < 0.
    # Wait, for 3-SAT, the sum of 3 terms each being +/-1 is in {-3, -1, 1, 3}.
    # So satisfied if sum > 0. Unsatisfied if sum < 0.
    
    def get_unsat_mask(x_bool):
        # x_bool: (n,) bool
        # Map to +/-1
        x_mapped = np.where(x_bool, 1, -1).astype(np.int8)
        # Clause values: (m,)
        vals = A @ x_mapped
        # Unsatisfied if vals < 0
        return vals < 0, vals

    rng = np.random.default_rng(42)
    
    best_unsat = m
    best_assignment = None
    
    # Strategy: Multiple restarts of local search
    # Each restart: random init, then flip variables to minimize unsat count.
    
    max_restarts = 5000
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        x = rng.random(n) < 0.5
        
        unsat_mask, _ = get_unsat_mask(x)
        cur_unsat = np.sum(unsat_mask)
        
        if cur_unsat == 0:
            return x.tolist()
            
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = x.copy()
            if best_unsat == 0:
                return best_assignment.tolist()
        
        # Local search loop
        stuck_count = 0
        last_unsat = cur_unsat
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if cur_unsat == 0:
                return x.tolist()
                
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = x.copy()
                if best_unsat == 0:
                    return best_assignment.tolist()
            
            # If stuck (no improvement for a while), break out of this restart
            if cur_unsat > last_unsat:
                stuck_count += 1
            else:
                stuck_count = 0
                last_unsat = cur_unsat
                
            if stuck_count > 50:
                break
                
            # Pick a random unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return x.tolist()
                
            c_idx = unsat_indices[rng.integers(len(unsat_indices))]
            
            # Get variables in this clause
            # Find non-zero entries in A[c_idx]
            var_indices = np.nonzero(A[c_idx])[0]
            
            # Calculate gain for flipping each variable in the clause
            # Flip variable v:
            # 1. Update x
            # 2. Check all clauses containing v
            # We need to know which clauses contain each variable.
            # Precompute var_to_clauses mapping for speed?
            # Since n=150, m~450 (typical 3x ratio), we can just loop over clauses in Python if needed, 
            # but vectorized is better.
            
            # However, calculating exact gain for each of 3 variables in the clause requires checking
            # impact on all m clauses.
            # Impact of flipping v on clause i:
            # If A[i,v] == 0, no change.
            # If A[i,v] != 0:
            #   Old contribution: A[i,v] * x[v]
            #   New contribution: A[i,v] * (-x[v])
            #   Change in clause value: -2 * A[i,v] * x[v]
            
            # Let's evaluate the 3 candidates in the selected clause.
            best_gain = -100
            best_var = -1
            gains = []
            
            for v in var_indices:
                # Calculate delta for all clauses
                # delta_i = -2 * A[i, v] * x[v]
                # If x[v] is True (1), delta = -2 * A[i, v]
                # If x[v] is False (-1), delta = 2 * A[i, v]
                
                # Current clause values: vals
                # New vals: vals + delta
                # New unsat: (vals + delta) < 0
                # Old unsat: vals < 0
                
                # Gain = old_unsat_count - new_unsat_count
                # This is complex to vectorize fully for just 3 vars without overhead.
                # Let's just simulate the flip and count? No, that's O(m) per var, O(3m) per step.
                # With vectorization, we can do it for all 3 vars in parallel?
                
                pass
            
            # Alternative: Precompute clause-variable incidence?
            # Let's try a simpler heuristic often used in fast WalkSAT:
            # Pick a random variable from the unsat clause.
            # If greedy (prob p), pick the one with max gain.
            
            # To get gain efficiently:
            # We have `vals` (m,).
            # We have `x` (n,).
            # For variable v, the change in clause values is `delta = -2 * A[:, v] * x[v]` (where x[v] is +/-1)
            # `new_vals = vals + delta`
            # `old_unsat = (vals < 0).sum()`
            # `new_unsat = (new_vals < 0).sum()`
            # `gain = old_unsat - new_unsat`
            
            x_mapped = np.where(x, 1, -1).astype(np.int8)
            
            gains = []
            for v in var_indices:
                delta = -2 * A[:, v] * x_mapped[v]
                new_vals = vals + delta
                new_unsat = np.sum(new_vals < 0)
                gain = cur_unsat - new_unsat
                gains.append(gain)
            
            # Adaptive greedy probability
            # If we are close to solution, be more greedy.
            p_greedy = 0.8 - 0.5 * (cur_unsat / m)
            if p_greedy < 0.2: p_greedy = 0.2
            if p_greedy > 0.9: p_greedy = 0.9
            
            if rng.random() < p_greedy:
                # Pick best gain
                max_gain = max(gains)
                # Randomly pick among those with max gain
                best_vars = [var_indices[i] for i in range(len(gains)) if gains[i] == max_gain]
                v_to_flip = best_vars[rng.integers(len(best_vars))]
            else:
                # Random pick from clause
                v_to_flip = var_indices[rng.integers(len(var_indices))]
            
            # Flip variable
            x[v_to_flip] = not x[v_to_flip]
            
            # Update vals efficiently
            # delta = -2 * A[:, v_to_flip] * x_mapped_old[v_to_flip]
            # Since we flipped, new x_mapped is -old.
            # Actually, we can just recalculate vals for the flipped variable's impact.
            # Or simpler: recalculate entire vals?
            # Recalculating vals is A @ x_mapped.
            # A is (m, n), x_mapped is (n,). This is a matrix-vector product.
            # With m~450, n=150, this is very fast.
            
            x_mapped = np.where(x, 1, -1).astype(np.int8)
            vals = A @ x_mapped
            unsat_mask = vals < 0
            cur_unsat = np.sum(unsat_mask)
            
            # If this step was very slow, we might timeout, but matrix mult is fast.
            
    if best_assignment is not None:
        # Final check before returning
        x_mapped = np.where(best_assignment, 1, -1).astype(np.int8)
        vals = A @ x_mapped
        if np.sum(vals < 0) == 0:
            return best_assignment.tolist()
        return best_assignment.tolist()
        
    return [True] * n
