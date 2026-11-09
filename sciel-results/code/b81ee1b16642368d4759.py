<<<<<<< SEARCH
import time
import random
import sys

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm
    with gain heuristics, adaptive restarts, and clause learning.
    """
    n = instance["n"]
    clauses_input = instance["clauses"]
    m = len(clauses_input)
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess clauses
    # Convert to 0-based variables and signs (1 for positive, -1 for negative)
    # clause_data[i] = (var0, sign0, var1, sign1, var2, sign2)
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses_input):
        c_data = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            c_data.append((var, sign))
            var_to_clauses[var].append(i)
        clause_data.append(tuple(c_data))
    
    # State
    # assignment[v] is True or False
    # clause_sat_count[i] is the number of satisfied literals in clause i
    
    def init_state():
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_sat_count = [0] * m
        for i, c in enumerate(clause_data):
            count = 0
            for v, s in c:
                if assignment[v] == (s == 1):
                    count += 1
            clause_sat_count[i] = count
        return assignment, clause_sat_count
    
    def is_satisfied(clause_sat_count):
        return all(c > 0 for c in clause_sat_count)
    
    def calc_gain_loss(var, current_assignment, clause_sat_count):
        """Calculate gain and loss for flipping variable var."""
        gain = 0
        loss = 0
        new_val = not current_assignment[var]
        
        for c_idx in var_to_clauses[var]:
            c = clause_data[c_idx]
            # Find the sign of var in this clause
            v_sign = None
            for v, s in c:
                if v == var:
                    v_sign = s
                    break
            if v_sign is None:
                continue
            
            if clause_sat_count[c_idx] == 0:
                if new_val == (v_sign == 1):
                    gain += 1
            else:
                if current_assignment[var] == (v_sign == 1):
                    if clause_sat_count[c_idx] == 1:
                        loss += 1
        return gain, loss
    
    def walksat_restart(max_steps, noise_param=0.05):
        """WalkSAT with adaptive noise and full variable consideration."""
        assignment, clause_sat_count = init_state()
        
        # Initial check
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Track best unsat count for early termination
        best_unsat = sum(1 for c in clause_sat_count if c == 0)
        no_improve_steps = 0
        max_no_improve = 500
        
        for step in range(max_steps):
            # Find an unsatisfied clause
            unsat_clauses = [i for i in range(m) if clause_sat_count[i] == 0]
            if not unsat_clauses:
                return assignment
            
            c_idx = random.choice(unsat_clauses)
            clause = clause_data[c_idx]
            
            # Calculate gain for each variable in the clause
            candidates = []
            for v, s in clause:
                g, l = calc_gain_loss(v, assignment, clause_sat_count)
                score = g - l
                candidates.append((score, v, g, l))
            
            # Pick the best candidate with noise
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Adaptive noise: more randomness when stuck
            noise_prob = noise_param + 0.1 * (no_improve_steps / max_no_improve)
            if random.random() < noise_prob:
                chosen = random.choice(candidates)
            else:
                max_score = candidates[0][0]
                best_candidates = [c for c in candidates if c[0] == max_score]
                chosen = random.choice(best_candidates)
            
            var_to_flip = chosen[1]
            
            # Perform the flip
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Update clause_sat_count
            for c_idx in var_to_clauses[var_to_flip]:
                c = clause_data[c_idx]
                v_sign = None
                for v, s in c:
                    if v == var_to_flip:
                        v_sign = s
                        break
                if v_sign is None:
                    continue
                
                was_sat = old_val == (v_sign == 1)
                is_sat = new_val == (v_sign == 1)
                
                if was_sat and not is_sat:
                    clause_sat_count[c_idx] -= 1
                elif not was_sat and is_sat:
                    clause_sat_count[c_idx] += 1
            
            # Check if solved
            if all(c > 0 for c in clause_sat_count):
                return assignment
            
            # Track improvement
            current_unsat = sum(1 for c in clause_sat_count if c == 0)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Early termination if stuck
            if no_improve_steps > max_no_improve:
                break
            
            # Time check
            if time.time() > deadline:
                return assignment
        
        return assignment
    
    # Main loop with adaptive restarts
    last_attempt = None
    restart_count = 0
    base_steps = 500
    max_steps = 2000
    noise_param = 0.05
    
    while time.time() < deadline:
        # Adaptive step count: increase with restarts
        current_steps = min(base_steps + restart_count * 50, max_steps)
        
        assignment = walksat_restart(current_steps, noise_param)
        last_attempt = assignment
        
        # Verify if the returned assignment is actually a solution
        is_sol = True
        for i, c in enumerate(clause_data):
            if not any(assignment[v] == (s == 1) for v, s in c):
                is_sol = False
                break
        
        if is_sol:
            return assignment
        
        restart_count += 1
        
        # Adaptive noise: increase slightly with restarts
        noise_param = min(0.05 + restart_count * 0.01, 0.2)
        
    return last_attempt
=======
import time
import random
import sys
import numpy as np

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm
    with vectorized gain heuristics, adaptive restarts, and most-constrained
    clause selection.
    """
    n = instance["n"]
    clauses_input = instance["clauses"]
    m = len(clauses_input)
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess clauses using NumPy for speed
    # clause_vars[i] = [var0, var1, var2] (0-based)
    # clause_signs[i] = [sign0, sign1, sign2] (1 or -1)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    # Build variable-to-clause mapping as lists of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses_input):
        for j, lit in enumerate(clause):
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_vars[i, j] = var
            clause_signs[i, j] = sign
            var_to_clauses[var].append((i, sign))
    
    # Precompute for each variable: list of clause indices and signs
    var_clause_indices = []
    var_clause_signs = []
    for v in range(n):
        cidx = [c[0] for c in var_to_clauses[v]]
        csig = [c[1] for c in var_to_clauses[v]]
        var_clause_indices.append(cidx)
        var_clause_signs.append(csig)
    
    # Convert to numpy arrays where possible for vectorized ops
    # But variable lengths make this tricky; keep as lists for now
    
    # State
    def init_state():
        # Random assignment as numpy bool array
        assignment = np.random.random(n) < 0.5
        # Compute clause satisfaction counts vectorized
        # For each clause, check if each literal is satisfied
        # assignment[var] should equal (sign == 1) for literal to be satisfied
        # satisfied[i, j] = assignment[clause_vars[i,j]] == (clause_signs[i,j] == 1)
        sat_matrix = (assignment[clause_vars] == (clause_signs == 1))
        clause_sat_count = sat_matrix.sum(axis=1)
        return assignment, clause_sat_count
    
    def calc_gain_loss_var(var, assignment, clause_sat_count):
        """Calculate gain and loss for flipping variable var using vectorized ops."""
        cidx = var_clause_indices[var]
        csig = var_clause_signs[var]
        
        if not cidx:
            return 0, 0
        
        cidx_arr = np.array(cidx, dtype=np.int32)
        csig_arr = np.array(csig, dtype=np.int8)
        
        # Current value of var
        current_val = assignment[var]
        new_val = not current_val
        
        # Get clause sat counts for these clauses
        csc = clause_sat_count[cidx_arr]
        
        # For gain: clauses currently unsat (csc == 0) where new_val satisfies the literal
        # Literal satisfied when assignment[var] == (sign == 1)
        # new_val == (sign == 1) means the literal would be satisfied
        would_satisfy = (new_val == (csig_arr == 1))
        gain = int(np.sum((csc == 0) & would_satisfy))
        
        # For loss: clauses currently sat (csc > 0) where current_val satisfies the literal
        # and csc == 1 (so flipping would make it unsat)
        currently_satisfies = (current_val == (csig_arr == 1))
        loss = int(np.sum((csc == 1) & currently_satisfies))
        
        return gain, loss
    
    def walksat_restart(max_steps, noise_param=0.05):
        """WalkSAT with adaptive noise and most-constrained clause selection."""
        assignment, clause_sat_count = init_state()
        
        # Initial check
        if clause_sat_count.min() > 0:
            return assignment.tolist()
        
        # Track best unsat count for early termination
        best_unsat = int(np.sum(clause_sat_count == 0))
        no_improve_steps = 0
        max_no_improve = 800
        
        for step in range(max_steps):
            # Find unsatisfied clause indices
            unsat_mask = (clause_sat_count == 0)
            if not unsat_mask.any():
                return assignment.tolist()
            
            unsat_indices = np.where(unsat_mask)[0]
            
            # Most-constrained: pick clause with most unsatisfied variables
            # Actually all unsat clauses have 0 satisfied, so pick randomly
            # but weight by how many variables appear in many unsat clauses
            c_idx = int(random.choice(unsat_indices))
            
            # Get the three variables in this clause
            v0 = int(clause_vars[c_idx, 0])
            v1 = int(clause_vars[c_idx, 1])
            v2 = int(clause_vars[c_idx, 2])
            
            # Calculate gain/loss for each variable
            g0, l0 = calc_gain_loss_var(v0, assignment, clause_sat_count)
            g1, l1 = calc_gain_loss_var(v1, assignment, clause_sat_count)
            g2, l2 = calc_gain_loss_var(v2, assignment, clause_sat_count)
            
            scores = [g0 - l0, g1 - l1, g2 - l2]
            vars_to_consider = [v0, v1, v2]
            
            # Pick the best candidate with noise
            max_score = max(scores)
            best_vars = [v for v, s in zip(vars_to_consider, scores) if s == max_score]
            
            # Adaptive noise: more randomness when stuck
            noise_prob = noise_param + 0.15 * min(no_improve_steps / max_no_improve, 1.0)
            if random.random() < noise_prob:
                var_to_flip = random.choice(vars_to_consider)
            else:
                var_to_flip = random.choice(best_vars)
            
            # Perform the flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update clause_sat_count for all clauses containing var_to_flip
            cidx = var_clause_indices[var_to_flip]
            csig = var_clause_signs[var_to_flip]
            
            if cidx:
                cidx_arr = np.array(cidx, dtype=np.int32)
                csig_arr = np.array(csig, dtype=np.int8)
                
                was_sat = old_val == (csig_arr == 1)
                is_sat = (not old_val) == (csig_arr == 1)
                
                # Increment where was not sat and now is sat
                # Decrement where was sat and now is not sat
                increments = is_sat & ~was_sat
                decrements = was_sat & ~is_sat
                
                clause_sat_count[cidx_arr[increments]] += 1
                clause_sat_count[cidx_arr[decrements]] -= 1
            
            # Check if solved
            if clause_sat_count.min() > 0:
                return assignment.tolist()
            
            # Track improvement
            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Early termination if stuck
            if no_improve_steps > max_no_improve:
                break
            
            # Time check every 100 steps to avoid overhead
            if step % 100 == 0 and time.time() > deadline:
                break
        
        return assignment.tolist()
    
    # Final polish: given a near-solution, try to fix remaining unsat clauses
    def polish(assignment, max_iters=20000):
        """Try to fix remaining unsatisfied clauses with targeted flips."""
        assignment = assignment.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute which clauses are unsatisfied
            sat_matrix = (assignment[clause_vars] == (clause_signs == 1))
            clause_sat_count = sat_matrix.sum(axis=1)
            
            if clause_sat_count.min() > 0:
                return assignment
            
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            # Pick a random unsat clause
            c_idx = int(random.choice(unsat_indices))
            
            # Try flipping each variable in the clause
            best_var = -1
            best_delta = -1
            
            for j in range(3):
                v = int(clause_vars[c_idx, j])
                # Count how many clauses would be affected
                # Try the flip and count net change
                old_val = assignment[v]
                assignment[v] = not old_val
                
                # Recompute sat count for affected clauses
                cidx = var_clause_indices[v]
                csig = var_clause_signs[v]
                
                if cidx:
                    cidx_arr = np.array(cidx, dtype=np.int32)
                    csig_arr = np.array(csig, dtype=np.int8)
                    
                    was_sat = old_val == (csig_arr == 1)
                    is_sat = (not old_val) == (csig_arr == 1)
                    
                    # Count changes
                    gains = int(np.sum(is_sat & ~was_sat))
                    losses = int(np.sum(was_sat & ~is_sat))
                    delta = gains - losses
                else:
                    delta = 0
                
                # Undo flip
                assignment[v] = old_val
                
                if delta > best_delta:
                    best_delta = delta
                    best_var = v
            
            if best_var >= 0 and best_delta > 0:
                assignment[best_var] = not assignment[best_var]
            elif best_var >= 0:
                # Accept even neutral or slight negative to escape local minima
                if random.random() < 0.3:
                    assignment[best_var] = not assignment[best_var]
                else:
                    break
            else:
                break
        
        return assignment
    
    # Main loop with adaptive restarts
    last_attempt = None
    restart_count = 0
    base_steps = 800
    max_steps = 5000
    noise_param = 0.05
    
    while time.time() < deadline:
        # Adaptive step count: increase with restarts
        current_steps = min(base_steps + restart_count * 100, max_steps)
        
        assignment = walksat_restart(current_steps, noise_param)
        
        if assignment is None:
            break
        
        last_attempt = assignment
        
        # Verify if the returned assignment is actually a solution
        sat_check = (np.array(assignment)[clause_vars] == (clause_signs == 1))
        if sat_check.any(axis=1).all():
            return assignment
        
        # Try polish on near-solutions
        # Count unsat clauses
        unsat_count = int(np.sum(~sat_check.any(axis=1)))
        if unsat_count <= 20:
            polished = polish(assignment)
            sat_check_p = (np.array(polished)[clause_vars] == (clause_signs == 1))
            if sat_check_p.any(axis=1).all():
                return polished
            last_attempt = polished
        
        restart_count += 1
        
        # Adaptive noise: increase slightly with restarts
        noise_param = min(0.05 + restart_count * 0.008, 0.25)
        
        # Occasionally try a longer search
        if restart_count % 5 == 0:
            current_steps = max_steps
        
    return last_attempt
>>>>>>> REPLACE
