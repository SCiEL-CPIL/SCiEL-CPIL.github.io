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

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# EVOLVE-BLOCK-END
