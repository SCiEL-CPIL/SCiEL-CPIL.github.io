import random
import time
import sys

# Module-level state for time management
_start_time = time.time()
# Assume 330s total limit, leave 15s buffer for safety/overhead
_DEADLINE = _start_time + 315.0

def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # 1. Pre-processing
    # Map variable indices (0-based) to list of (clause_idx, position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    clause_literals = [] # List of lists of (var_idx, sign) where sign is 1 for pos, -1 for neg
    
    for c_idx, clause in enumerate(clauses):
        lits = []
        for pos, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            lits.append((var_idx, sign))
            var_to_clauses[var_idx].append((c_idx, pos))
        clause_literals.append(lits)
        
    # Pre-compute for fast access: for each clause, the variables and signs
    # clause_vars[c_idx] = [var0, var1, var2]
    # clause_signs[c_idx] = [sign0, sign1, sign2]
    clause_vars = [[l[0] for l in lits] for lits in clause_literals]
    clause_signs = [[l[1] for l in lits] for lits in clause_literals]

    def compute_state(assignment):
        """
        Returns:
        sat_counts: list of int, number of true literals in each clause
        unsat_indices: list of indices of unsatisfied clauses
        unsat_count: int
        """
        sat_counts = [0] * m
        unsat_indices = []
        for i in range(m):
            count = 0
            vars_i = clause_vars[i]
            signs_i = clause_signs[i]
            for j in range(3):
                v = vars_i[j]
                s = signs_i[j]
                if assignment[v] == (s == 1):
                    count += 1
            sat_counts[i] = count
            if count == 0:
                unsat_indices.append(i)
        return sat_counts, unsat_indices, len(unsat_indices)

    # 2. Initial Assignment
    # Heuristic: Count literal occurrences. Assign to satisfy more occurrences.
    lit_counts = [0] * n
    for clause in clauses:
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                lit_counts[v] += 1
            else:
                lit_counts[v] -= 1
                
    # Initialize assignment based on counts, with some noise
    assignment = [False] * n
    for i in range(n):
        if lit_counts[i] > 0:
            assignment[i] = True
        elif lit_counts[i] < 0:
            assignment[i] = False
        else:
            assignment[i] = random.random() < 0.5
            
    # Add noise to avoid getting stuck in a bad local optimum immediately
    for i in range(n):
        if random.random() < 0.1:
            assignment[i] = not assignment[i]

    sat_counts, unsat_indices, unsat_count = compute_state(assignment)
    
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count
    
    if unsat_count == 0:
        return assignment

    # 3. WalkSAT Loop
    max_restarts = 1000
    base_steps = 3000
    p_random = 0.35 # Probability of random flip
    
    for restart in range(max_restarts):
        # Time check
        if time.time() > _DEADLINE - 3:
            return best_assignment
            
        if restart > 0:
            # Restart strategy
            if restart % 5 == 0:
                # Full random restart
                assignment = [random.random() < 0.5 for _ in range(n)]
            else:
                # Perturb best solution
                assignment = best_assignment.copy()
                num_flips = max(1, n // 20)
                for _ in range(num_flips):
                    v = random.randrange(n)
                    assignment[v] = not assignment[v]
            
            sat_counts, unsat_indices, unsat_count = compute_state(assignment)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
        
        # Adaptive steps: increase steps if we are making progress? 
        # Or just fixed steps per restart.
        steps = base_steps
        
        # Local best tracking for this restart
        local_best_unsat = unsat_count
        stuck_count = 0
        
        for step in range(steps):
            if unsat_count == 0:
                return assignment
            
            # Time check every 100 steps to balance overhead
            if step % 100 == 0 and time.time() > _DEADLINE - 3:
                return best_assignment

            # Pick random unsatisfied clause
            if not unsat_indices:
                return assignment
            c_idx = random.choice(unsat_indices)
            
            # Get literals in this clause
            v0, s0 = clause_literals[c_idx][0]
            v1, s1 = clause_literals[c_idx][1]
            v2, s2 = clause_literals[c_idx][2]
            
            # Calculate delta for flipping each variable
            # Delta = (newly unsatisfied) - (newly satisfied)
            # Flipping v in clause c (which is unsat) will satisfy c (delta -1)
            # But may unsatisfy other clauses containing v.
            
            def calc_delta(v):
                delta = -1 # Satisfies the chosen unsat clause
                for (ci, pos) in var_to_clauses[v]:
                    if ci == c_idx:
                        continue
                    # Check if clause ci is currently satisfied
                    if sat_counts[ci] > 0:
                        # If we flip v, does this clause become unsatisfied?
                        # It becomes unsatisfied if the literal for v in ci was the ONLY true literal.
                        # And if the literal for v in ci is currently TRUE.
                        # Literal is true if assignment[v] == (sign == 1)
                        var_sign = clause_signs[ci][pos]
                        lit_was_true = (assignment[v] == (var_sign == 1))
                        if lit_was_true and sat_counts[ci] == 1:
                            delta += 1
                return delta
            
            d0 = calc_delta(v0)
            d1 = calc_delta(v1)
            d2 = calc_delta(v2)
            
            # Choose flip
            if random.random() < p_random:
                # Random flip
                choice = random.choice([0, 1, 2])
                if choice == 0: v_flip = v0
                elif choice == 1: v_flip = v1
                else: v_flip = v2
            else:
                # Best flip
                min_d = min(d0, d1, d2)
                # Tie break randomly
                candidates = []
                if d0 == min_d: candidates.append(v0)
                if d1 == min_d: candidates.append(v1)
                if d2 == min_d: candidates.append(v2)
                v_flip = random.choice(candidates)
                
            # Perform Flip
            assignment[v_flip] = not assignment[v_flip]
            
            # Update State
            # We need to update sat_counts for all clauses containing v_flip
            for (ci, pos) in var_to_clauses[v_flip]:
                var_sign = clause_signs[ci][pos]
                lit_is_true = (assignment[v_flip] == (var_sign == 1))
                if lit_is_true:
                    # Literal became true
                    if sat_counts[ci] == 0:
                        # Clause was unsat, now sat
                        sat_counts[ci] = 1
                        unsat_count -= 1
                        # Remove from unsat_indices (lazy)
                        # We don't remove immediately, we just mark it as stale
                        # We will rebuild unsat_indices periodically
                    else:
                        sat_counts[ci] += 1
                else:
                    # Literal became false
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        # Clause became unsat
                        unsat_count += 1
                        unsat_indices.append(ci)
                        
            # Clean up unsat_indices periodically to prevent it from growing too large with stale entries
            # Stale entries are indices that are actually satisfied
            if len(unsat_indices) > unsat_count * 2 or step % 500 == 499:
                unsat_indices = [i for i in range(m) if sat_counts[i] == 0]
                
            # Track progress
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                stuck_count = 0
            else:
                stuck_count += 1
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
                
            # Adaptive p_random: increase randomness if stuck
            if stuck_count > 200:
                p_random = min(0.6, 0.35 + 0.01 * (stuck_count // 100))
            else:
                p_random = 0.35

    return best_assignment
