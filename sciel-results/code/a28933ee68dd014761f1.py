"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess clauses into numpy arrays for vectorization
    # Convert 1-based signed literals to 0-based indices and signs
    # indices: shape (m, 3), values in [0, n-1]
    # signs: shape (m, 3), values in {+1, -1}
    
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
            
    # Variable occurrence lists for fast gain calculation
    # var_positions[var] = list of (clause_idx, position_in_clause)
    var_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_positions[v].append((i, j))

    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def compute_initial_state(assignment):
        """Compute lit_sat and sat_counts from assignment."""
        val = np.take(assignment, indices)
        lit_sat = (val == (signs > 0))
        sat_counts = np.sum(lit_sat, axis=1)
        return lit_sat, sat_counts

    def flip_variable(var, assignment, lit_sat, sat_counts, unsat_clauses_set, unsat_clauses_list):
        """
        Flip variable var, update assignment, lit_sat, and sat_counts incrementally.
        Also updates unsat_clauses_set and unsat_clauses_list.
        """
        positions = var_positions[var]
        if not positions:
            return assignment, lit_sat, sat_counts, unsat_clauses_set, unsat_clauses_list
        
        # Update assignment
        assignment[var] = not assignment[var]
        
        # Update lit_sat and sat_counts for affected clauses
        for ci, pos in positions:
            old_sat = lit_sat[ci, pos]
            new_sat = not old_sat
            lit_sat[ci, pos] = new_sat
            
            old_count = sat_counts[ci]
            new_count = old_count + (1 if new_sat else -1)
            sat_counts[ci] = new_count
            
            # Update unsat tracking
            if old_count == 0 and new_count > 0:
                # Clause was unsat, now sat
                unsat_clauses_set.discard(ci)
            elif old_count > 0 and new_count == 0:
                # Clause was sat, now unsat
                unsat_clauses_set.add(ci)
                unsat_clauses_list.append(ci)
        
        return assignment, lit_sat, sat_counts, unsat_clauses_set, unsat_clauses_list

    def calculate_gain(var, lit_sat, sat_counts):
        """Calculate the gain if we flip var."""
        positions = var_positions[var]
        if not positions:
            return 0
        
        gain = 0
        for ci, pos in positions:
            if lit_sat[ci, pos]:
                # This literal is currently satisfied; flipping makes it unsatisfied
                if sat_counts[ci] == 1:
                    gain -= 1
            else:
                # This literal is currently unsatisfied; flipping makes it satisfied
                if sat_counts[ci] == 0:
                    gain += 1
        
        return gain

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    noise = 0.5 # Start with higher noise
    base_flip_budget = 2000 # Lower budget, more restarts
    restart_count = 0
    
    while time.time() < deadline:
        # Restart logic
        if best_assignment is None or random.random() < 0.5:
            assignment = initialize_assignment()
        else:
            # Perturb best assignment
            assignment = best_assignment.copy()
            num_flips = random.randint(1, max(1, n // 5))
            flip_vars = random.sample(range(n), min(num_flips, n))
            for v in flip_vars:
                assignment[v] = not assignment[v]
        
        lit_sat, sat_counts = compute_initial_state(assignment)
        
        unsat_clauses_list = np.where(sat_counts == 0)[0].tolist()
        unsat_clauses_set = set(unsat_clauses_list)
        current_unsat = len(unsat_clauses_list)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget
        flip_budget = base_flip_budget
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        local_best_assignment = assignment.copy()
        
        while steps < flip_budget and time.time() < deadline:
            if current_unsat == 0:
                return assignment.tolist()
            
            # Clean unsat_clauses_list if it has grown too large with stale entries
            # Heuristic: if len(list) > 2 * len(set), rebuild
            if len(unsat_clauses_list) > 2 * len(unsat_clauses_set) + 10:
                unsat_clauses_list = list(unsat_clauses_set)
            
            if not unsat_clauses_list:
                current_unsat = 0
                break
            
            # Pick a random unsatisfied clause from the list
            # We might pick a stale one, so we check
            while True:
                clause_idx = unsat_clauses_list[random.randint(0, len(unsat_clauses_list) - 1)]
                if clause_idx in unsat_clauses_set:
                    break
            
            # Pick a literal to flip
            # WalkSAT heuristic: flip with probability noise randomly, otherwise pick best gain
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # Find var in clause that maximizes gain
                best_gain = -100
                best_var = -1
                
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    gain = calculate_gain(var, lit_sat, sat_counts)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            assignment, lit_sat, sat_counts, unsat_clauses_set, unsat_clauses_list = flip_variable(
                var, assignment, lit_sat, sat_counts, unsat_clauses_set, unsat_clauses_list
            )
            
            # Update current_unsat
            current_unsat = len(unsat_clauses_set)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                local_best_assignment = assignment.copy()
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 100 == 0:
                if no_improvement_steps > 500:
                    noise = min(0.9, noise + 0.1)
                else:
                    noise = max(0.1, noise - 0.05)
        
        restart_count += 1
        # Decay noise back to baseline between restarts
        noise = max(0.3, noise * 0.9)

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
