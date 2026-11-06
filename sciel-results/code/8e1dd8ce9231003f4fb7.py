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

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Preprocess clauses into numpy arrays for vectorization
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
    
    # Build variable-to-clause mapping for fast gain calculation
    var_clause_idx = []
    var_pos_in_clause = []
    for v in range(n):
        ci_list = []
        pos_list = []
        for i in range(m):
            for j in range(3):
                if indices[i, j] == v:
                    ci_list.append(i)
                    pos_list.append(j)
        var_clause_idx.append(np.array(ci_list, dtype=np.int32))
        var_pos_in_clause.append(np.array(pos_list, dtype=np.int32))
    
    # Precompute for each variable: which clauses it appears in and the sign
    # var_clause_signs[var] = array of signs for each occurrence
    var_clause_signs = []
    for v in range(n):
        ci_arr = var_clause_idx[v]
        pos_arr = var_pos_in_clause[v]
        if len(ci_arr) > 0:
            var_clause_signs.append(signs[ci_arr, pos_arr])
        else:
            var_clause_signs.append(np.empty(0, dtype=np.int8))
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def calculate_sat_counts(assignment):
        val = np.take(assignment, indices)
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    def flip_variable_fast(var, assignment, sat_counts, unsat_mask):
        """
        Flip variable var, update assignment, sat_counts, and unsat_mask incrementally.
        Uses vectorized operations for speed.
        """
        ci_arr = var_clause_idx[var]
        pos_arr = var_pos_in_clause[var]
        sign_arr = var_clause_signs[var]
        
        if len(ci_arr) == 0:
            return sat_counts, unsat_mask
        
        # Update assignment
        assignment[var] = not assignment[var]
        
        # Vectorized update of sat_counts for affected clauses
        # For each affected clause, recompute its sat count
        old_counts = sat_counts[ci_arr]
        
        # Get current assignment values for all literals in these clauses
        clause_indices = indices[ci_arr]  # shape (k, 3)
        clause_signs = signs[ci_arr]      # shape (k, 3)
        vals = np.take(assignment, clause_indices)  # shape (k, 3)
        desired = clause_signs > 0         # shape (k, 3)
        is_sat = (vals == desired)         # shape (k, 3)
        new_counts = np.sum(is_sat, axis=1)  # shape (k,)
        
        # Update sat_counts
        sat_counts[ci_arr] = new_counts
        
        # Update unsat_mask vectorized
        was_unsat = old_counts == 0
        now_unsat = new_counts == 0
        
        # Clauses that were unsat but now sat: remove from unsat
        became_sat = was_unsat & ~now_unsat
        if np.any(became_sat):
            unsat_mask[ci_arr[became_sat]] = False
        
        # Clauses that were sat but now unsat: add to unsat
        became_unsat = ~was_unsat & now_unsat
        if np.any(became_unsat):
            unsat_mask[ci_arr[became_unsat]] = True
        
        return sat_counts, unsat_mask

    def calculate_gains_vectorized(clause_idx, assignment, sat_counts):
        """
        Calculate the gain for each of the 3 variables in clause_idx.
        Returns an array of 3 gains.
        """
        gains = np.zeros(3, dtype=np.int32)
        clause_vars = indices[clause_idx]
        clause_signs = signs[clause_idx]
        
        for pos in range(3):
            var = clause_vars[pos]
            ci_arr = var_clause_idx[var]
            pos_arr = var_pos_in_clause[var]
            sign_arr = var_clause_signs[var]
            
            if len(ci_arr) == 0:
                gains[pos] = 0
                continue
            
            current_val = assignment[var]
            new_val = not current_val
            
            # Vectorized gain calculation
            # For each clause containing var:
            # - If currently sat and would become unsat: if count==1, gain -= 1
            # - If currently unsat and would become sat: if count==0, gain += 1
            
            # Determine for each occurrence whether flipping changes satisfaction
            # lit_desired: whether the literal is satisfied when var is True
            lit_desired = sign_arr > 0  # shape (k,)
            
            # old_sat: is the literal currently satisfied?
            old_sat = (current_val == lit_desired)  # shape (k,)
            # new_sat: would the literal be satisfied after flip?
            new_sat = (new_val == lit_desired)  # shape (k,)
            
            # Clauses that lose satisfaction: old_sat & ~new_sat
            lose_sat = old_sat & ~new_sat
            # Clauses that gain satisfaction: ~old_sat & new_sat
            gain_sat = ~old_sat & new_sat
            
            # Only count if the clause's sat count makes it critical
            # Lose: if sat_counts[ci] == 1, then flipping makes it unsat
            # Gain: if sat_counts[ci] == 0, then flipping makes it sat
            
            counts = sat_counts[ci_arr]  # shape (k,)
            
            lose_critical = lose_sat & (counts == 1)
            gain_critical = gain_sat & (counts == 0)
            
            gains[pos] = np.sum(gain_critical) - np.sum(lose_critical)
        
        return gains

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    base_noise = 0.3
    base_flip_budget = 50000
    restart_count = 0
    
    # Learned clauses: list of (clause_indices, clause_signs) that should be satisfied
    # Each learned clause is a set of literals that must not all be false simultaneously
    learned_clauses = []
    MAX_LEARNED = 200
    
    def check_learned_clauses(assignment):
        """Check which learned clauses are violated. Returns list of violated clause indices."""
        violated = []
        for li, (lc_idx, lc_sign) in enumerate(learned_clauses):
            vals = np.take(assignment, lc_idx)
            desired = lc_sign > 0
            is_sat = (vals == desired)
            if not np.any(is_sat):
                violated.append(li)
        return violated
    
    def learn_from_conflict(assignment, unsat_mask, sat_counts):
        """
        Learn a clause from the current conflict (unsatisfied clauses).
        The learned clause says: 'not all these literals can be false at once.'
        We pick a subset of unsatisfied clauses and extract their literals.
        """
        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return None
        
        # Pick up to 3 random unsatisfied clauses to form the conflict
        num_to_use = min(3, len(unsat_indices))
        chosen = random.sample(unsat_indices.tolist(), num_to_use)
        
        # Collect all literals from these clauses (as signed 1-based)
        # The learned clause is the disjunction of all these literals
        # (i.e., at least one must be true)
        lit_set = set()
        for ci in chosen:
            for j in range(3):
                var = indices[ci, j]
                sign = signs[ci, j]
                lit = var + 1 if sign > 0 else -(var + 1)
                lit_set.add(lit)
        
        if len(lit_set) == 0:
            return None
        
        # Convert to numpy arrays
        lit_list = list(lit_set)
        lc_idx = np.array([abs(l) - 1 for l in lit_list], dtype=np.int32)
        lc_sign = np.array([1 if l > 0 else -1 for l in lit_list], dtype=np.int8)
        
        return (lc_idx, lc_sign)
    
    while time.time() < deadline:
        # Restart logic: bias toward best assignment when stuck
        if restart_count == 0 or best_assignment is None:
            assignment = initialize_assignment()
        elif random.random() < 0.4:
            # Perturb best assignment
            assignment = best_assignment.copy()
            # Flip some random variables
            num_flips = max(1, n // 20)
            flip_vars = random.sample(range(n), num_flips)
            for v in flip_vars:
                assignment[v] = not assignment[v]
        else:
            assignment = initialize_assignment()
        
        sat_counts = calculate_sat_counts(assignment)
        
        # Build unsat mask (vectorized)
        unsat_mask = (sat_counts == 0)
        current_unsat = int(np.sum(unsat_mask))
        
        # Check learned clauses and add them to the "effective" unsat count
        # We treat violated learned clauses as additional penalties
        learned_violations = check_learned_clauses(assignment)
        effective_unsat = current_unsat + len(learned_violations)
        
        if effective_unsat < best_unsat:
            best_unsat = effective_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget
        if restart_count == 0:
            flip_budget = base_flip_budget
        else:
            flip_budget = int(base_flip_budget * (0.8 + 0.4 * random.random()))
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = effective_unsat
        noise = base_noise
        
        while steps < flip_budget and time.time() < deadline:
            if not np.any(unsat_mask):
                return assignment.tolist()
            
            # Pick a clause to focus on
            # With some probability, pick from learned clauses if violated
            if learned_violations and random.random() < 0.3:
                # Pick a violated learned clause and flip a random literal in it
                li = random.choice(learned_violations)
                lc_idx, lc_sign = learned_clauses[li]
                pos = random.randint(0, len(lc_idx) - 1)
                var = lc_idx[pos]
            else:
                # Pick a random unsatisfied clause from original formula
                unsat_indices = np.where(unsat_mask)[0]
                clause_idx = unsat_indices[random.randint(0, len(unsat_indices) - 1)]
                
                # Pick a literal to flip
                if random.random() < noise:
                    # Random flip from clause
                    pos = random.randint(0, 2)
                    var = indices[clause_idx, pos]
                else:
                    # Find var in clause that maximizes gain (vectorized)
                    gains = calculate_gains_vectorized(clause_idx, assignment, sat_counts)
                    var = indices[clause_idx, np.argmax(gains)]
                    
            # Perform flip
            sat_counts, unsat_mask = flip_variable_fast(var, assignment, sat_counts, unsat_mask)
            
            current_unsat = int(np.sum(unsat_mask))
            learned_violations = check_learned_clauses(assignment)
            effective_unsat = current_unsat + len(learned_violations)
            
            if effective_unsat < best_unsat:
                best_unsat = effective_unsat
                best_assignment = assignment.copy()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
            
            if effective_unsat < local_best_unsat:
                local_best_unsat = effective_unsat
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 100 == 0:
                if no_improvement_steps > 300:
                    noise = min(0.7, noise + 0.05)
                else:
                    noise = max(0.1, noise - 0.02)
                
                # Learn a clause if stuck
                if no_improvement_steps > 500 and len(learned_clauses) < MAX_LEARNED:
                    learned = learn_from_conflict(assignment, unsat_mask, sat_counts)
                    if learned is not None:
                        learned_clauses.append(learned)
        
        restart_count += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
