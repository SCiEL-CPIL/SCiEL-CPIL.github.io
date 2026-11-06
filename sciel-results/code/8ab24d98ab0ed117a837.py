"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Preprocessing: Map variables to their occurrences in clauses
    # var_clauses[v] = list of (clause_idx, sign) where sign is 1 for +v, -1 for -v
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((c_idx, sign))
    
    num_clauses = len(clauses)
    all_clauses_mask = (1 << num_clauses) - 1
    
    # Function to initialize state from an assignment
    def init_state(assignment):
        clause_sat_count = [0] * num_clauses
        sat_mask = 0
        for c_idx, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                is_true = assignment[v] == (lit > 0)
                if is_true:
                    count += 1
            clause_sat_count[c_idx] = count
            if count > 0:
                sat_mask |= (1 << c_idx)
        return clause_sat_count, sat_mask

    # Function to flip a variable and update state
    # Returns new sat_mask
    def flip_variable(assignment, v, clause_sat_count, sat_mask):
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        for c_idx, sign in var_clauses[v]:
            # Check if the literal was satisfied before
            # Literal is satisfied if (val == (sign > 0))
            was_sat = (old_val == (sign > 0))
            is_sat = (new_val == (sign > 0))
            
            if was_sat and not is_sat:
                # Literal went from True to False
                clause_sat_count[c_idx] -= 1
                if clause_sat_count[c_idx] == 0:
                    sat_mask &= ~(1 << c_idx)
            elif not was_sat and is_sat:
                # Literal went from False to True
                clause_sat_count[c_idx] += 1
                if clause_sat_count[c_idx] == 1:
                    sat_mask |= (1 << c_idx)
        
        return sat_mask

    # WalkSAT with restarts
    best_assignment = None
    min_unsat = num_clauses + 1
    
    rng = random.Random(42)
    
    max_restarts = 50
    steps_per_restart = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count, sat_mask = init_state(assignment)
        
        if sat_mask == all_clauses_mask:
            return assignment
        
        unsat_count = num_clauses - bin(sat_mask).count("1")
        if unsat_count < min_unsat:
            min_unsat = unsat_count
            best_assignment = assignment.copy()
            
        # Local search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
                
            if sat_mask == all_clauses_mask:
                return assignment
            
            # Find unsat clauses
            unsat_mask = all_clauses_mask ^ sat_mask
            
            # Pick a random unsat clause
            # To pick random bit from mask:
            # We can iterate through bits or use a trick.
            # Since n=50, num_clauses is likely small (< 1000), we can just find one.
            # For efficiency, we can maintain a list of unsat clause indices, but updating it is O(k).
            # Given the small size, picking a random index from 0 to num_clauses-1 and checking if it's unsat is fast enough.
            # Or better: find the lowest set bit? No, random is better for WalkSAT.
            
            # Efficient random unsat clause selection:
            # If unsat_mask is sparse, this might be slow if we just sample.
            # But num_clauses is small.
            # Let's just sample an index and check.
            c_idx = rng.randint(0, num_clauses - 1)
            if not (unsat_mask & (1 << c_idx)):
                # Retry a few times or just pick a known one?
                # For robustness, let's just pick the first unsat clause found by scanning?
                # No, scanning is O(C).
                # Let's use a helper to pick a random set bit.
                # Python doesn't have built-in random bit selection.
                # Given C is small, we can just do:
                # unsat_indices = [i for i in range(num_clauses) if (unsat_mask >> i) & 1]
                # This is O(C). With C ~ 100-1000, and 1000 steps, it's 100k ops. Fine.
                # But we can do better.
                pass
            
            # Let's implement a faster random bit picker for small integers
            # Or just accept O(C) for clause selection if C is small.
            # Actually, we can store unsat clauses in a list and update it?
            # Let's stick to the bitmask but optimize the selection.
            
            # Simple approach: pick random clause, if sat, try again (up to 5 times), else pick first unsat.
            # This is inefficient if few clauses are unsat.
            
            # Better: Maintain a list of unsat clause indices.
            # Let's refactor to maintain `unsat_indices` list.
            # This makes selection O(1).
            
            break # Break to refactor logic in main loop below for clarity
            
    # Refactored WalkSAT with unsat list maintenance
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count, sat_mask = init_state(assignment)
        
        if sat_mask == all_clauses_mask:
            return assignment
        
        # Build initial unsat list
        unsat_indices = [i for i in range(num_clauses) if not (sat_mask & (1 << i))]
        
        if len(unsat_indices) < min_unsat:
            min_unsat = len(unsat_indices)
            best_assignment = assignment.copy()
            
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment if best_assignment is not None else assignment
                
            if not unsat_indices:
                return assignment
            
            # Pick random unsat clause
            c_idx = rng.choice(unsat_indices)
            clause = clauses[c_idx]
            
            # Pick random variable from this clause
            lit = rng.choice(clause)
            v = abs(lit) - 1
            
            # Flip
            old_sat_mask = sat_mask
            sat_mask = flip_variable(assignment, v, clause_sat_count, sat_mask)
            
            # Update unsat_indices list
            # Only clauses containing v changed.
            # We need to update the list for those clauses.
            # This is tricky to do in-place efficiently.
            # Alternative: Rebuild unsat_indices every K steps?
            # Or just check if the specific clauses affected are now sat/unsat and update the list.
            
            # For simplicity and speed in Python, if num_clauses is small,
            # rebuilding the list is O(C).
            # If we do this every step, it's O(C*Steps).
            # Let's try to update only affected clauses in the list.
            
            # Actually, maintaining a sorted list or set is annoying.
            # Let's just rebuild the list every 10 steps or if the list is large.
            # Or better, just use the bitmask to find an unsat clause directly.
            
            # Let's go back to bitmask selection but optimize it.
            # If unsat_mask is not zero:
            # We can pick a random bit by:
            # m = unsat_mask
            # r = rng.getrandbits(m.bit_length())
            # while not (m & (1 << r)):
            #    r = rng.getrandbits(m.bit_length())
            # This is expected O(1) if density is not too low.
            
            if sat_mask == all_clauses_mask:
                return assignment
                
            # Update unsat_indices only if we are using it.
            # Let's switch to the bitmask random selection method for simplicity and correctness.
            
            m = all_clauses_mask ^ sat_mask
            if m == 0:
                return assignment
            
            # Pick random bit from m
            # Since m can be up to 1000 bits, bit_length is ~10.
            # This is fast.
            r = rng.getrandbits(m.bit_length())
            while not (m & (1 << r)):
                r = rng.getrandbits(m.bit_length())
                
            c_idx = r
            clause = clauses[c_idx]
            lit = rng.choice(clause)
            v = abs(lit) - 1
            
            sat_mask = flip_variable(assignment, v, clause_sat_count, sat_mask)
            
            # Track best
            current_unsat = num_clauses - bin(sat_mask).count("1")
            if current_unsat < min_unsat:
                min_unsat = current_unsat
                best_assignment = assignment.copy()

    return best_assignment if best_assignment is not None else assignment
# EVOLVE-BLOCK-END
