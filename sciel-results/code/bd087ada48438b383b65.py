"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin for 330s kill

    # --- Preprocessing ---
    m = len(clauses)
    
    # Convert to 0-based (var_idx, is_pos)
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
        
    # Build reverse index: var_idx -> list of (clause_idx, is_pos_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))

    # --- Helper Functions ---
    
    def time_up():
        return time.time() > deadline

    def calc_sat_counts(assignment):
        """Calculate satisfaction counts for all clauses."""
        counts = [0] * m
        for i in range(m):
            cnt = 0
            for var_idx, is_pos in clause_vars[i]:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts

    def init_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walk_sat(assignment, max_steps, noise):
        """
        Optimized WalkSAT with incremental count updates and efficient unsat tracking.
        Returns (assignment, solved, unsat_count)
        """
        counts = calc_sat_counts(assignment)
        
        # Maintain a list of unsatisfied clauses for O(1) random access and O(1) removal (swap-pop)
        unsat_list = []
        unsat_pos = [-1] * m  # Index in unsat_list, or -1 if not unsat
        
        for i in range(m):
            if counts[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)
                
        if not unsat_list:
            return assignment, True, 0

        for _ in range(max_steps):
            if time_up():
                return assignment, False, len(unsat_list)
            
            if not unsat_list:
                return assignment, True, 0
                
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_list)
            clause = clause_vars[clause_idx]
            
            # Heuristic: Pick variable with minimum damage
            best_var = -1
            best_flip_val = False
            best_damage = float('inf')
            
            for var_idx, is_pos in clause:
                # We want to flip var_idx to is_pos to satisfy this clause.
                # Since the clause is unsat, currently assignment[var_idx] != is_pos.
                
                # Calculate damage:
                # How many satisfied clauses containing (var_idx, NOT is_pos) will become unsatisfied?
                damage = 0
                opposite = not is_pos
                
                for c_idx, c_is_pos in var_clauses[var_idx]:
                    if c_idx == clause_idx:
                        continue
                    if c_is_pos == opposite:
                        # This other clause has the literal that will become false.
                        # If it is currently satisfied (counts > 0), and this is its ONLY satisfied literal (counts == 1),
                        # it will become unsatisfied.
                        if counts[c_idx] == 1:
                            damage += 1
                            # Early exit if damage exceeds current best
                            if damage >= best_damage:
                                break
                
                if damage < best_damage:
                    best_damage = damage
                    best_var = var_idx
                    best_flip_val = is_pos
                elif damage == best_damage and rng.random() < 0.5:
                    # Random tie-breaking
                    best_var = var_idx
                    best_flip_val = is_pos
            
            # With some probability, pick a random variable from the clause (noise)
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
                best_var = var_idx
                best_flip_val = is_pos
            
            # Perform flip
            if best_var != -1:
                old_val = assignment[best_var]
                new_val = best_flip_val
                
                # If no change needed (shouldn't happen for unsat clause literals, but safe)
                if old_val == new_val:
                    continue
                
                assignment[best_var] = new_val
                
                # Update counts and unsat list incrementally
                for c_idx, c_is_pos in var_clauses[best_var]:
                    was_sat = (old_val == c_is_pos)
                    is_sat = (new_val == c_is_pos)
                    
                    if was_sat and not is_sat:
                        # Lost satisfaction
                        counts[c_idx] -= 1
                        if counts[c_idx] == 0:
                            # Newly unsatisfied
                            unsat_pos[c_idx] = len(unsat_list)
                            unsat_list.append(c_idx)
                    elif not was_sat and is_sat:
                        # Gained satisfaction
                        counts[c_idx] += 1
                        if counts[c_idx] == 1:
                            # Was unsat, now sat
                            pos = unsat_pos[c_idx]
                            if pos != -1:
                                # Swap and pop
                                last_idx = unsat_list[-1]
                                unsat_list[pos] = last_idx
                                unsat_pos[last_idx] = pos
                                unsat_list.pop()
                                unsat_pos[c_idx] = -1

        return assignment, len(unsat_list) == 0, len(unsat_list)

    # --- Main Loop ---
    
    best_assignment = init_assignment()
    best_unsat = float('inf')
    
    # Adaptive parameters
    steps_per_restart = 5000
    noise = 0.25
    
    while not time_up():
        # Generate new assignment
        # Mix of random and best-so-far to balance exploration and exploitation
        if best_unsat < float('inf') and rng.random() < 0.2:
            assignment = best_assignment[:]
        else:
            assignment = init_assignment()
            
        assignment, solved, cur_unsat = walk_sat(assignment, steps_per_restart, noise)
        
        if solved:
            return assignment
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Optional: Adjust noise or steps if stuck? 
        # For now, keep simple. The "damage" heuristic is quite strong.
        
    return best_assignment

# EVOLVE-BLOCK-END
