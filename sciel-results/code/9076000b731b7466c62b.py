"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    # var_clauses[v] = list of (clause_idx, is_positive_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def init_sat_counts(assignment):
        """Compute initial satisfaction counts for each clause."""
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
        return sat_counts

    def get_unsat_clauses(sat_counts):
        """Get list of unsatisfied clause indices."""
        return [i for i, c in enumerate(sat_counts) if c == 0]

    def walk_sat(assignment, rng, max_flips):
        """
        Efficient WalkSAT with incremental sat_counts update.
        """
        sat_counts = init_sat_counts(assignment)
        unsat_clauses = get_unsat_clauses(sat_counts)
        
        if not unsat_clauses:
            return True
            
        flip_count = 0
        # Adaptive noise: start high, decay to low
        # We use a schedule that allows exploration early and exploitation later
        
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Calculate delta for each variable in the clause
            best_var = -1
            best_delta = float('inf')
            
            # Noise magnitude: decreases as we get closer to max_flips
            # This helps escape local minima early and converge later
            progress = flip_count / max_flips
            noise = 0.5 * (1.0 - progress) ** 2
            
            for var_idx, is_pos in cv:
                # Calculate delta in unsat count if we flip var_idx
                delta = 0
                old_val = assignment[var_idx]
                new_val = not old_val
                
                for ci2, is_pos2 in var_clauses[var_idx]:
                    # ci2 is the clause index, is_pos2 is the sign of var_idx in that clause
                    old_count = sat_counts[ci2]
                    
                    # Determine if var_idx currently satisfies clause ci2
                    # var_idx is in clause ci2 with sign is_pos2
                    # It satisfies if assignment[var_idx] == is_pos2
                    old_contrib = 1 if old_val == is_pos2 else 0
                    new_contrib = 1 if new_val == is_pos2 else 0
                    
                    new_count = old_count - old_contrib + new_contrib
                    
                    old_sat = old_count > 0
                    new_sat = new_count > 0
                    
                    if old_sat and not new_sat:
                        delta += 1
                    elif not old_sat and new_sat:
                        delta -= 1
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip from the clause
                best_var = rng.choice([v for v, _ in cv])[0]
            
            # Perform the flip and update sat_counts
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update sat_counts for all clauses containing best_var
            for ci2, is_pos2 in var_clauses[best_var]:
                # Re-evaluate contribution of best_var to ci2
                # Old contribution
                old_contrib = 1 if old_val == is_pos2 else 0
                # New contribution
                new_contrib = 1 if assignment[best_var] == is_pos2 else 0
                
                sat_counts[ci2] += (new_contrib - old_contrib)
            
            flip_count += 1
            
            # Periodically update the list of unsatisfied clauses to avoid stale data
            # Doing it every flip is O(m), doing it every K flips is O(m/K)
            # Since m is small (approx 450 for 150 vars), we can do it frequently
            # But to be safe and fast, let's check if the specific clauses affected changed status
            # Or simply rebuild the list every 10 flips
            
            if flip_count % 10 == 0:
                unsat_clauses = get_unsat_clauses(sat_counts)
                if not unsat_clauses:
                    return True
            else:
                # Optimized check: only check if the clauses affected by the flip changed status
                # This is complex to maintain incrementally without a set of unsat clauses.
                # For simplicity and correctness, we rely on the periodic rebuild.
                # However, if we want to be faster, we can check the specific clauses.
                # Let's stick to periodic rebuild for now as m is small.
                pass

        # Final check
        unsat_clauses = get_unsat_clauses(sat_counts)
        return len(unsat_clauses) == 0

    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart
        # Start with a reasonable number, increase if we have time
        # 10000 flips is usually enough for 150 vars if the heuristic is good
        max_flips = 10000 + restart * 1000
        
        if walk_sat(assignment, rng, max_flips):
            return assignment
        
        # If not solved, check unsat count to keep best attempt
        sat_counts = init_sat_counts(assignment)
        unsat = sum(1 for c in sat_counts if c == 0)
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
        
        # Safety check to ensure we don't run out of time in the middle of a long restart
        # The loop condition handles this, but we might want to break early if we've done many restarts
        # and are not making progress?
        # For now, just let the time limit handle it.
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
