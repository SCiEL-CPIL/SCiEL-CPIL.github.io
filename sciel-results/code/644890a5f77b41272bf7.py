"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


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
    
    # Build variable-to-clause index: for each var, list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def init_clause_state(assignment):
        """Compute sat_count for each clause and unsat list."""
        sat_counts = [0] * m
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)
        return sat_counts, unsat_clauses
    
    def walk_sat(assignment, rng, max_flips, deadline, noise_scale=0.3):
        """Incremental WalkSAT with clause state tracking."""
        sat_counts, unsat_clauses = init_clause_state(assignment)
        if not unsat_clauses:
            return True
        
        flip_count = 0
        while flip_count < max_flips:
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decay smoothly
            progress = flip_count / max_flips
            noise = noise_scale * math.exp(-3.0 * progress)
            
            # For each variable in the unsatisfied clause, compute delta
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    sc = sat_counts[ci2]
                    old_contrib = (assignment[var_idx] == is_pos2)
                    
                    if sc > 0:
                        # Clause is satisfied. If this literal is the ONLY satisfying one,
                        # flipping it would make the clause unsat.
                        if old_contrib and sc == 1:
                            delta += 1
                    else:
                        # Clause is unsatisfied. If flipping makes this literal satisfy,
                        # the clause becomes satisfied.
                        if not old_contrib:
                            delta -= 1
                
                # Add noise for exploration
                if noise > 0.01:
                    delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            # Apply flip and update state incrementally
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            for ci2, is_pos2 in var_clauses[best_var]:
                old_contrib = (old_val == is_pos2)
                new_contrib = (new_val == is_pos2)
                
                if old_contrib and not new_contrib:
                    # This literal was satisfying, now it's not
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        unsat_clauses.append(ci2)
                elif not old_contrib and new_contrib:
                    # This literal was not satisfying, now it is
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        # Remove from unsat list
                        try:
                            unsat_clauses.remove(ci2)
                        except ValueError:
                            pass
            
            flip_count += 1
            
            # Check for solution periodically
            if flip_count % 32 == 0 and not unsat_clauses:
                return True
        
        return len(unsat_clauses) == 0
    
    def count_unsat(assignment):
        count = 0
        for cv in clause_vars:
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart: start moderate, increase over time
        if restart < 5:
            max_flips = 8000 + restart * 2000
        elif restart < 15:
            max_flips = 20000 + (restart - 5) * 2000
        else:
            max_flips = 40000 + (restart - 15) * 1000
        
        # Vary noise scale slightly per restart for diversity
        noise_scale = 0.25 + 0.1 * (restart % 5) / 5.0
        
        if walk_sat(assignment, rng, max_flips, deadline, noise_scale):
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    # Final polish: if we have a best assignment, try a short local search on it
    if best_assignment is not None and best_unsat > 0:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            rng = random.Random(12345)
            polish_max_flips = int(remaining_time * 5000)
            polish_max_flips = max(polish_max_flips, 5000)
            if walk_sat(best_assignment, rng, polish_max_flips, deadline, noise_scale=0.15):
                return best_assignment
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
