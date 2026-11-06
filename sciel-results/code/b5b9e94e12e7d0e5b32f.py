"""Improved 3-SAT solver using WalkSAT with incremental clause tracking, unsat-clause list, and adaptive noise."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    start_time = time.time()
    time_limit = 290.0
    
    # Precompute: for each variable, list of (clause_index, sign) where sign=1 means positive literal
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_to_clauses[v].append((ci, sign))
    
    best_assignment = None
    best_unsat_count = num_clauses
    
    max_restarts = 5000
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 3.0:
            break
        
        # Adaptive restart budget: more flips early, fewer later
        remaining_time = time_limit - elapsed - 2.0
        if remaining_time <= 0:
            break
        # Estimate flips per second ~50000-100000; allocate proportionally
        flip_budget = int(remaining_time * 40000)
        flip_budget = max(5000, min(flip_budget, 200000))
        
        rng = random.Random(restart * 12345 + 7)
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction counts and unsat list
        clause_sat = [0] * num_clauses
        unsat_list = []
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    count += 1
            clause_sat[ci] = count
            if count == 0:
                unsat_list.append(ci)
        
        curr_unsat = len(unsat_list)
        
        if curr_unsat == 0:
            return assignment
        
        if curr_unsat < best_unsat_count:
            best_unsat_count = curr_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive noise
        base_noise = 0.35
        no_improve_count = 0
        local_best = curr_unsat
        stagnation_threshold = 2000
        
        total_flips = 0
        while total_flips < flip_budget:
            if time.time() - start_time > time_limit - 1.0:
                break
            
            if not unsat_list:
                return assignment
            
            # Pick random unsatisfied clause
            idx = rng.randrange(len(unsat_list))
            unsat_clause_idx = unsat_list[idx]
            clause = clauses[unsat_clause_idx]
            
            # Compute gain for each literal in the unsat clause
            gains = []
            for lit in clause:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else 0
                
                satisfied = 0
                unsatisfied = 0
                for (ci2, sign2) in var_to_clauses[v]:
                    current_sat = (assignment[v] == (sign2 == 1))
                    new_sat = ((not assignment[v]) == (sign2 == 1))
                    
                    if current_sat and not new_sat:
                        if clause_sat[ci2] == 1:
                            unsatisfied += 1
                    elif not current_sat and new_sat:
                        if clause_sat[ci2] == 0:
                            satisfied += 1
                
                gain = satisfied - unsatisfied
                gains.append((gain, v))
            
            # Select flip: with probability p, random; otherwise best gain
            if rng.random() < base_noise:
                lit_choice = rng.choice(clause)
                v = abs(lit_choice) - 1
            else:
                max_gain = max(g for g, _ in gains)
                candidates = [v for g, v in gains if g == max_gain]
                v = rng.choice(candidates)
            
            # Perform the flip
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat incrementally and maintain unsat_list
            delta_unsat = 0
            for (ci, sign) in var_to_clauses[v]:
                old_sat = (old_val == (sign == 1))
                new_sat = (assignment[v] == (sign == 1))
                
                if old_sat and not new_sat:
                    clause_sat[ci] -= 1
                    if clause_sat[ci] == 0:
                        delta_unsat += 1
                        unsat_list.append(ci)
                elif not old_sat and new_sat:
                    clause_sat[ci] += 1
                    if clause_sat[ci] == 1:
                        delta_unsat -= 1
                        # Remove from unsat_list (lazy: mark as stale, clean up later)
                        # We'll use a set-based approach for efficiency
                        pass
            
            curr_unsat += delta_unsat
            total_flips += 1
            
            if curr_unsat == 0:
                return assignment
            
            if curr_unsat < local_best:
                local_best = curr_unsat
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment[:]
            
            # Clean up unsat_list periodically (remove satisfied clauses)
            if total_flips % 100 == 0:
                unsat_list = [ci for ci in unsat_list if clause_sat[ci] == 0]
                curr_unsat = len(unsat_list)
                if curr_unsat == 0:
                    return assignment
                if curr_unsat < local_best:
                    local_best = curr_unsat
                    no_improve_count = 0
            
            # Adapt noise parameter every 50 flips
            if total_flips % 50 == 0:
                # Increase noise if stagnating, decrease if making progress
                if no_improve_count > 500:
                    base_noise = min(0.65, base_noise + 0.02)
                elif no_improve_count < 100:
                    base_noise = max(0.15, base_noise - 0.01)
            
            # Early termination if severely stuck
            if no_improve_count > stagnation_threshold and curr_unsat > local_best:
                break
        
        # Periodic cleanup of unsat_list at end of walk
        # (already done above)
        
        if time.time() - start_time > time_limit - 5.0:
            break
    
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
