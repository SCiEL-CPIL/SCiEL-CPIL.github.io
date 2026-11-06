"""Optimized 3-SAT solver using WalkSAT with adaptive clause weighting."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    start_time = time.time()
    time_limit = 295.0  # Safety margin
    
    # Precompute var_to_clauses: list of (clause_idx, sign) for each variable
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_to_clauses[v].append((ci, sign))
    
    best_unsat = num_clauses
    best_assignment = None
    
    # Adaptive clause weighting: weight unsat clauses higher to focus search
    clause_weight = [1.0] * num_clauses
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        rng = random.Random(restart * 12345 + 7)
        
        # Initial assignment: random
        assignment = [False] * n
        for i in range(n):
            assignment[i] = rng.random() < 0.5
            
        # Initialize clause_sat_count
        clause_sat_count = [0] * num_clauses
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    count += 1
            clause_sat_count[ci] = count
            
        curr_unsat = sum(1 for c in clause_sat_count if c == 0)
        
        if curr_unsat == 0:
            return assignment
            
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = assignment[:]
            
        max_flips = 80000 + 300 * n
        base_noise = 0.25
        
        # Track unsat history for adaptive noise
        unsat_history = []
        
        for flip_count in range(max_flips):
            if time.time() - start_time > time_limit - 3.0:
                break
                
            if curr_unsat == 0:
                return assignment
                
            # Pick random unsat clause with adaptive weighting
            # Build weighted selection: unsat clauses get higher weight
            unsat_indices = [i for i in range(num_clauses) if clause_sat_count[i] == 0]
            if not unsat_indices:
                break
            
            # Weight by clause_weight (which increases for persistently unsat clauses)
            weights = [clause_weight[i] for i in unsat_indices]
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumulative = 0
            ci = unsat_indices[0]
            for idx, w in zip(unsat_indices, weights):
                cumulative += w
                if r <= cumulative:
                    ci = idx
                    break
            
            clause = clauses[ci]
            
            # Calculate gain for each literal in clause
            gains = []
            for lit in clause:
                v = abs(lit) - 1
                
                unsat_gain = 0
                sat_loss = 0
                
                for (ci2, sign2) in var_to_clauses[v]:
                    if clause_sat_count[ci2] == 0:
                        unsat_gain += 1
                    elif clause_sat_count[ci2] == 1:
                        if assignment[v] == (sign2 == 1):
                            sat_loss += 1
                            
                gain = unsat_gain - sat_loss
                gains.append((gain, v))
                
            # Adaptive noise: increase when stuck, decrease when improving
            if len(unsat_history) >= 100:
                recent = unsat_history[-100:]
                if len(set(recent)) <= 2:  # Stuck
                    noise_prob = min(0.5, base_noise + 0.15)
                elif recent[-1] < recent[0]:  # Improving
                    noise_prob = max(0.1, base_noise - 0.1)
                else:
                    noise_prob = base_noise
            else:
                noise_prob = base_noise
                
            # Select flip
            if rng.random() < noise_prob:
                lit_choice = rng.choice(clause)
                v = abs(lit_choice) - 1
            else:
                max_gain = max(g for g, _ in gains)
                candidates = [v for g, v in gains if g == max_gain]
                v = rng.choice(candidates)
                
            # Perform flip
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat_count and clause weights
            delta_unsat = 0
            for (ci2, sign) in var_to_clauses[v]:
                old_sat = (old_val == (sign == 1))
                new_sat = (assignment[v] == (sign == 1))
                
                if old_sat and not new_sat:
                    clause_sat_count[ci2] -= 1
                    if clause_sat_count[ci2] == 0:
                        delta_unsat += 1
                        # Increase weight for newly unsat clause
                        clause_weight[ci2] = min(10.0, clause_weight[ci2] * 1.5)
                elif not old_sat and new_sat:
                    clause_sat_count[ci2] += 1
                    if clause_sat_count[ci2] == 1:
                        delta_unsat -= 1
                        # Reset weight for newly sat clause
                        clause_weight[ci2] = 1.0
                        
            curr_unsat += delta_unsat
            
            if curr_unsat == 0:
                return assignment
                
            if curr_unsat < best_unsat:
                best_unsat = curr_unsat
                best_assignment = assignment[:]
                
            # Track unsat history
            if flip_count % 10 == 0:
                unsat_history.append(curr_unsat)
                if len(unsat_history) > 200:
                    unsat_history = unsat_history[-200:]
                    
            # Decay clause weights slowly to allow exploration
            if flip_count % 1000 == 0:
                for i in range(num_clauses):
                    if clause_weight[i] > 1.0:
                        clause_weight[i] = max(1.0, clause_weight[i] * 0.95)
                
    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
