"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    start_time = time.time()
    time_limit = 280.0  # Safety margin (300s kill, leave buffer)
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess: build clause variables and variable-to-clause index
    clause_vars = []  # list of lists of (var_idx, is_pos)
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(i)
        clause_vars.append(cv)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # WalkSAT with adaptive parameters
    rng = random.Random()
    
    def walk_sat():
        nonlocal best_assignment, best_unsat_count
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Build support counts: for each clause, count how many literals are satisfied
        # support[i] = number of satisfied literals in clause i
        support = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    support[i] += 1
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m):
            if support[i] == 0:
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk parameters
        max_steps = 50000
        noise = 0.45  # Probability of random flip
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick a random unsat clause
            idx = rng.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            # Try to find a flip with minimum damage
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            
            # Evaluate each literal in the clause
            for var_idx, is_pos in clause:
                # Compute damage: how many currently-satisfied clauses become unsat
                # A clause becomes unsat if support[c_idx] == 1 and the satisfied literal is the one we're flipping
                damage = 0
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx == idx:
                        continue
                    if support[c_idx] == 1:
                        # Check if the satisfied literal in c_idx is the one for var_idx
                        cv = clause_vars[c_idx]
                        for v, p in cv:
                            if assignment[v] == p:
                                if v == var_idx:
                                    damage += 1
                                break
                
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            # Decide which variable to flip
            if best_var_idx == -1:
                var_idx, is_pos = rng.choice(clause)
            elif rng.random() < noise or best_damage > 10:
                # Random flip for exploration
                var_idx, is_pos = rng.choice(clause)
            else:
                var_idx = best_var_idx
                is_pos = best_is_pos
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update support counts and unsat_set incrementally
            for c_idx in var_to_clauses[var_idx]:
                cv = clause_vars[c_idx]
                # Find the polarity of var_idx in this clause
                p_c = None
                for v, p in cv:
                    if v == var_idx:
                        p_c = p
                        break
                
                if p_c is None:
                    continue
                
                old_support = support[c_idx]
                if old_val == p_c:
                    # Was satisfied, now unsatisfied
                    support[c_idx] = old_support - 1
                    if old_support == 1:
                        # Becomes unsat
                        unsat_set.add(c_idx)
                else:
                    # Was unsatisfied, now satisfied
                    support[c_idx] = old_support + 1
                    if old_support == 0:
                        # Becomes sat
                        unsat_set.discard(c_idx)
            
            # Check for solution
            if not unsat_set:
                return assignment
            
            # Track best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Adaptive noise: increase noise if stuck
            if step % 500 == 0 and step > 0:
                if current_unsat >= best_unsat_count:
                    noise = min(0.6, noise + 0.02)
                else:
                    noise = max(0.3, noise - 0.01)
        
        return None
    
    # Main loop with restarts
    restarts = 0
    max_restarts = 200
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        result = walk_sat()
        if result is not None:
            return result
        restarts += 1
    
    # Final polish: try to improve best_assignment with a focused local search
    if best_assignment is not None and best_unsat_count > 0:
        polish_time_limit = time_limit - (time.time() - start_time)
        if polish_time_limit > 5.0:
            assignment = best_assignment[:]
            
            # Build support counts
            support = [0] * m
            unsat_set = set()
            for i in range(m):
                cv = clause_vars[i]
                for var_idx, is_pos in cv:
                    if assignment[var_idx] == is_pos:
                        support[i] += 1
                if support[i] == 0:
                    unsat_set.add(i)
            
            current_unsat = len(unsat_set)
            
            # Focused local search with low noise
            max_polish_steps = 10000
            for step in range(max_polish_steps):
                if time.time() - start_time > time_limit:
                    break
                
                if not unsat_set:
                    return assignment
                
                # Pick a random unsat clause
                idx = rng.choice(list(unsat_set))
                clause = clause_vars[idx]
                
                # Try to find a flip with minimum damage
                best_var_idx = -1
                best_damage = float('inf')
                best_is_pos = False
                
                for var_idx, is_pos in clause:
                    damage = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == idx:
                            continue
                        if support[c_idx] == 1:
                            cv = clause_vars[c_idx]
                            for v, p in cv:
                                if assignment[v] == p:
                                    if v == var_idx:
                                        damage += 1
                                    break
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_var_idx = var_idx
                        best_is_pos = is_pos
                
                if best_var_idx == -1:
                    var_idx, is_pos = rng.choice(clause)
                elif rng.random() < 0.3:
                    var_idx, is_pos = rng.choice(clause)
                else:
                    var_idx = best_var_idx
                    is_pos = best_is_pos
                
                # Perform flip
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                # Update support counts and unsat_set
                for c_idx in var_to_clauses[var_idx]:
                    cv = clause_vars[c_idx]
                    p_c = None
                    for v, p in cv:
                        if v == var_idx:
                            p_c = p
                            break
                    
                    if p_c is None:
                        continue
                    
                    old_support = support[c_idx]
                    if old_val == p_c:
                        support[c_idx] = old_support - 1
                        if old_support == 1:
                            unsat_set.add(c_idx)
                    else:
                        support[c_idx] = old_support + 1
                        if old_support == 0:
                            unsat_set.discard(c_idx)
                
                current_unsat = len(unsat_set)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
        
        return best_assignment
    
    # Final fallback: return random
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
