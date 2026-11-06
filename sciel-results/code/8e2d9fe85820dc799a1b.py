"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    start_time = time.time()
    time_limit = 295.0  # Safety margin before 330s kill
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocessing: pure literal elimination and unit propagation
    active_clauses = list(clauses)
    fixed = {}
    
    def preprocess(clause_list, fixed_assign):
        changed = True
        while changed:
            changed = False
            var_polarity = {}
            for c in clause_list:
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        if v not in var_polarity:
                            var_polarity[v] = set()
                        var_polarity[v].add(p)
            
            for v, pols in var_polarity.items():
                if len(pols) == 1 and v not in fixed_assign:
                    p = pols.pop()
                    fixed_assign[v] = p
                    changed = True
            
            new_clauses = []
            for c in clause_list:
                remaining = []
                satisfied = False
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v in fixed_assign:
                        if fixed_assign[v] == p:
                            satisfied = True
                            break
                        else:
                            continue
                    remaining.append(l)
                if satisfied:
                    continue
                new_clauses.append(remaining)
            
            for c in new_clauses:
                if len(c) == 0:
                    return None
                if len(c) == 1:
                    l = c[0]
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        fixed_assign[v] = p
                        changed = True
            
            clause_list = new_clauses
        
        return clause_list
    
    fixed_assign = {}
    remaining_clauses = preprocess(active_clauses, fixed_assign)
    
    if remaining_clauses is None:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    all_sat = True
    for c in remaining_clauses:
        if len(c) == 0:
            all_sat = False
            break
        sat = False
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if v in fixed_assign and fixed_assign[v] == p:
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    unfixed_vars = [i for i in range(n) if i not in fixed_assign]
    var_map = {v: i for i, v in enumerate(unfixed_vars)}
    num_unfixed = len(unfixed_vars)
    
    if num_unfixed == 0:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    # Build internal representation
    # clause_vars[i] = list of (var_idx, polarity_bool)
    # var_to_clauses[j] = list of clause indices containing var j
    clause_vars = []
    var_to_clauses = [[] for _ in range(num_unfixed)]
    
    for i, c in enumerate(remaining_clauses):
        cv = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if v in var_map:
                local_idx = var_map[v]
                cv.append((local_idx, p))
                var_to_clauses[local_idx].append(i)
        clause_vars.append(cv)
    
    m2 = len(clause_vars)
    
    def build_full(assignment):
        full_assign = [fixed_assign.get(i, False) for i in range(n)]
        for j, v in enumerate(unfixed_vars):
            full_assign[v] = assignment[j]
        return full_assign
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def compute_damage(var_idx, current_clause_idx, assignment):
        """
        Compute the damage of flipping var_idx.
        Damage is the number of clauses that become unsatisfied (0 or 1 for 3-SAT usually, 
        but we count all). We exclude the current clause from the "damage" count in the 
        selection heuristic usually, or we just look at total change.
        Standard WalkSAT: pick var that minimizes damage (number of clauses that become 
        unsatisfied). The current clause is already unsatisfied, so if we satisfy it, 
        that's a gain.
        
        Here we calculate:
        1. How many clauses containing var_idx are currently satisfied and will become unsatisfied.
        """
        damage = 0
        for c_idx in var_to_clauses[var_idx]:
            if c_idx == current_clause_idx:
                continue
            # Check if clause c_idx is currently satisfied
            if is_satisfied(c_idx, assignment):
                # Check if var_idx is the ONLY literal satisfying it
                # If yes, flipping it will make it unsatisfied -> damage += 1
                cv = clause_vars[c_idx]
                only_sat = True
                for v, p in cv:
                    if v != var_idx and assignment[v] == p:
                        only_sat = False
                        break
                if only_sat:
                    damage += 1
        return damage

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Optimized WalkSAT with Clause Weights
    def walksat_phase(time_budget):
        nonlocal best_assignment, best_unsat_count
        
        max_restarts = 10000
        base_noise = 0.2
        max_noise = 0.8
        noise_increase = 0.01
        
        while time.time() - start_time < time_budget:
            assignment = [random.random() < 0.5 for _ in range(num_unfixed)]
            
            # Initialize unsat set and weights
            unsat_set = set()
            clause_weights = [1.0] * m2
            for i in range(m2):
                if not is_satisfied(i, assignment):
                    unsat_set.add(i)
            
            if not unsat_set:
                return build_full(assignment)
            
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            steps = 0
            no_improve_steps = 0
            prev_unsat = current_unsat
            max_steps_per_restart = 500 + (100 * len(unsat_set)) # Adaptive limit
            noise = base_noise
            
            while steps < max_steps_per_restart and time.time() - start_time < time_budget:
                if not unsat_set:
                    return build_full(assignment)
                
                # Pick a random unsatisfied clause
                # Weighted pick? For now, uniform is fine, weights are used for damage tie-breaking or just to prioritize hard clauses
                # Let's stick to uniform pick for clause, but use weights to influence variable choice?
                # Standard WalkSAT picks uniform clause.
                idx = random.choice(list(unsat_set))
                clause = clause_vars[idx]
                
                # Pick variable
                if random.random() < noise:
                    var_idx, is_pos = random.choice(clause)
                else:
                    best_var_idx = -1
                    best_damage = float('inf')
                    
                    for var_idx, is_pos in clause:
                        damage = compute_damage(var_idx, idx, assignment)
                        if damage < best_damage:
                            best_damage = damage
                            best_var_idx = var_idx
                            best_is_pos = is_pos
                    
                    if best_var_idx == -1:
                        var_idx, is_pos = random.choice(clause)
                    else:
                        var_idx = best_var_idx
                        is_pos = best_is_pos
                
                # Flip variable
                assignment[var_idx] = not assignment[var_idx]
                
                # Update unsat set and weights
                for c_idx in var_to_clauses[var_idx]:
                    was_unsat = c_idx in unsat_set
                    is_now_sat = is_satisfied(c_idx, assignment)
                    is_now_unsat = not is_now_sat
                    
                    if was_unsat and not is_now_unsat:
                        # Satisfied: remove from set, reset weight
                        unsat_set.remove(c_idx)
                        clause_weights[c_idx] = 1.0
                    elif not was_unsat and is_now_unsat:
                        # Became unsatisfied: add to set, increase weight
                        unsat_set.add(c_idx)
                        clause_weights[c_idx] += 1.0
                    elif not was_unsat and not is_now_unsat:
                        # Still satisfied: increase weight slightly? Or keep?
                        # Usually weights are for unsat clauses.
                        pass
                    else:
                        # Still unsatisfied: increase weight
                        clause_weights[c_idx] += 1.0
                
                steps += 1
                current_unsat = len(unsat_set)
                
                if current_unsat < prev_unsat:
                    no_improve_steps = 0
                    prev_unsat = current_unsat
                    noise = max(base_noise, noise - 0.05) # Reduce noise on improvement
                else:
                    no_improve_steps += 1
                    noise = min(max_noise, noise + noise_increase) # Increase noise on stall
                
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                
                if no_improve_steps > 100 + (50 * (steps // 500)):
                    break
        
        return None

    # Phase 1: Main WalkSAT
    # Allocate 90% of time to search
    phase1_time = start_time + (time_limit - (time.time() - start_time)) * 0.9
    result = walksat_phase(phase1_time)
    if result is not None:
        return result

    # Phase 2: Polish / Refinement
    # If we have a good candidate, try to fix the remaining few unsat clauses
    if best_assignment is not None and best_unsat_count > 0:
        assignment = best_assignment[:]
        unsat_set = set()
        for i in range(m2):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        polish_steps = 0
        max_polish_steps = 5000
        # Use a more aggressive noise for polishing
        noise = 0.5
        
        while polish_steps < max_polish_steps and time.time() - start_time < time_limit - 2.0:
            if not unsat_set:
                return build_full(assignment)
            
            idx = random.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            if random.random() < noise:
                var_idx, is_pos = random.choice(clause)
            else:
                best_var_idx = -1
                best_damage = float('inf')
                for var_idx, is_pos in clause:
                    damage = compute_damage(var_idx, idx, assignment)
                    if damage < best_damage:
                        best_damage = damage
                        best_var_idx = var_idx
                        best_is_pos = is_pos
                
                if best_var_idx == -1:
                    var_idx, is_pos = random.choice(clause)
                else:
                    var_idx = best_var_idx
                    is_pos = best_is_pos
            
            assignment[var_idx] = not assignment[var_idx]
            
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
            
            polish_steps += 1
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
        
        if not unsat_set:
            return build_full(assignment)

    if best_assignment:
        return build_full(best_assignment)
    
    return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

# EVOLVE-BLOCK-END
