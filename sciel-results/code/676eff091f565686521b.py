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
        Uses a smooth surrogate: instead of counting only "only-satisfied" clauses,
        we weight each satisfied clause by how critical it is (inverse of number of satisfying literals).
        This provides a smoother landscape for variable selection.
        """
        damage = 0.0
        for c_idx in var_to_clauses[var_idx]:
            if c_idx == current_clause_idx:
                continue
            cv = clause_vars[c_idx]
            # Count how many literals in this clause are currently satisfied
            num_satisfied = 0
            for v, p in cv:
                if assignment[v] == p:
                    num_satisfied += 1
            
            if num_satisfied == 0:
                # Clause is already unsatisfied - flipping won't make it worse
                continue
            
            # If var_idx is currently satisfying this clause, flipping will reduce num_satisfied by 1
            # The "damage" is higher if this clause has few other satisfying literals
            # Use a smooth approximation: 1/(num_satisfied) as weight
            # If num_satisfied == 1 and var_idx is the only one, damage = 1.0
            # If num_satisfied == 2 and var_idx is one of them, damage = 0.5
            # If num_satisfied == 3 and var_idx is one of them, damage = 0.33
            if assignment[var_idx] == dict(cv).get(var_idx, False):
                # var_idx is currently satisfying this clause
                # After flip, num_satisfied becomes num_satisfied - 1
                if num_satisfied - 1 == 0:
                    damage += 1.0  # Clause will become unsatisfied
                else:
                    damage += 1.0 / num_satisfied  # Smooth penalty
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
            clause_unsat_since = [0] * m2  # Track how long each clause has been unsatisfied
            for i in range(m2):
                if not is_satisfied(i, assignment):
                    unsat_set.add(i)
                    clause_unsat_since[i] = 1
            
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
                
                # Weighted clause selection: prioritize clauses that have been unsatisfied longer
                # Use exponential weighting based on how long clause has been unsatisfied
                weights = []
                for c_idx in unsat_set:
                    # Weight increases with time unsatisfied, capped to avoid overflow
                    w = min(clause_unsat_since[c_idx], 100)
                    weights.append(w)
                
                # Weighted random choice
                total_weight = sum(weights)
                if total_weight == 0:
                    idx = random.choice(list(unsat_set))
                else:
                    r = random.random() * total_weight
                    cumulative = 0
                    idx = list(unsat_set)[0]
                    for c_idx, w in zip(unsat_set, weights):
                        cumulative += w
                        if r <= cumulative:
                            idx = c_idx
                            break
                
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
                        # Satisfied: remove from set, reset weight and unsat_since
                        unsat_set.remove(c_idx)
                        clause_weights[c_idx] = 1.0
                        clause_unsat_since[c_idx] = 0
                    elif not was_unsat and is_now_unsat:
                        # Became unsatisfied: add to set, increase weight
                        unsat_set.add(c_idx)
                        clause_weights[c_idx] += 1.0
                        clause_unsat_since[c_idx] = 1
                    elif not was_unsat and not is_now_unsat:
                        # Still satisfied: keep weight, reset unsat_since
                        clause_unsat_since[c_idx] = 0
                    else:
                        # Still unsatisfied: increase weight and unsat_since
                        clause_weights[c_idx] += 1.0
                        clause_unsat_since[c_idx] += 1
                
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
        max_polish_steps = 10000
        # Use an adaptive noise schedule: start aggressive, become more greedy as we get closer
        noise = 0.6
        noise_decay = 0.005
        
        while polish_steps < max_polish_steps and time.time() - start_time < time_limit - 2.0:
            if not unsat_set:
                return build_full(assignment)
            
            # Weighted clause selection based on how many times each clause has been selected
            # (proxy for hardness)
            weights = [1.0] * m2
            for c_idx in unsat_set:
                # Increase weight for clauses that are part of many unsat clauses' variable overlaps
                weights[c_idx] = 1.0 + 0.1 * len(var_to_clauses[var_idx]) if var_idx is not None else 1.0
            
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
            
            # Decay noise as we make progress
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
                noise = max(0.1, noise - noise_decay)
        
        if not unsat_set:
            return build_full(assignment)

    if best_assignment:
        return build_full(best_assignment)
    
    return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

# EVOLVE-BLOCK-END
