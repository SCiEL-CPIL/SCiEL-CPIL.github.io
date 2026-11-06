"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    start_time = time.time()
    time_limit = 295.0  # Safety margin before 330s kill
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocessing: pure literal elimination and unit propagation
    # Represent clauses as lists of signed literals (1-based)
    active_clauses = list(clauses)
    fixed = {}  # var_idx -> bool
    
    def preprocess(clause_list, fixed_assign):
        """Apply pure literal elimination and unit propagation iteratively."""
        changed = True
        while changed:
            changed = False
            # Find pure literals
            var_polarity = {}  # var_idx -> set of polarities seen
            for c in clause_list:
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        if v not in var_polarity:
                            var_polarity[v] = set()
                        var_polarity[v].add(p)
            
            # Pure literal: only one polarity appears
            for v, pols in var_polarity.items():
                if len(pols) == 1 and v not in fixed_assign:
                    p = pols.pop()
                    fixed_assign[v] = p
                    changed = True
            
            # Unit propagation
            new_clauses = []
            for c in clause_list:
                # Remove satisfied literals, check if any fixed to False
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
                            continue  # literal is False, skip
                    remaining.append(l)
                if satisfied:
                    continue
                new_clauses.append(remaining)
            
            # If any clause is empty, UNSAT (shouldn't happen for planted satisfiable)
            for c in new_clauses:
                if len(c) == 0:
                    return None  # UNSAT
                if len(c) == 1:
                    l = c[0]
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        fixed_assign[v] = p
                        changed = True
            
            clause_list = new_clauses
        
        return clause_list
    
    # Run preprocessing
    fixed_assign = {}
    remaining_clauses = preprocess(active_clauses, fixed_assign)
    
    if remaining_clauses is None:
        # Shouldn't happen for satisfiable instances, but fallback
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    # Check if all clauses satisfied by fixed assignments
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
    
    # Build data structures for remaining problem
    # Only consider unfixed variables
    unfixed_vars = [i for i in range(n) if i not in fixed_assign]
    var_map = {v: i for i, v in enumerate(unfixed_vars)}
    num_unfixed = len(unfixed_vars)
    
    # Rebuild clauses with only unfixed variables
    clause_vars = []  # list of (local_var_idx, is_pos)
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
    
    # If no unfixed variables needed, we're done
    if num_unfixed == 0:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Adaptive restart strategy with noise
    max_steps_per_restart = 1000
    restarts = 0
    max_restarts = 10000
    noise_param = 0.2  # Probability of random flip
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        # Random initialization
        assignment = [random.random() < 0.5 for _ in range(num_unfixed)]
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m2):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        if not unsat_set:
            # Build full assignment
            full_assign = [fixed_assign.get(i, False) for i in range(n)]
            for j, v in enumerate(unfixed_vars):
                full_assign[v] = assignment[j]
            return full_assign
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        steps = 0
        no_improve_steps = 0
        prev_unsat = current_unsat
        
        while steps < max_steps_per_restart and time.time() - start_time < time_limit:
            if not unsat_set:
                full_assign = [fixed_assign.get(i, False) for i in range(n)]
                for j, v in enumerate(unfixed_vars):
                    full_assign[v] = assignment[j]
                return full_assign
            
            # Pick random unsat clause
            idx = random.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            # Adaptive noise: increase noise if stuck
            current_noise = noise_param * (1.0 + no_improve_steps / 200.0)
            current_noise = min(current_noise, 0.8)
            
            if random.random() < current_noise:
                # Random flip
                var_idx, is_pos = random.choice(clause)
            else:
                # Greedy: pick literal that minimizes damage
                best_var_idx = -1
                best_damage = float('inf')
                best_is_pos = False
                
                for var_idx, is_pos in clause:
                    # Count damage: clauses containing var_idx that become unsat
                    damage = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == idx:
                            continue
                        
                        # Find polarity of var_idx in c_idx
                        p_c = None
                        for v, p in clause_vars[c_idx]:
                            if v == var_idx:
                                p_c = p
                                break
                        if p_c is None:
                            continue
                        
                        was_sat = (assignment[var_idx] == p_c)
                        
                        if was_sat:
                            # Check if any other literal in c_idx is satisfied
                            remains_sat = False
                            for v, p in clause_vars[c_idx]:
                                if v == var_idx:
                                    continue
                                if assignment[v] == p:
                                    remains_sat = True
                                    break
                            if not remains_sat:
                                damage += 1
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_var_idx = var_idx
                        best_is_pos = is_pos
                
                if best_var_idx == -1:
                    var_idx, is_pos = random.choice(clause)
                else:
                    var_idx = best_var_idx
                    is_pos = best_is_pos
            
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_set
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
            
            steps += 1
            current_unsat = len(unsat_set)
            
            if current_unsat < prev_unsat:
                no_improve_steps = 0
                prev_unsat = current_unsat
            else:
                no_improve_steps += 1
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Early exit if stuck for too long
            if no_improve_steps > 500:
                break
        
        restarts += 1
    
    # Final polish: if we have a good best_assignment, try to improve it
    if best_assignment is not None and best_unsat_count > 0 and best_unsat_count < m2 * 0.1:
        polish_time_limit = time_limit - (time.time() - start_time)
        if polish_time_limit > 5.0:
            assignment = best_assignment[:]
            unsat_set = set()
            for i in range(m2):
                if not is_satisfied(i, assignment):
                    unsat_set.add(i)
            
            polish_steps = 0
            max_polish_steps = 5000
            while polish_steps < max_polish_steps and time.time() - start_time < time_limit - 2.0:
                if not unsat_set:
                    full_assign = [fixed_assign.get(i, False) for i in range(n)]
                    for j, v in enumerate(unfixed_vars):
                        full_assign[v] = assignment[j]
                    return full_assign
                
                idx = random.choice(list(unsat_set))
                clause = clause_vars[idx]
                
                # Always greedy in polish
                best_var_idx = -1
                best_damage = float('inf')
                best_is_pos = False
                
                for var_idx, is_pos in clause:
                    damage = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == idx:
                            continue
                        p_c = None
                        for v, p in clause_vars[c_idx]:
                            if v == var_idx:
                                p_c = p
                                break
                        if p_c is None:
                            continue
                        was_sat = (assignment[var_idx] == p_c)
                        if was_sat:
                            remains_sat = False
                            for v, p in clause_vars[c_idx]:
                                if v == var_idx:
                                    continue
                                if assignment[v] == p:
                                    remains_sat = True
                                    break
                            if not remains_sat:
                                damage += 1
                    
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
                
                if not unsat_set:
                    full_assign = [fixed_assign.get(i, False) for i in range(n)]
                    for j, v in enumerate(unfixed_vars):
                        full_assign[v] = assignment[j]
                    return full_assign
    
    if best_assignment:
        full_assign = [fixed_assign.get(i, False) for i in range(n)]
        for j, v in enumerate(unfixed_vars):
            full_assign[v] = best_assignment[j]
        return full_assign
    
    # Fallback: random assignment
    return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

# EVOLVE-BLOCK-END
