import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Preprocessing
    clause_data = [] # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)] # list of lists of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Parameters tuned for 150 vars, ~450-600 clauses
    max_restarts = 8000 
    base_steps = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7 + 1)
        
        # Greedy initialization: try to satisfy more clauses initially
        assignment = [False] * n
        
        # First pass: set each variable to satisfy as many clauses as possible
        for v in range(n):
            true_count = 0
            false_count = 0
            for c_idx, p in var_clauses[v]:
                # Check if setting v to True or False helps
                if p:
                    true_count += 1
                else:
                    false_count += 1
            assignment[v] = true_count > false_count
        
        # Calc initial sat counts
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive noise: increase as we stall
        stall_counter = 0
        prev_unsat = unsat_count
        noise_prob = 0.2
        
        # Walk with adaptive steps
        max_steps = base_steps + restart // 50  # Gradually increase steps
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Track stalls
            if unsat_count >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = unsat_count
            
            # Increase noise when stalling
            noise_prob = min(0.5, 0.2 + 0.05 * stall_counter)
            
            # Pick random unsat clause
            c_idx = None
            for _ in range(10):
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return assignment

            # Fast gain calculation
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                for c2_idx, p2 in var_clauses[v]:
                    if c2_idx == c_idx:
                        continue
                    
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Apply noise
            if rng.random() < noise_prob:
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                continue

            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
        
        # If we're very close, do a focused local search
        if best_unsat_count <= 5 and best_assignment is not None:
            local_assignment = best_assignment[:]
            local_sat_count = [0] * m
            local_is_unsat = [False] * m
            local_unsat_count = 0
            
            for c_idx in range(m):
                cnt = 0
                for v, p in clause_data[c_idx]:
                    if local_assignment[v] == p:
                        cnt += 1
                local_sat_count[c_idx] = cnt
                if cnt == 0:
                    local_is_unsat[c_idx] = True
                    local_unsat_count += 1
            
            # Try to fix remaining unsat clauses
            for _ in range(200):
                if time.time() > deadline:
                    break
                if local_unsat_count == 0:
                    return local_assignment
                
                # Find an unsat clause
                c_idx = -1
                for idx in range(m):
                    if local_is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx == -1:
                    return local_assignment
                
                # Try flipping each variable in this clause
                best_v = -1
                best_gain = float('inf')
                for v, p in clause_data[c_idx]:
                    gain = 0
                    for c2_idx, p2 in var_clauses[v]:
                        if c2_idx == c_idx:
                            continue
                        was_sat = (local_assignment[v] == p2)
                        if was_sat:
                            if local_sat_count[c2_idx] == 1:
                                gain += 1
                        else:
                            if local_sat_count[c2_idx] == 0:
                                gain -= 1
                    if gain < best_gain:
                        best_gain = gain
                        best_v = v
                
                if best_v != -1:
                    old_val = local_assignment[best_v]
                    local_assignment[best_v] = not old_val
                    for c2_idx, p2 in var_clauses[best_v]:
                        was_sat = (old_val == p2)
                        if was_sat:
                            local_sat_count[c2_idx] -= 1
                            if local_sat_count[c2_idx] == 0:
                                if not local_is_unsat[c2_idx]:
                                    local_is_unsat[c2_idx] = True
                                    local_unsat_count += 1
                        else:
                            local_sat_count[c2_idx] += 1
                            if local_sat_count[c2_idx] == 1:
                                if local_is_unsat[c2_idx]:
                                    local_is_unsat[c2_idx] = False
                                    local_unsat_count -= 1

    if best_assignment is not None:
        return best_assignment
    return [False] * n
