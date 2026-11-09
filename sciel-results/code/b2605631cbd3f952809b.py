import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
    # Data structures
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
    # High restarts, moderate steps
    max_restarts = 5000 
    steps_per_restart = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7 + 1)
        
        # Init random
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
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
            
        # Walk
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Pick random unsat clause
            # unsat_list might contain stale entries (marked is_unsat=False)
            # We pick a random index from unsat_list. If it's stale, we retry.
            # To prevent infinite loop if list is mostly stale, we can clean it occasionally
            # or just rely on the fact that unsat_count > 0 implies at least one valid entry.
            # Since we don't remove from unsat_list when fixed, it can grow.
            # However, we only add when it becomes unsat.
            # Stale entries are those that became sat.
            # If unsat_list is large and unsat_count is small, retries could be expensive.
            # For N=150, M~600, unsat_list max size is M.
            # Let's just pick a random one and check.
            
            c_idx = None
            # Limit retries to avoid pathological cases, though unlikely here
            for _ in range(10):
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                # Fallback: scan for an unsat clause (rare/expensive)
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return assignment # Should not happen if unsat_count > 0

            # Evaluate flips for literals in c_idx
            # Heuristic: Min-Conflicts (minimize number of newly unsat clauses)
            # Score = (clauses that become unsat) - (clauses that become sat)
            # Actually, standard WalkSAT uses:
            # 1. Calculate "gain" for each var in clause:
            #    gain = (# clauses that become unsat) - (# clauses that become sat)
            #    We want to minimize gain.
            #    Note: Flipping a var in an unsat clause:
            #    - If var was False and clause needs True, flipping to True satisfies this clause (gain -1 for this clause? No, this clause is currently unsat, so it becomes sat. That's a "fix").
            #    - Other clauses containing this var:
            #       - If they were sat and this was the only sat literal, they become unsat (break).
            #       - If they were unsat, they might become sat (fix).
            
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                # Iterate over clauses containing v
                for c2_idx, p2 in var_clauses[v]:
                    if c2_idx == c_idx:
                        continue # Skip the clause we are trying to fix? 
                        # Actually, flipping v in c_idx:
                        # If v satisfies c_idx, c_idx becomes sat (fixed +1)
                        # If v does not satisfy c_idx, c_idx remains unsat (no change)
                        # But we know c_idx is unsat.
                        # If we flip v, does it become sat?
                        # If p == assignment[v] (current value matches polarity), then v is already satisfying c_idx?
                        # No, c_idx is unsat, so NO literal in c_idx satisfies it.
                        # So for all (v, p) in c_idx, assignment[v] != p.
                        # If we flip v, assignment[v] becomes not assignment[v].
                        # Since assignment[v] != p, not assignment[v] == p.
                        # So v WILL satisfy c_idx.
                        # Thus, c_idx becomes sat. fixed += 1.
                        
                        # Wait, let's verify.
                        # c_idx is unsat => for all l in c_idx, assignment[l.var] != l.pol.
                        # We pick v from c_idx.
                        # Current assignment[v] != p.
                        # Flip v => new assignment[v] = not assignment[v].
                        # Since old != p, new == p.
                        # So v satisfies c_idx.
                        # c_idx becomes sat.
                        # So "fixed" for c_idx is 1.
                        
                        # However, in the loop below, we iterate var_clauses[v].
                        # c_idx is in var_clauses[v].
                        # sat_count[c_idx] is 0.
                        # was_sat = (assignment[v] == p2). Here p2 is p.
                        # assignment[v] != p, so was_sat is False.
                        # So it goes to "else" branch (Gained satisfaction).
                        # sat_count[c_idx] becomes 1.
                        # is_unsat[c_idx] becomes False.
                        # So it is counted as "fixed" if we check sat_count[c2_idx] == 0.
                        # So we don't need special handling for c_idx if we use the general logic.
                        
                    # Check if v is currently satisfied in c2
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        # Becomes unsat if this was the only satisfied literal
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        # Becomes sat
                        # If it was 0, it becomes 1 (fixed)
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Noise
            # Standard WalkSAT: with probability p, pick random var from clause
            # Otherwise pick best var
            # p is often 0.1 to 0.5. Let's use 0.3.
            if rng.random() < 0.3:
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                # Should not happen
                continue

            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    # Lost a satisfaction
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        # Became unsat
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    # Gained a satisfaction
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        # Became sat
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            # Do not remove from unsat_list to avoid O(N) shift
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

    # Fallback
    if best_assignment is not None:
        return best_assignment
    return assignment
