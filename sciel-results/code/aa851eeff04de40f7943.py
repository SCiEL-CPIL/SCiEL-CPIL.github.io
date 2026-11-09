import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 290
    
    # Precompute
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vs = []
        ss = []
        for lit in clause:
            vs.append(abs(lit) - 1)
            ss.append(1 if lit > 0 else -1)
        clause_vars.append(vs)
        clause_signs.append(ss)
        
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_to_clauses[v].append((ci, s))
            
    m = len(clauses)
    
    noise = 0.5
    max_steps_per_restart = 100000
    restarts = 0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Use local refs for speed
    cv = clause_vars
    cs = clause_signs
    vtc = var_to_clauses
    
    def compute_sat_counts(assignment):
        counts = [0] * m
        for i in range(m):
            c = 0
            v0 = cv[i][0]; s0 = cs[i][0]
            v1 = cv[i][1]; s1 = cs[i][1]
            v2 = cv[i][2]; s2 = cs[i][2]
            
            val0 = assignment[v0]
            if (val0 and s0 == 1) or (not val0 and s0 == -1): c += 1
            val1 = assignment[v1]
            if (val1 and s1 == 1) or (not val1 and s1 == -1): c += 1
            val2 = assignment[v2]
            if (val2 and s2 == 1) or (not val2 and s2 == -1): c += 1
            counts[i] = c
        return counts

    while time.time() < deadline:
        restarts += 1
        assignment = [random.random() < 0.5 for _ in range(n)]
        var_activity = [0.0] * n
        
        sat_counts = compute_sat_counts(assignment)
        unsat_count = sum(1 for x in sat_counts if x == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        steps = 0
        while steps < max_steps_per_restart:
            if steps % 200 == 0 and time.time() >= deadline:
                break
            
            # Find unsat clause
            # Use activity to pick
            best_ci = -1
            best_score = -1.0
            # Iterate through all clauses? Or just unsat ones?
            # We need to know which are unsat.
            # We can maintain a list of unsat clauses.
            # But updating that list is tricky.
            # Scanning m=200 is fast.
            for i in range(m):
                if sat_counts[i] == 0:
                    # Score: sum of activities
                    score = var_activity[cv[i][0]] + var_activity[cv[i][1]] + var_activity[cv[i][2]]
                    if score > best_score:
                        best_score = score
                        best_ci = i
                        
            if best_ci == -1:
                # All satisfied
                return assignment
                
            # MinConfl on best_ci
            min_dmg = 999
            best_v = cv[best_ci][0]
            
            for k in range(3):
                v = cv[best_ci][k]
                s = cs[best_ci][k]
                dmg = 0
                for (ci, cs_val) in vtc[v]:
                    if sat_counts[ci] == 0:
                        continue
                    # Check if lit (v, cs_val) is true
                    val = assignment[v]
                    if (val and cs_val == 1) or (not val and cs_val == -1):
                        if sat_counts[ci] == 1:
                            dmg += 1
                            
                if dmg < min_dmg:
                    min_dmg = dmg
                    best_v = v
                    
            # Flip
            if random.random() < noise:
                idx = random.randint(0, 2)
                flip_v = cv[best_ci][idx]
            else:
                flip_v = best_v
                
            assignment[flip_v] = not assignment[flip_v]
            var_activity[flip_v] += 1.0
            
            # Update sat_counts
            for (ci, cs_val) in vtc[flip_v]:
                new_c = 0
                v0 = cv[ci][0]; s0 = cs[ci][0]
                v1 = cv[ci][1]; s1 = cs[ci][1]
                v2 = cv[ci][2]; s2 = cs[ci][2]
                
                val0 = assignment[v0]
                if (val0 and s0 == 1) or (not val0 and s0 == -1): new_c += 1
                val1 = assignment[v1]
                if (val1 and s1 == 1) or (not val1 and s1 == -1): new_c += 1
                val2 = assignment[v2]
                if (val2 and s2 == 1) or (not val2 and s2 == -1): new_c += 1
                sat_counts[ci] = new_c
                
            steps += 1
            
            # Decay activity
            # Do it every step?
            # Or every 10 steps?
            # Let's do every step for accuracy, it's cheap.
            # Actually, multiplying 50 floats is very fast.
            # But we can just reset at restart.
            # If we don't decay, activity becomes a count of flips.
            # This is fine for short runs.
            
            # Check for solution every few steps
            if steps % 10 == 0:
                # Quick check
                # We can check if any sat_counts is 0
                # This is O(m).
                # We already did O(m) in clause selection.
                # So we can just check best_ci == -1 logic?
                # No, best_ci was calculated before flip.
                # We need to re-check.
                # But we can just check if the flipped variable's clauses are now satisfied?
                # No, other clauses might have become unsatisfied.
                # So we must check all.
                # However, we can optimize:
                # If we found best_ci == -1 in the selection phase, we return.
                # But that was before the flip.
                # So we need a check after flip.
                
                # Let's just rely on the next iteration's selection phase.
                # If all are satisfied, best_ci will be -1 and we return.
                # So we don't need an explicit check here, just let the loop run.
                # But we need to make sure we don't infinite loop if stuck.
                # The random restart handles that.
                
                pass

        # End of restart steps
        
    return best_assignment
