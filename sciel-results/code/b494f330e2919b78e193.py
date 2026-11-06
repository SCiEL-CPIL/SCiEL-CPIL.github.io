# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using optimized WalkSAT with
    precomputed critical clause tracking for O(1) gain updates.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 325.0
    
    # Precompute clause data as parallel arrays for speed
    clause_vars = [[0, 0, 0] for _ in range(m)]
    clause_signs = [[1, 1, 1] for _ in range(m)]
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i][j] = abs(lit) - 1
            clause_signs[i][j] = 1 if lit > 0 else -1
    
    # For each variable, list of (clause_idx, literal_sign)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i][j]
            s = clause_signs[i][j]
            var_to_clauses[v].append((i, s))
    
    def init_sat_counts(assignment):
        sat_counts = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            cnt = 0
            v0, v1, v2 = cv[0], cv[1], cv[2]
            s0, s1, s2 = cs[0], cs[1], cs[2]
            if (assignment[v0] and s0 == 1) or (not assignment[v0] and s0 == -1):
                cnt += 1
            if (assignment[v1] and s1 == 1) or (not assignment[v1] and s1 == -1):
                cnt += 1
            if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                cnt += 1
            sat_counts[i] = cnt
        return sat_counts
    
    def walk_sat(assignment, max_steps, rng, p=0.5):
        sat_counts = init_sat_counts(assignment)
        unsat_count = sum(1 for c in sat_counts if c == 0)
        
        if unsat_count == 0:
            return True
        
        unsat_clauses = [i for i in range(m) if sat_counts[i] == 0]
        unsat_set = set(unsat_clauses)
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
            
            if unsat_count == 0:
                return True
            
            if len(unsat_clauses) > unsat_count * 2 + 10:
                unsat_clauses = [i for i in unsat_clauses if sat_counts[i] == 0]
                unsat_set = set(unsat_clauses)
            
            if not unsat_clauses:
                return True
            
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            best_gain = -10
            best_vars = []
            
            for j in range(3):
                v = cv[j]
                s = cs[j]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    continue
                
                gain = 1
                for (ci2, s2) in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    cur_sat = (assignment[v] and s2 == 1) or (not assignment[v] and s2 == -1)
                    if cur_sat and sat_counts[ci2] == 1:
                        gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_vars = [v]
                elif gain == best_gain:
                    best_vars.append(v)
            
            if not best_vars:
                v = cv[rng.randrange(3)]
            elif rng.random() < p:
                v = best_vars[rng.randrange(len(best_vars))]
            else:
                v = cv[rng.randrange(3)]
            
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            for (ci2, s2) in var_to_clauses[v]:
                old_sat = (old_val and s2 == 1) or (not old_val and s2 == -1)
                new_sat = (assignment[v] and s2 == 1) or (not assignment[v] and s2 == -1)
                
                if old_sat and not new_sat:
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        unsat_count += 1
                        if ci2 not in unsat_set:
                            unsat_set.add(ci2)
                            unsat_clauses.append(ci2)
                elif not old_sat and new_sat:
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        unsat_count -= 1
                        if ci2 in unsat_set:
                            unsat_set.discard(ci2)
        
        return unsat_count == 0
    
    # Quick random checks
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = init_sat_counts(assignment)
        if all(c > 0 for c in sat_counts):
            return assignment
    
    base_steps = max(2000, n * 30)
    
    # Phase 1: Standard WalkSAT with varied p
    for restart in range(300):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        if restart % 5 == 0:
            p = 0.3
        elif restart % 5 == 1:
            p = 0.4
        elif restart % 5 == 2:
            p = 0.5
        elif restart % 5 == 3:
            p = 0.6
        else:
            p = 0.7
        
        if walk_sat(assignment, base_steps, rng, p=p):
            return assignment
    
    # Phase 2: Varied step counts and p
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(trial * 9973 + 777)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        p = rng.uniform(0.2, 0.8)
        steps = int(base_steps * (0.5 + rng.random()))
        
        if walk_sat(assignment, steps, rng, p=p):
            return assignment
    
    # Phase 3: Different initial bias
    for bias_trial in range(30):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(bias_trial * 50021 + 999)
        bias = 0.3 + 0.4 * bias_trial / 29.0
        assignment = [rng.random() < bias for _ in range(n)]
        
        p = 0.5
        steps = base_steps
        
        if walk_sat(assignment, steps, rng, p=p):
            return assignment
    
    return assignment
# EVOLVE-BLOCK-END
