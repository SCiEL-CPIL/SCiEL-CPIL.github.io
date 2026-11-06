# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using optimized WalkSAT.
    
    Key optimizations:
    1. Use clause satisfaction counts for O(1) updates on flip
    2. Precompute variable-to-clause indices with literal positions
    3. Fast gain computation using satisfaction counts
    4. Multiple restart strategies with adaptive parameters
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 325.0
    
    # Precompute: for each variable, list of (clause_idx, literal_sign)
    # literal_sign: +1 if positive literal, -1 if negative
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var].append((i, sign))
    
    # Precompute clause literals as (var, sign) pairs for faster access
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        cv = []
        cs = []
        for lit in clause:
            cv.append(abs(lit) - 1)
            cs.append(1 if lit > 0 else -1)
        clause_vars.append(cv)
        clause_signs.append(cs)
    
    def init_sat_counts(assignment):
        """Initialize clause satisfaction counts."""
        sat_counts = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            cnt = 0
            for j in range(3):
                v = cv[j]
                s = cs[j]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            sat_counts[i] = cnt
        return sat_counts
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses using sat counts."""
        sat_counts = init_sat_counts(assignment)
        return all(c > 0 for c in sat_counts)
    
    def walk_sat(assignment, max_steps, rng, p=0.5):
        """
        Optimized WalkSAT with incremental clause satisfaction tracking.
        """
        sat_counts = init_sat_counts(assignment)
        unsat_count = sum(1 for c in sat_counts if c == 0)
        
        if unsat_count == 0:
            return True
        
        # Track unsatisfied clauses for efficient random selection
        unsat_clauses = [i for i in range(m) if sat_counts[i] == 0]
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
            
            if unsat_count == 0:
                return True
            
            # Pick a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            # Evaluate flip candidates
            best_gain = -10
            best_vars = []
            
            for j in range(3):
                v = cv[j]
                s = cs[j]
                # Check if this literal is currently satisfied
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    continue
                
                # Compute gain
                gain = 1  # This clause becomes satisfied (or more satisfied)
                
                for (ci2, s2) in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    # Check if flipping v would break clause ci2
                    # Current state of literal in ci2:
                    v2 = v
                    s2_val = s2
                    cur_sat = (assignment[v2] and s2_val == 1) or (not assignment[v2] and s2_val == -1)
                    if cur_sat and sat_counts[ci2] == 1:
                        # This is the only satisfying literal in ci2
                        gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_vars = [v]
                elif gain == best_gain:
                    best_vars.append(v)
            
            if not best_vars:
                # Shouldn't happen, but pick randomly
                v = cv[rng.randrange(3)]
            elif rng.random() < p:
                v = best_vars[rng.randrange(len(best_vars))]
            else:
                v = best_vars[rng.randrange(len(best_vars))]
            
            # Apply flip
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            
            # Update satisfaction counts for all clauses containing v
            for (ci2, s2) in var_to_clauses[v]:
                old_sat = (old_val and s2 == 1) or (not old_val and s2 == -1)
                new_sat = (assignment[v] and s2 == 1) or (not assignment[v] and s2 == -1)
                
                if old_sat and not new_sat:
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        unsat_count += 1
                        unsat_clauses.append(ci2)
                elif not old_sat and new_sat:
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        unsat_count -= 1
                        # Remove ci2 from unsat_clauses (mark as -1 for lazy deletion)
                        # Find and remove
                        for k in range(len(unsat_clauses)):
                            if unsat_clauses[k] == ci2:
                                unsat_clauses[k] = unsat_clauses[-1]
                                unsat_clauses.pop()
                                break
        
        return unsat_count == 0
    
    # Strategy 1: Quick random checks
    for attempt in range(10):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        if is_satisfied(assignment):
            return assignment
    
    # Strategy 2: WalkSAT with multiple restarts and varying p
    base_steps = max(2000, n * 30)
    
    # Phase 1: Standard WalkSAT
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Vary p: mix of low and high
        if restart % 3 == 0:
            p = 0.3
        elif restart % 3 == 1:
            p = 0.5
        else:
            p = 0.7
        
        if walk_sat(assignment, base_steps, rng, p=p):
            return assignment
    
    # Phase 2: More aggressive with varied step counts
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(trial * 9973 + 777)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        p = rng.uniform(0.2, 0.8)
        steps = int(base_steps * (0.5 + rng.random()))
        
        if walk_sat(assignment, steps, rng, p=p):
            return assignment
    
    # Phase 3: Try with different initial bias
    for bias_trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(bias_trial * 50021 + 999)
        bias = 0.3 + 0.4 * bias_trial / 19.0
        assignment = [rng.random() < bias for _ in range(n)]
        
        p = 0.5
        steps = base_steps
        
        if walk_sat(assignment, steps, rng, p=p):
            return assignment
    
    # Fallback: return last assignment
    return assignment
# EVOLVE-BLOCK-END
