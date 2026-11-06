import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Preprocessing: build efficient data structures
    # clause_vars[i] = list of (var_idx, sign) for clause i
    # var_to_clauses[v] = list of (clause_idx, pos) for variable v
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cv = []
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
            var_to_clauses[v].append((i, j))
        clause_vars.append(cv)
    
    # Precompute for each variable and each clause it appears in,
    # the other two literals (var, sign) to speed up delta computation
    var_other_lits = [{} for _ in range(n)]
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            others = []
            for p in range(3):
                if p != pos:
                    others.append(clause_vars[ci][p])
            var_other_lits[v][ci] = (others[0][0], others[0][1], others[1][0], others[1][1])
    
    # Fast evaluation: returns (num_unsat, unsat_clause_list)
    def fast_evaluate(assignment):
        num_unsat = 0
        unsat_clauses = []
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                val = assignment[v]
                if (s == 1 and val) or (s == -1 and not val):
                    sat = True
                    break
            if not sat:
                num_unsat += 1
                unsat_clauses.append(c)
        return num_unsat, unsat_clauses
    
    # Compute delta for flipping variable v given current assignment
    # delta = change in num_unsat (negative = improvement)
    def compute_delta(v, assignment, unsat_set):
        delta = 0
        for (ci, pos) in var_to_clauses[v]:
            o1v, o1s, o2v, o2s = var_other_lits[v][ci]
            
            o1_sat = (o1s == 1 and assignment[o1v]) or (o1s == -1 and not assignment[o1v])
            o2_sat = (o2s == 1 and assignment[o2v]) or (o2s == -1 and not assignment[o2v])
            
            if o1_sat and o2_sat:
                if ci in unsat_set:
                    delta += 1  # unsat -> sat, improvement
            elif not o1_sat and not o2_sat:
                if ci not in unsat_set:
                    delta -= 1  # sat -> unsat, worsening
        return delta
    
    # WalkSAT with efficient bookkeeping
    def walk_sat(assignment, num_unsat, unsat_clauses, rng, max_steps, deadline):
        unsat_set = set(unsat_clauses)
        
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True
            
            # Pick a random unsatisfied clause
            c = rng.choice(unsat_clauses)
            cv = clause_vars[c]
            
            # For each variable in this clause, compute delta
            best_v = -1
            best_score = -1e9
            for (v, s) in cv:
                delta = compute_delta(v, assignment, unsat_set)
                # Weighted score: prefer larger delta, add small random perturbation
                score = delta * 2.0 + rng.random() * 0.3
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Flip best_v
            old_val = assignment[best_v]
            assignment[best_v] = not old_val
            
            # Update num_unsat and unsat_clauses incrementally
            affected = list(var_to_clauses[best_v])
            
            # Re-evaluate only affected clauses
            # First, remove from unsat_set those that become sat
            # Then, add to unsat_set those that become unsat
            changed = False
            for (ci, pos) in affected:
                cv_ci = clause_vars[ci]
                sat = False
                for (vv, ss) in cv_ci:
                    val = assignment[vv]
                    if (ss == 1 and val) or (ss == -1 and not val):
                        sat = True
                        break
                
                if sat:
                    if ci in unsat_set:
                        unsat_set.discard(ci)
                        num_unsat -= 1
                        changed = True
                else:
                    if ci not in unsat_set:
                        unsat_set.add(ci)
                        num_unsat += 1
                        changed = True
            
            if changed:
                # Rebuild unsat_clauses list from unsat_set
                unsat_clauses = list(unsat_set)
            
            if num_unsat == 0:
                return True
        
        return False
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    best_num_unsat = C + 1
    best_assignment = None
    
    # Multiple restarts with varying parameters
    restart = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Evaluate
        num_unsat, unsat_clauses = fast_evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
        
        # Adaptive step limit: more steps for harder instances
        base_steps = 8000 + restart * 2000
        max_steps = min(base_steps, 200000)
        
        solved = walk_sat(assignment, num_unsat, unsat_clauses, rng, max_steps, deadline)
        
        if solved:
            return assignment
        
        restart += 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
