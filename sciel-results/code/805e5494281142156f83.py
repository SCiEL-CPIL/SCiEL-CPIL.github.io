"""Improved planted 3-SAT solver with efficient WalkSAT + restarts."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0
    
    m = len(clauses)
    
    # Preprocess: convert to 0-based, store clause data
    # clause_lits[ci] = list of (var_idx, sign) where sign=1 means +var, -1 means -var
    # var_clauses[v] = list of clause indices containing variable v
    # var_clause_pos[v] = list of (ci, k) where k is position of v in clause ci
    
    clause_lits = []
    var_clauses = [[] for _ in range(n)]
    var_clause_pos = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        cl = []
        for k, lit in enumerate(clause):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cl.append((v, s))
            var_clauses[v].append(ci)
            var_clause_pos[v].append((ci, k))
        clause_lits.append(cl)
    
    # For each clause, precompute for each position the other two positions
    clause_other = []
    for ci in range(m):
        cl = clause_lits[ci]
        others = []
        for k in range(len(cl)):
            o = []
            for j in range(len(cl)):
                if j != k:
                    o.append((j, cl[j]))
            others.append(o)
        clause_other.append(others)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def init_state(assignment):
        clause_sat = [False] * m
        num_unsat = 0
        for ci in range(m):
            sat = False
            for v, s in clause_lits[ci]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            clause_sat[ci] = sat
            if not sat:
                num_unsat += 1
        return clause_sat, num_unsat
    
    def count_unsat(assignment):
        cnt = 0
        for ci in range(m):
            sat = False
            for v, s in clause_lits[ci]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def local_search_polish(assignment, clause_sat, num_unsat, max_steps=2000, noise=0.0):
        """Final polish pass with tighter noise and more iterations"""
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return assignment, clause_sat, num_unsat
            
            ci = random.randrange(m)
            if clause_sat[ci]:
                found = False
                for trial in range(10):
                    ci2 = random.randrange(m)
                    if not clause_sat[ci2]:
                        ci = ci2
                        found = True
                        break
                if not found:
                    for ci2 in range(m):
                        if not clause_sat[ci2]:
                            ci = ci2
                            found = True
                            break
                    if not found:
                        return assignment, clause_sat, num_unsat
            
            cl = clause_lits[ci]
            best_var = -1
            best_score = float('inf')
            
            for k, (v, s) in enumerate(cl):
                gain = 0
                for c_idx, pos in var_clause_pos[v]:
                    others = clause_other[c_idx][pos]
                    sat_by_others = False
                    for j, (v2, s2) in others:
                        if assignment[v2] == (s2 == 1):
                            sat_by_others = True
                            break
                    if sat_by_others:
                        continue
                    if clause_sat[c_idx]:
                        gain -= 1
                    else:
                        gain += 1
                
                score = -gain + random.random() * noise
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                v = random.randrange(n)
            else:
                v = best_var
            
            old_val = assignment[v]
            assignment[v] = not old_val
            
            for c_idx in var_clauses[v]:
                new_sat = False
                for v2, s2 in clause_lits[c_idx]:
                    if assignment[v2] == (s2 == 1):
                        new_sat = True
                        break
                old_sat = clause_sat[c_idx]
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment, clause_sat, num_unsat
        
        return assignment, clause_sat, num_unsat
    
    # Main search: multiple restarts with WalkSAT
    restart = 0
    max_restarts = 8000
    base_steps = 2000
    
    while time.time() < deadline:
        if restart >= max_restarts:
            break
        
        rng = random.Random(restart * 7919 + 31)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
        
        # Adaptive steps: shorter early, longer later
        steps = base_steps + restart * 5
        if steps > 8000:
            steps = 8000
        
        # Noise: start low, ramp up
        noise_start = 0.0
        noise_end = 0.5
        noise_step = (noise_end - noise_start) / max(1, steps)
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            if num_unsat == 0:
                return assignment
            
            # Find unsat clause
            ci = rng.randrange(m)
            if clause_sat[ci]:
                found = False
                for trial in range(15):
                    ci2 = rng.randrange(m)
                    if not clause_sat[ci2]:
                        ci = ci2
                        found = True
                        break
                if not found:
                    for ci2 in range(m):
                        if not clause_sat[ci2]:
                            ci = ci2
                            found = True
                            break
                    if not found:
                        return assignment
            
            cl = clause_lits[ci]
            best_var = -1
            best_score = float('inf')
            
            noise = noise_start + step * noise_step
            noise = min(noise, 0.6)
            
            for k, (v, s) in enumerate(cl):
                gain = 0
                for c_idx, pos in var_clause_pos[v]:
                    others = clause_other[c_idx][pos]
                    sat_by_others = False
                    for j, (v2, s2) in others:
                        if assignment[v2] == (s2 == 1):
                            sat_by_others = True
                            break
                    if sat_by_others:
                        continue
                    if clause_sat[c_idx]:
                        gain -= 1
                    else:
                        gain += 1
                
                score = -gain + rng.random() * noise
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                v = rng.randrange(n)
            else:
                v = best_var
            
            old_val = assignment[v]
            assignment[v] = not old_val
            
            for c_idx in var_clauses[v]:
                new_sat = False
                for v2, s2 in clause_lits[c_idx]:
                    if assignment[v2] == (s2 == 1):
                        new_sat = True
                        break
                old_sat = clause_sat[c_idx]
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment
            
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
        
        restart += 1
    
    # Final polish pass on best assignment found
    if best_assignment is not None and best_unsat_count > 0 and time.time() < deadline - 5:
        best_assignment, _, best_unsat_count = local_search_polish(
            best_assignment[:], 
            init_state(best_assignment)[0], 
            best_unsat_count,
            max_steps=5000,
            noise=0.1
        )
        if best_unsat_count == 0:
            return best_assignment
    
    if best_assignment is not None:
        return best_assignment
    
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
