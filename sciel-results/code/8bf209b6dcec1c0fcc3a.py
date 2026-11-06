"""Optimized WalkSAT solver for planted 3-SAT."""
import random
import time
import heapq
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline with safety margin
    deadline = time.time() + 290
    
    # Preprocessing: Convert to 0-based, handle duplicates
    # clauses are lists of signed 1-based literals
    cnf = []
    for c in clauses:
        # Remove duplicates in clause (if x and -x, clause is tautology, remove clause)
        lits = set()
        taut = False
        for lit in c:
            v = abs(lit)
            if v in lits:
                taut = True
                break
            lits.add(v)
        if taut:
            continue
        cnf.append(tuple(abs(lit) - 1 for lit in c)) # 0-based vars, keep sign info separately? 
        # Actually, keep original literals for sign check, but map to 0-based for indexing
    
    # Re-cnf with 0-based vars but keeping sign
    cnf_0based = []
    for c in clauses:
        taut = False
        seen_vars = set()
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            if v in seen_vars:
                taut = True
                break
            seen_vars.add(v)
            new_c.append(lit)
        if not taut:
            cnf_0based.append(tuple(new_c))
            
    if not cnf_0based:
        return [False] * n
        
    m = len(cnf_0based)
    
    # Build index: var -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(cnf_0based):
        for lit in c:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[vi].append((ci, sign))
            
    # Preprocessing: Unit Propagation and Pure Literal Elimination
    # Returns assignment (list of bool/None) and reduced clauses
    def preprocess(cnf_list):
        assignment = [None] * n
        # Track unsatisfied clauses
        active_clauses = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            # Remove satisfied clauses
            new_active = []
            for c in active_clauses:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_active.append(c)
            active_clauses = new_active
            
            # Unit Prop
            for i, c in enumerate(active_clauses):
                if len(c) == 0:
                    return None, None, assignment # UNSAT
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    val = lit > 0
                    if assignment[vi] is None:
                        assignment[vi] = val
                        changed = True
                    elif assignment[vi] != val:
                        return None, None, assignment # Contradiction
                    else:
                        active_clauses[i] = []
                        changed = True
            
            # Pure Literal
            lit_counts = [[0, 0] for _ in range(n)]
            for c in active_clauses:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen: continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1
            for vi in range(n):
                if assignment[vi] is not None: continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True
                    changed = True
                    
        if all(a is not None for a in assignment):
            return [bool(a) for a in assignment], None, assignment
        return None, active_clauses, assignment

    solved, reduced_cnf, partial_assign = preprocess(cnf_0based)
    if solved is not None:
        return solved
        
    if reduced_cnf is not None:
        # Rebuild index for reduced set
        cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
        m = len(cnf)
        var_to_clauses = [[] for _ in range(n)]
        for ci, c in enumerate(cnf):
            for lit in c:
                vi = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                var_to_clauses[vi].append((ci, sign))
    else:
        cnf = cnf_0based
        m = len(cnf)
        
    def count_unsat(assign):
        cnt = 0
        for c in cnf:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assign[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # Initialize assignment
    rng = random.Random(42)
    assign = []
    for i in range(n):
        if partial_assign[i] is not None:
            assign.append(bool(partial_assign[i]))
        else:
            assign.append(rng.random() < 0.5)
            
    if count_unsat(assign) == 0:
        return assign
        
    best_assign = assign[:]
    best_unsat = count_unsat(assign)
    
    # WalkSAT
    def walksat(assign, max_steps, p_bias, seed):
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        # Precompute clause true counts
        clause_true_count = [0] * m
        for ci, c in enumerate(cnf):
            cnt = 0
            for lit in c:
                vi = abs(lit) - 1
                if cur[vi] == (lit > 0):
                    cnt += 1
            clause_true_count[ci] = cnt
            
        cur_unsat = sum(1 for c in clause_true_count if c == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if clause_true_count[ci] == 0]
            if not unsat_clauses:
                return True, cur, 0
                
            ci = rng.choice(unsat_clauses)
            clause = cnf[ci]
            
            # Calculate weights for flipping
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                w = 1.0
                # Simple heuristic: favor variables with lower degree?
                # Or just random
                weights.append(w)
                
            # Weighted random selection
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
                    
            literal = clause[chosen_idx]
            vi = abs(literal) - 1
            
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]
                
            # Update counts
            for aci, sign in var_to_clauses[vi]:
                old_count = clause_true_count[aci]
                new_count = 0
                for lit in cnf[aci]:
                    v = abs(lit) - 1
                    if cur[v] == (lit > 0):
                        new_count += 1
                clause_true_count[aci] = new_count
                if old_count == 0 and new_count > 0:
                    cur_unsat -= 1
                elif old_count > 0 and new_count == 0:
                    cur_unsat += 1
                    
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
                
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat
        
    # Multi-restart
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    base_steps = 10000
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(base_steps + attempt * 2000, 50000)
        
        success, result, min_unsat = walksat(best_assign, steps, p_bias, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
            
        attempt += 1
        if attempt % 5 == 0:
            fresh = [random.Random(attempt).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu
                
    if count_unsat(best_assign) == 0:
        return best_assign
        
    return best_assign
