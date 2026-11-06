import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0 # Safety margin

    # 1. WalkSAT Phase (Fast, Incomplete)
    # Reuse the efficient logic from the current program but with a short time limit.
    
    # Precompute for WalkSAT
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vs, ss = [], []
        for lit in clause:
            vs.append(abs(lit) - 1)
            ss.append(lit > 0)
        clause_vars.append(vs)
        clause_signs.append(ss)
        
    m = len(clauses)
    
    # Variable -> list of (clause_index, position)
    var_to_clauses = [[] for _ in range(n)]
    for i, vs in enumerate(clause_vars):
        for j, v in enumerate(vs):
            var_to_clauses[v].append((i, j))
            
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        idxs = []
        signs = []
        for ci, j in var_to_clauses[v]:
            idxs.append(ci)
            signs.append(clause_signs[ci][j])
        var_clause_idx.append(idxs)
        var_clause_sign.append(signs)
        
    clause_vars_arr = np.array(clause_vars, dtype=np.int32)
    clause_signs_arr = np.array(clause_signs, dtype=np.int8)
    
    def delta_if_flip(v, assignment, csc):
        nv = not assignment[v]
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        delta = 0
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                if old == 0:
                    delta -= 1
            else:
                if old == 1:
                    delta += 1
        return delta

    def flip_var(v, assignment, csc, unsat_count):
        assignment[v] = not assignment[v]
        nv = assignment[v]
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                csc[ci] = old + 1
                if old == 0:
                    unsat_count -= 1
            else:
                csc[ci] = old - 1
                if old == 1:
                    unsat_count += 1
        return unsat_count

    # Run WalkSAT for 5 seconds
    walk_sat_deadline = start_time + 5.0
    rng = random.Random(42)
    best_unsat = m + 1
    best_assignment = None
    noise = 0.20
    
    while time.time() < walk_sat_deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        a_arr = np.array(assignment, dtype=np.int8)
        sat = (a_arr[clause_vars_arr] == clause_signs_arr).astype(np.int32)
        csc = sat.sum(axis=1)
        current_unsat = int(np.sum(csc == 0))
        
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        # Quick greedy
        for _ in range(10):
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return assignment
            cand = set()
            for ci in unsat:
                cand.update(clause_vars[ci])
            any_improved = False
            for v in cand:
                d = delta_if_flip(v, assignment, csc)
                if d < 0:
                    current_unsat = flip_var(v, assignment, csc, current_unsat)
                    any_improved = True
                    if current_unsat == 0:
                        return assignment
            if not any_improved:
                break
        
        # Walk
        for _ in range(1000):
            if time.time() >= walk_sat_deadline:
                break
            if current_unsat == 0:
                return assignment
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return assignment
            ci = int(rng.choice(unsat))
            vs = clause_vars[ci]
            if rng.random() < noise:
                v = vs[rng.randint(0, 2)]
            else:
                # Simple greedy
                best_v = -1
                best_d = 100
                for j in range(3):
                    v = vs[j]
                    d = delta_if_flip(v, assignment, csc)
                    if d < best_d:
                        best_d = d
                        best_v = v
                v = best_v
            current_unsat = flip_var(v, assignment, csc, current_unsat)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
            # Adaptive noise
            if current_unsat > best_unsat:
                noise = min(0.5, noise + 0.05)
            else:
                noise = max(0.05, noise - 0.02)
                
    # 2. DPLL Phase (Complete, Slower)
    # If WalkSAT didn't solve it, try DPLL.
    
    # Convert to DPLL format
    def lit_to_int(lit):
        v = abs(lit) - 1
        return 2*v + 1 if lit > 0 else 2*v + 2
        
    def lit_var(i):
        return (i - 1) // 2
        
    def lit_complement(i):
        return i + 1 if i % 2 == 1 else i - 1
        
    dpll_clauses = []
    for clause in clauses:
        c = [lit_to_int(l) for l in clause]
        c_set = set(c)
        if any(l in c_set for l in [lit_complement(x) for x in c]):
            continue
        dpll_clauses.append(frozenset(c))
        
    if not dpll_clauses:
        return [False] * n

    dpll_deadline = timeout - 5.0
    
    def dpll(clauses, assignment):
        if time.time() > dpll_deadline:
            raise TimeoutError()
            
        # Unit Prop
        while True:
            for c in clauses:
                if not c:
                    return None
                    
            unit_clauses = [c for c in clauses if len(c) == 1]
            if not unit_clauses:
                break
                
            forced = {}
            for c in unit_clauses:
                lit = list(c)[0]
                v = lit_var(lit)
                val = True if lit % 2 == 1 else False
                if v in forced:
                    if forced[v] != val:
                        return None
                else:
                    forced[v] = val
                    
            if not forced:
                break
                
            new_clauses = []
            for c in clauses:
                new_c = set(c)
                sat = False
                for v, val in forced.items():
                    sat_lit = 2*v + 1 if val else 2*v + 2
                    fals_lit = lit_complement(sat_lit)
                    if sat_lit in new_c:
                        sat = True
                        break
                    elif fals_lit in new_c:
                        new_c.remove(fals_lit)
                if not sat:
                    new_clauses.append(new_c)
                    
            for v, val in forced.items():
                assignment[v] = val
            clauses = new_clauses
            
        # Pure Literal
        var_counts = {v: [0, 0] for v in range(n)}
        for c in clauses:
            for lit in c:
                v = lit_var(lit)
                if lit % 2 == 1:
                    var_counts[v][0] += 1
                else:
                    var_counts[v][1] += 1
                    
        pure_vars = {}
        for v in range(n):
            if v in assignment:
                continue
            pos, neg = var_counts[v]
            if pos > 0 and neg == 0:
                pure_vars[v] = True
            elif neg > 0 and pos == 0:
                pure_vars[v] = False
                
        if pure_vars:
            new_clauses = []
            for c in clauses:
                new_c = set(c)
                sat = False
                for v, val in pure_vars.items():
                    sat_lit = 2*v + 1 if val else 2*v + 2
                    if sat_lit in new_c:
                        sat = True
                        break
                if not sat:
                    new_clauses.append(new_c)
            for v, val in pure_vars.items():
                assignment[v] = val
            clauses = new_clauses
            return dpll(clauses, assignment)
            
        if not clauses:
            return assignment
            
        var_deg = {v: 0 for v in range(n)}
        for c in clauses:
            for lit in c:
                v = lit_var(lit)
                if v not in assignment:
                    var_deg[v] += 1
                    
        best_var = -1
        max_deg = -1
        for v in range(n):
            if v not in assignment:
                if var_deg[v] > max_deg:
                    max_deg = var_deg[v]
                    best_var = v
                    
        if best_var == -1:
            return assignment
            
        # Branch True
        assignment[best_var] = True
        lit_t = 2*best_var + 1
        lit_f = 2*best_var + 2
        new_clauses = []
        for c in clauses:
            if lit_t in c:
                continue
            if lit_f in c:
                new_c = set(c)
                new_c.remove(lit_f)
                new_clauses.append(new_c)
            else:
                new_clauses.append(c)
                
        res = dpll(new_clauses, assignment)
        if res is not None:
            return res
        del assignment[best_var]
        
        # Branch False
        assignment[best_var] = False
        lit_t = 2*best_var + 1
        lit_f = 2*best_var + 2
        new_clauses = []
        for c in clauses:
            if lit_f in c:
                continue
            if lit_t in c:
                new_c = set(c)
                new_c.remove(lit_t)
                new_clauses.append(new_c)
            else:
                new_clauses.append(c)
                
        res = dpll(new_clauses, assignment)
        if res is not None:
            return res
        del assignment[best_var]
        return None

    try:
        res = dpll(list(dpll_clauses), {})
        if res is not None:
            return [res.get(i, False) for i in range(n)]
    except RecursionError:
        pass
    except TimeoutError:
        pass
        
    # Fallback
    if best_assignment:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
