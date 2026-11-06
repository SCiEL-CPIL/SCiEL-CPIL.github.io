"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 295s to be safe
    deadline = time.time() + 295.0
    
    # Convert 1-based signed literals to 0-based var indices and signs
    cvars = []
    csigns = []
    for cl in clauses:
        vs = []
        ss = []
        for l in cl:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vs.append(v)
            ss.append(s)
        cvars.append(vs)
        csigns.append(ss)
        
    # Build variable-clause incidence
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in cvars[ci]:
            var_clauses[v].append(ci)
            
    # Precompute for each clause and each variable in it, the index of that variable
    # and the sign, to speed up lookups
    clause_var_info = []
    for ci in range(m):
        info = {}
        for k, v in enumerate(cvars[ci]):
            info[v] = (k, csigns[ci][k])
        clause_var_info.append(info)
        
    # Also precompute: for each variable v, for each clause cj containing v, 
    # the sign of v in cj
    var_clause_sign = [[] for _ in range(n)]
    for v in range(n):
        for cj in var_clauses[v]:
            k, s = clause_var_info[cj][v]
            var_clause_sign[v].append((cj, s))
            
    def check_all(assignment):
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                return False
        return True

    # Preprocessing: Pure literal elimination and unit propagation
    def preprocess():
        """
        Apply pure literal elimination and unit propagation.
        Returns: (reduced_cvars, reduced_csigns, fixed_assignment, active_vars)
        fixed_assignment: dict var -> bool for fixed variables
        active_vars: list of variables not fixed
        """
        # Work on copies
        active_clauses = list(range(m))
        clause_active = [True] * m
        var_fixed = {}  # var -> bool
        var_active = [True] * n
        
        # Track which literals appear positively and negatively
        # For each var, count positive and negative occurrences in active clauses
        pos_count = [0] * n
        neg_count = [0] * n
        for ci in range(m):
            for k, v in enumerate(cvars[ci]):
                if csigns[ci][k] == 1:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        changed = True
        max_iter = 1000
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1
            
            # Pure literal elimination
            for v in range(n):
                if not var_active[v]:
                    continue
                if pos_count[v] == 0 and neg_count[v] > 0:
                    # Only appears negatively -> set to False (satisfies all)
                    var_fixed[v] = False
                    var_active[v] = False
                    changed = True
                    # Update counts for clauses containing v
                    for cj in var_clauses[v]:
                        if not clause_active[cj]:
                            continue
                        k, s = clause_var_info[cj][v]
                        if s == 1:
                            pos_count[v] = 0
                        else:
                            neg_count[v] = 0
                        # This clause is now satisfied (since v is pure and we set it to satisfy)
                        # Actually, if v appears only negatively and we set v=False, 
                        # then literal -v is True, so clause is satisfied
                        # Mark clause as satisfied (remove from active)
                        clause_active[cj] = False
                        active_clauses = [c for c in active_clauses if c != cj]
                        # Update counts for other variables in this clause
                        for k2, v2 in enumerate(cvars[cj]):
                            if v2 == v:
                                continue
                            if csigns[cj][k2] == 1:
                                pos_count[v2] -= 1
                            else:
                                neg_count[v2] -= 1
                elif neg_count[v] == 0 and pos_count[v] > 0:
                    # Only appears positively -> set to True
                    var_fixed[v] = True
                    var_active[v] = False
                    changed = True
                    for cj in var_clauses[v]:
                        if not clause_active[cj]:
                            continue
                        k, s = clause_var_info[cj][v]
                        if s == 1:
                            pos_count[v] = 0
                        else:
                            neg_count[v] = 0
                        clause_active[cj] = False
                        active_clauses = [c for c in active_clauses if c != cj]
                        for k2, v2 in enumerate(cvars[cj]):
                            if v2 == v:
                                continue
                            if csigns[cj][k2] == 1:
                                pos_count[v2] -= 1
                            else:
                                neg_count[v2] -= 1
            
            # Unit propagation
            for ci in active_clauses[:]:
                if not clause_active[ci]:
                    continue
                # Count unassigned literals in this clause
                unassigned = []
                for k, v in enumerate(cvars[ci]):
                    if var_active[v]:
                        unassigned.append((k, v, csigns[ci][k]))
                if len(unassigned) == 1:
                    k, v, s = unassigned[0]
                    # Must set v to satisfy this clause
                    val = (s == 1)
                    if v in var_fixed:
                        if var_fixed[v] != val:
                            # Conflict - shouldn't happen for SAT instances, but handle
                            return None
                    else:
                        var_fixed[v] = val
                        var_active[v] = False
                        changed = True
                        # Remove this clause and update counts
                        clause_active[ci] = False
                        active_clauses = [c for c in active_clauses if c != ci]
                        # Update counts for other vars in this clause
                        for k2, v2 in enumerate(cvars[ci]):
                            if v2 == v:
                                continue
                            if csigns[ci][k2] == 1:
                                pos_count[v2] -= 1
                            else:
                                neg_count[v2] -= 1
                        # Also check if setting v satisfies other clauses
                        for cj in var_clauses[v]:
                            if not clause_active[cj]:
                                continue
                            # Check if this clause is now satisfied
                            sat = False
                            for k2, v2 in enumerate(cvars[cj]):
                                if v2 == v:
                                    s2 = csigns[cj][k2]
                                    if (val and s2 == 1) or (not val and s2 == -1):
                                        sat = True
                                        break
                                elif not var_active[v2]:
                                    fv = var_fixed.get(v2)
                                    s2 = csigns[cj][k2]
                                    if fv is not None:
                                        if (fv and s2 == 1) or (not fv and s2 == -1):
                                            sat = True
                                            break
                            if sat:
                                clause_active[cj] = False
                                if cj in active_clauses:
                                    active_clauses = [c for c in active_clauses if c != cj]
                                for k2, v2 in enumerate(cvars[cj]):
                                    if var_active[v2]:
                                        if csigns[cj][k2] == 1:
                                            pos_count[v2] -= 1
                                        else:
                                            neg_count[v2] -= 1
                        # Re-check unit propagation (loop will handle)
                        break  # Restart unit propagation loop
        
        if it >= max_iter:
            # Too many iterations, skip preprocessing
            return None
        
        # Build reduced instance
        reduced_cvars = []
        reduced_csigns = []
        for ci in active_clauses:
            if not clause_active[ci]:
                continue
            vs = []
            ss = []
            for k, v in enumerate(cvars[ci]):
                if var_active[v]:
                    vs.append(v)
                    ss.append(csigns[ci][k])
            if len(vs) == 0:
                # Clause is empty and unsatisfied -> UNSAT (shouldn't happen for planted SAT)
                return None
            if len(vs) == 1:
                # Unit clause in reduced instance - propagate
                v = vs[0]
                s = ss[0]
                val = (s == 1)
                var_fixed[v] = val
                var_active[v] = False
                # Remove this clause
                continue
            reduced_cvars.append(vs)
            reduced_csigns.append(ss)
        
        active_vars = [v for v in range(n) if var_active[v]]
        
        return (reduced_cvars, reduced_csigns, var_fixed, active_vars)

    # Try preprocessing
    pre_result = preprocess()
    
    if pre_result is not None:
        red_cvars, red_csigns, var_fixed, active_vars = pre_result
        rn = len(active_vars)
        rm = len(red_cvars)
        
        # Check if all variables are fixed
        if rn == 0:
            # All variables fixed, check if all clauses satisfied
            full_assignment = [False] * n
            for v, val in var_fixed.items():
                full_assignment[v] = val
            if check_all(full_assignment):
                return full_assignment
            # Otherwise, fall through to full search
            red_cvars = cvars
            red_csigns = csigns
            rn = n
            rm = m
            var_fixed = {}
            active_vars = list(range(n))
        else:
            # Build reduced incidence
            var_to_idx = {v: i for i, v in enumerate(active_vars)}
            
            red_var_clauses = [[] for _ in range(rn)]
            for ci in range(rm):
                for v in red_cvars[ci]:
                    idx = var_to_idx[v]
                    red_var_clauses[idx].append(ci)
                    
            red_clause_var_info = []
            for ci in range(rm):
                info = {}
                for k, v in enumerate(red_cvars[ci]):
                    idx = var_to_idx[v]
                    info[idx] = (k, red_csigns[ci][k])
                red_clause_var_info.append(info)
                
            red_var_clause_sign = [[] for _ in range(rn)]
            for idx in range(rn):
                for cj in red_var_clauses[idx]:
                    k, s = red_clause_var_info[cj][idx]
                    red_var_clause_sign[idx].append((cj, s))
            
            # Clause weights: weight = 1 / (number of free vars in clause)
            # or use frequency-based weighting
            clause_weight = [1.0 / len(cl) for cl in red_cvars]
            
            # Build full assignment from fixed + reduced
            def full_assignment_from_reduced(red_assign):
                fa = [False] * n
                for v, val in var_fixed.items():
                    fa[v] = val
                for i, v in enumerate(active_vars):
                    fa[v] = red_assign[i]
                return fa
            
            best_assignment = None
            best_unsat = rm + 1
            rng = random.Random(42)
            
            def check_reduced(red_assign):
                for ci in range(rm):
                    vs = red_cvars[ci]
                    ss = red_csigns[ci]
                    sat = False
                    for k in range(len(vs)):
                        v_idx = var_to_idx[vs[k]]
                        s = ss[k]
                        if s == 1:
                            if red_assign[v_idx]:
                                sat = True
                                break
                        else:
                            if not red_assign[v_idx]:
                                sat = True
                                break
                    if not sat:
                        return False
                return True
            
            def run_walksat_reduced():
                nonlocal best_assignment, best_unsat, deadline
                
                red_assign = [rng.random() < 0.5 for _ in range(rn)]
                
                unsat_count = [0] * rm
                total_unsat = 0
                for ci in range(rm):
                    vs = red_cvars[ci]
                    ss = red_csigns[ci]
                    sat = False
                    for k in range(len(vs)):
                        v_idx = var_to_idx[vs[k]]
                        s = ss[k]
                        if s == 1:
                            if red_assign[v_idx]:
                                sat = True
                                break
                        else:
                            if not red_assign[v_idx]:
                                sat = True
                                break
                    if not sat:
                        unsat_count[ci] = 1
                        total_unsat += 1
                        
                if total_unsat == 0:
                    return red_assign
                    
                if total_unsat < best_unsat:
                    best_unsat = total_unsat
                    best_assignment = red_assign[:]
                    
                unsat_clauses = [ci for ci in range(rm) if unsat_count[ci] > 0]
                
                max_steps = 300000
                noise_start = 0.25
                noise_end = 0.05
                noise_decay = 0.99999
                
                noise = noise_start
                
                for step in range(max_steps):
                    if time.time() > deadline:
                        break
                        
                    if not unsat_clauses:
                        return red_assign
                        
                    if len(unsat_clauses) < best_unsat:
                        best_unsat = len(unsat_clauses)
                        best_assignment = red_assign[:]
                        
                    # Weighted clause selection
                    total_w = sum(clause_weight[ci] for ci in unsat_clauses)
                    if total_w <= 0:
                        ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
                    else:
                        r = rng.random() * total_w
                        cum = 0
                        ci = unsat_clauses[-1]
                        for uci in unsat_clauses:
                            cum += clause_weight[uci]
                            if cum >= r:
                                ci = uci
                                break
                                
                    vs = red_cvars[ci]
                    ss = red_csigns[ci]
                    
                    # Calculate scores
                    scores = []
                    for v in vs:
                        v_idx = var_to_idx[v]
                        score = 0.0
                        for cj, s_cj in red_var_clause_sign[v_idx]:
                            if cj == ci:
                                continue
                            if unsat_count[cj] == 0:
                                continue
                            # Check if flipping v satisfies cj
                            if s_cj == 1:
                                if not red_assign[v_idx]:
                                    score += clause_weight[cj]
                            else:
                                if red_assign[v_idx]:
                                    score += clause_weight[cj]
                        scores.append(score)
                        
                    # Pick variable to flip
                    if rng.random() < (1 - noise):
                        max_score = max(scores)
                        candidates = [i for i, s in enumerate(scores) if s == max_score]
                        idx = candidates[rng.randrange(len(candidates))]
                    else:
                        idx = rng.randrange(len(vs))
                        
                    v_to_flip = vs[idx]
                    v_idx = var_to_idx[v_to_flip]
                    
                    old_val = red_assign[v_idx]
                    new_val = not old_val
                    red_assign[v_idx] = new_val
                    
                    # Update unsat counts
                    for cj, s_v in red_var_clause_sign[v_idx]:
                        was_unsat = unsat_count[cj] > 0
                        
                        lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                        lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                        
                        if was_unsat:
                            if lit_now_true:
                                unsat_count[cj] = 0
                                total_unsat -= 1
                        else:
                            if lit_was_true and not lit_now_true:
                                any_other_true = False
                                vs_cj = red_cvars[cj]
                                ss_cj = red_csigns[cj]
                                for k in range(len(vs_cj)):
                                    v_k_idx = var_to_idx[vs_cj[k]]
                                    if v_k_idx == v_idx:
                                        continue
                                    s_k = ss_cj[k]
                                    if (red_assign[v_k_idx] and s_k == 1) or (not red_assign[v_k_idx] and s_k == -1):
                                        any_other_true = True
                                        break
                                if not any_other_true:
                                    unsat_count[cj] = 1
                                    total_unsat += 1
                    
                    if step % 50 == 0:
                        unsat_clauses = [ci2 for ci2 in range(rm) if unsat_count[ci2] > 0]
                        
                    if total_unsat == 0:
                        return red_assign
                        
                    if total_unsat < best_unsat:
                        best_unsat = total_unsat
                        best_assignment = red_assign[:]
                        
                    noise = max(noise_end, noise * noise_decay)
                    
                return red_assign
            
            # Run WalkSAT on reduced instance
            while time.time() < deadline:
                if best_unsat == 0:
                    break
                    
                result = run_walksat_reduced()
                if result is not None:
                    fa = full_assignment_from_reduced(result)
                    if check_all(fa):
                        return fa
                
            if best_assignment is not None:
                fa = full_assignment_from_reduced(best_assignment)
                if check_all(fa):
                    return fa
                return fa
            return [False] * n
    
    # Fallback: full WalkSAT on original instance
    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    def run_walksat():
        nonlocal best_assignment, best_unsat, deadline
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = [0] * m
        total_unsat = 0
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                unsat_count[ci] = 1
                total_unsat += 1
                
        if total_unsat == 0:
            return assignment
            
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]
            
        unsat_clauses = [ci for ci in range(m) if unsat_count[ci] > 0]
        
        max_steps = 200000
        noise_start = 0.3
        noise_end = 0.05
        noise_decay = 0.999995
        
        noise = noise_start
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                return assignment
                
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
                
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            vs = cvars[ci]
            ss = csigns[ci]
            
            scores = []
            for v in vs:
                score = 0
                for cj, s_cj in var_clause_sign[v]:
                    if cj == ci:
                        continue
                    if unsat_count[cj] == 0:
                        continue
                    if s_cj == 1:
                        if not assignment[v]:
                            score += 1
                    else:
                        if assignment[v]:
                            score += 1
                scores.append(score)
                
            if rng.random() < (1 - noise):
                max_score = max(scores)
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = candidates[rng.randrange(len(candidates))]
            else:
                idx = rng.randrange(len(vs))
                
            v_to_flip = vs[idx]
            
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            for cj, s_v in var_clause_sign[v_to_flip]:
                was_unsat = unsat_count[cj] > 0
                
                lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                
                if was_unsat:
                    if lit_now_true:
                        unsat_count[cj] = 0
                        total_unsat -= 1
                else:
                    if lit_was_true and not lit_now_true:
                        any_other_true = False
                        vs_cj = cvars[cj]
                        ss_cj = csigns[cj]
                        for k in range(len(vs_cj)):
                            if vs_cj[k] == v_to_flip:
                                continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            unsat_count[cj] = 1
                            total_unsat += 1
            
            if step % 100 == 0:
                unsat_clauses = [ci2 for ci2 in range(m) if unsat_count[ci2] > 0]
                
            if total_unsat == 0:
                return assignment
                
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
                
            noise = max(noise_end, noise * noise_decay)
            
        return assignment

    while time.time() < deadline:
        if best_unsat == 0:
            break
            
        result = run_walksat()
        if result is not None:
            if check_all(result):
                return result
        
    if best_assignment is not None:
        if check_all(best_assignment):
            return best_assignment
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
