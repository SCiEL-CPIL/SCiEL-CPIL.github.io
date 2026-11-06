"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math
import sys


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 285  # safety margin before 330s kill
    
    # Build clause list as list of tuples of literals (signed 1-based)
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)
    
    # Build variable-to-clause index and per-clause literal positions
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)
    
    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        count = 0
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    # Unit propagation and pure literal elimination
    def preprocess(cnf_list):
        """Apply unit propagation and pure literal elimination iteratively."""
        assignment = [None] * n
        cnf_work = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment
                    else:
                        cnf_work[i] = []
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment
            
            lit_counts = [[0, 0] for _ in range(n)]
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen:
                        continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1
            
            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True
                    changed = True
        
        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            if is_satisfied(final_assign):
                return final_assign, None, final_assign
            return None, None, assignment
        
        return None, cnf_work, assignment
    
    # Try preprocessing
    solved, reduced_cnf, partial_assign = preprocess(cnf)
    if solved is not None:
        return solved
    
    def make_full_assignment(partial):
        rng = random.Random()
        assign = []
        for i in range(n):
            if partial[i] is not None:
                assign.append(bool(partial[i]))
            else:
                assign.append(rng.random() < 0.5)
        return assign
    
    if reduced_cnf is not None:
        cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
        m = len(cnf)
        var_to_clauses = [[] for _ in range(n)]
        clause_lit_pos = []
        for ci, c in enumerate(cnf):
            positions = []
            for lit in c:
                vi = abs(lit) - 1
                var_to_clauses[vi].append(ci)
                positions.append((vi, 1 if lit > 0 else -1))
            clause_lit_pos.append(positions)
        start_assign = make_full_assignment(partial_assign)
    else:
        start_assign = [random.Random(0).random() < 0.5 for _ in range(n)]
    
    if is_satisfied(start_assign):
        return start_assign
    
    best_assign = start_assign[:]
    best_unsat = count_unsat(start_assign)
    
    def walksat(assign, max_steps, p_bias, seed):
        """Run WalkSAT with O(degree(vi)) incremental clause tracking."""
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        # Precompute clause_true_count: number of true literals per clause
        clause_true_count = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_lit_pos[ci]:
                if cur[vi] == (sign > 0):
                    cnt += 1
            clause_true_count[ci] = cnt
        
        cur_unsat = sum(1 for c in clause_true_count if c == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        # Track the last time we improved to detect plateaus
        last_improve_step = 0
        plateau_threshold = 500
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            # Detect plateau and restart with new random assignment
            if step - last_improve_step > plateau_threshold and step > 1000:
                # Generate a new random assignment biased toward satisfying more clauses
                fresh = [rng.random() < 0.5 for _ in range(n)]
                fresh_unsat = 0
                for ci in range(m):
                    cnt = 0
                    for vi, sign in clause_lit_pos[ci]:
                        if fresh[vi] == (sign > 0):
                            cnt += 1
                    if cnt == 0:
                        fresh_unsat += 1
                if fresh_unsat < cur_unsat:
                    cur = fresh
                    cur_unsat = fresh_unsat
                    if cur_unsat < min_unsat:
                        min_unsat = cur_unsat
                        min_assign = cur[:]
                    last_improve_step = step
                    continue
            
            # Collect unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if clause_true_count[ci] == 0]
            if not unsat_clauses:
                return True, cur, 0
            
            # Weighted clause selection: use inverse degree weighting
            ci = rng.choice(unsat_clauses)
            
            clause = cnf[ci]
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                var_degree = len(var_to_clauses[vi])
                w = 1.0 / (1.0 + var_degree)
                
                if cur[vi] == (lit > 0):
                    # Flipping would break this clause's literal
                    w *= 0.3
                else:
                    # Flipping would satisfy this literal
                    # Quick net-help estimate
                    help_count = 0
                    hurt_count = 0
                    for aci in var_to_clauses[vi]:
                        if aci == ci:
                            continue
                        for vi2, sign2 in clause_lit_pos[aci]:
                            if vi2 == vi:
                                if sign2 > 0:
                                    if not cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                else:
                                    if cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                break
                    net = help_count - hurt_count
                    w *= (1.0 + 0.5 * net)
                    if w < 0.1:
                        w = 0.1
                weights.append(w)
            
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
            old_val = cur[vi]
            
            # With probability p_bias, set to satisfy; else random flip
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]
            
            # O(degree(vi)) incremental update
            for aci in var_to_clauses[vi]:
                old_count = clause_true_count[aci]
                new_count = 0
                for vi2, sign2 in clause_lit_pos[aci]:
                    if cur[vi2] == (sign2 > 0):
                        new_count += 1
                clause_true_count[aci] = new_count
                if old_count == 0 and new_count > 0:
                    cur_unsat -= 1
                elif old_count > 0 and new_count == 0:
                    cur_unsat += 1
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
                last_improve_step = step
        
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat
    
    # DPLL solver for small instances or as a preprocessing step
    def dpll(cnf_list, assignment, max_depth=20):
        """DPLL with unit propagation and pure literal elimination."""
        if time.time() > deadline - 2:
            return None
        
        # Check if all clauses are satisfied
        cnf_work = []
        for c in cnf_list:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assignment[vi] is not None and assignment[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                cnf_work.append(c)
        
        if not cnf_work:
            result = [bool(a) if a is not None else False for a in assignment]
            return result
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None
                elif len(c) == 0:
                    return None
        
        # Pure literal elimination
        lit_counts = [[0, 0] for _ in range(n)]
        for c in cnf_work:
            seen = set()
            for lit in c:
                vi = abs(lit) - 1
                if vi in seen:
                    continue
                seen.add(vi)
                if lit > 0:
                    lit_counts[vi][0] += 1
                else:
                    lit_counts[vi][1] += 1
        
        for vi in range(n):
            if assignment[vi] is not None:
                continue
            if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                assignment[vi] = False
                changed = True
            elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                assignment[vi] = True
                changed = True
        
        if changed:
            return dpll(cnf_list, assignment, max_depth)
        
        # Find a variable to branch on (choose one with highest degree)
        if max_depth <= 0:
            return None
        
        var_freq = [0] * n
        for c in cnf_work:
            seen = set()
            for lit in c:
                vi = abs(lit) - 1
                if vi not in seen:
                    var_freq[vi] += 1
                    seen.add(vi)
        
        best_var = -1
        best_freq = 0
        for vi in range(n):
            if assignment[vi] is None and var_freq[vi] > best_freq:
                best_freq = var_freq[vi]
                best_var = vi
        
        if best_var == -1:
            result = [bool(a) for a in assignment]
            if is_satisfied(result):
                return result
            return None
        
        old_val = assignment[best_var]
        
        # Try True first
        assignment[best_var] = True
        result = dpll(cnf_list, assignment, max_depth - 1)
        if result is not None:
            return result
        
        # Try False
        assignment[best_var] = False
        result = dpll(cnf_list, assignment, max_depth - 1)
        if result is not None:
            return result
        
        assignment[best_var] = old_val
        return None
    
    # Try DPLL on the reduced instance if it's small enough
    if reduced_cnf is not None and len(reduced_cnf) < 500:
        dpll_assign = [None] * n
        for i in range(n):
            if partial_assign[i] is not None:
                dpll_assign[i] = bool(partial_assign[i])
        
        result = dpll(reduced_cnf, dpll_assign, max_depth=15)
        if result is not None:
            return result
    
    # Multi-restart with adaptive parameters
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    base_steps = 3000
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(base_steps + attempt * 500, 30000)
        
        success, result, min_unsat = walksat(best_assign, steps, p_bias, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
        
        attempt += 1
        
        if attempt % 4 == 0:
            fresh = [random.Random(attempt * 31).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu
    
    if is_satisfied(best_assign):
        return best_assign
    
    # Final polish: greedy hill-climbing on best_assign
    rng = random.Random(12345)
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        vi = rng.randrange(n)
        best_assign[vi] = not best_assign[vi]
        if is_satisfied(best_assign):
            return best_assign
        u = count_unsat(best_assign)
        if u > best_unsat:
            best_assign[vi] = not best_assign[vi]
    
    return best_assign

# EVOLVE-BLOCK-END
