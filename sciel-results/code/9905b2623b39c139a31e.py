import random
import time
import sys
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Safety deadline: 290 seconds to leave buffer
    deadline = time.time() + 290.0

    # Preprocess: Remove duplicates, handle tautologies, convert to 0-based
    # We will work with a list of sets of literals for DPLL efficiency
    # Literal: (var, sign) where var is 0..n-1, sign is True (pos) / False (neg)
    
    def lit_to_tuple(lit):
        if lit > 0:
            return (lit - 1, True)
        else:
            return (-lit - 1, False)

    # Convert input clauses to list of sets of tuples
    cnf = []
    for c in clauses:
        cl_set = set()
        for lit in c:
            cl_set.add(lit_to_tuple(lit))
        # Handle tautologies: if (v, True) and (v, False) are in clause, it's always satisfied
        # Check for tautology
        is_taut = False
        for (v, s) in cl_set:
            if (v, not s) in cl_set:
                is_taut = True
                break
        if not is_taut:
            cnf.append(cl_set)
    
    m = len(cnf)
    
    def is_satisfied(assign):
        # assign is list of booleans or None
        # Check against original clauses for final verification
        for c in clauses:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assign[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assign):
        count = 0
        for c in clauses:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assign[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # DPLL Implementation
    # State: list of clauses (sets), assignment dict {var: bool}
    
    def unit_propagate(clauses, assignment):
        """
        Apply unit propagation.
        Returns (new_clauses, new_assignment, contradiction)
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for cl in clauses:
                # Filter assigned literals
                new_cl = set()
                unit_lit = None
                for lit in cl:
                    v, s = lit
                    if v in assignment:
                        if assignment[v] == s:
                            # Clause satisfied, discard
                            new_cl = None
                            break
                        else:
                            # Literal false, remove from clause
                            continue
                    else:
                        new_cl.add(lit)
                
                if new_cl is None:
                    continue # Satisfied
                
                if len(new_cl) == 0:
                    return None, assignment, True # Contradiction
                
                if len(new_cl) == 1:
                    # Unit clause
                    lit = next(iter(new_cl))
                    v, s = lit
                    if v in assignment:
                        if assignment[v] != s:
                            return None, assignment, True
                    else:
                        assignment[v] = s
                        changed = True
                        # Don't add unit clause to new_clauses, it's handled
                        continue
                
                new_clauses.append(new_cl)
        
        return new_clauses, assignment, False

    def pure_literal_eliminate(clauses, assignment):
        """
        Apply pure literal elimination.
        Returns (new_clauses, new_assignment, changed)
        """
        # Count occurrences of positive and negative literals
        pos_count = [0] * n
        neg_count = [0] * n
        
        for cl in clauses:
            seen_vars = set()
            for lit in cl:
                v, s = lit
                if v in assignment:
                    continue
                if v not in seen_vars:
                    seen_vars.add(v)
                    if s:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
        
        changed = False
        for v in range(n):
            if v in assignment:
                continue
            if pos_count[v] == 0 and neg_count[v] > 0:
                assignment[v] = False
                changed = True
            elif neg_count[v] == 0 and pos_count[v] > 0:
                assignment[v] = True
                changed = True
        
        if not changed:
            return clauses, assignment, False
        
        # Filter clauses again based on new assignments
        new_clauses = []
        for cl in clauses:
            new_cl = set()
            sat = False
            for lit in cl:
                v, s = lit
                if v in assignment:
                    if assignment[v] == s:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_cl.add(lit)
            if not sat:
                if len(new_cl) > 0:
                    new_clauses.append(new_cl)
                else:
                    # Empty clause -> Contradiction
                    return None, assignment, True
                    
        return new_clauses, assignment, changed

    def select_variable(clauses, assignment):
        """
        Heuristic to select the next variable to branch on.
        Uses Weighted Occurrence (VSIDS-like simplified).
        """
        if time.time() > deadline - 5:
            return -1 # Time out signal
        
        var_score = [0] * n
        for cl in clauses:
            for lit in cl:
                v, s = lit
                if v not in assignment:
                    # Weight by clause size: smaller clauses are more critical
                    weight = 1.0 / len(cl)
                    var_score[v] += weight
        
        # Find unassigned variable with highest score
        best_var = -1
        best_score = -1
        for v in range(n):
            if v not in assignment:
                if var_score[v] > best_score:
                    best_score = var_score[v]
                    best_var = v
        
        return best_var

    def dpll(clauses, assignment, depth=0):
        """
        Recursive DPLL.
        Returns assignment if solved, None if failed.
        """
        if time.time() > deadline - 5:
            return None

        # 1. Check for empty clause (contradiction)
        for cl in clauses:
            if len(cl) == 0:
                return None

        # 2. Unit Propagation
        clauses, assignment, contradiction = unit_propagate(clauses, assignment)
        if contradiction:
            return None
        
        # 3. Check for empty clause again after UP
        for cl in clauses:
            if len(cl) == 0:
                return None

        # 4. Pure Literal Elimination
        clauses, assignment, pple_changed = pure_literal_eliminate(clauses, assignment)
        if clauses is None:
            return None
            
        # Re-check for empty clause after PLE
        for cl in clauses:
            if len(cl) == 0:
                return None

        # 5. Check if all variables assigned
        if len(assignment) == n:
            # Verify solution
            # Create a list from dict
            full_assign = [assignment[i] for i in range(n)]
            if is_satisfied(full_assign):
                return full_assign
            else:
                return None

        # 6. Select variable
        v = select_variable(clauses, assignment)
        if v == -1:
            return None

        # Branch: True first
        assignment[v] = True
        # Filter clauses for branching
        clauses_true = []
        for cl in clauses:
            new_cl = set(cl)
            new_cl.discard((v, False)) # Remove neg literal
            if (v, True) not in new_cl:
                clauses_true.append(new_cl)
            # If (v, True) was in clause, it's satisfied, so we don't need to keep it? 
            # Wait, standard DPLL:
            # If clause contains (v, True), it is satisfied by assignment v=True.
            # So we can discard it.
            # If clause contains (v, False), it becomes false, so remove that literal.
            
        # Let's re-implement the branching filter correctly to be safe
        clauses_true = []
        for cl in clauses:
            has_pos = (v, True) in cl
            has_neg = (v, False) in cl
            if has_pos:
                continue # Satisfied
            new_cl = set(cl)
            if has_neg:
                new_cl.discard((v, False))
            clauses_true.append(new_cl)
        
        res = dpll(clauses_true, assignment, depth + 1)
        if res is not None:
            return res
        
        # Branch: False
        assignment[v] = False
        clauses_false = []
        for cl in clauses:
            has_pos = (v, True) in cl
            has_neg = (v, False) in cl
            if has_neg:
                continue # Satisfied
            new_cl = set(cl)
            if has_pos:
                new_cl.discard((v, True))
            clauses_false.append(new_cl)
        
        res = dpll(clauses_false, assignment, depth + 1)
        if res is not None:
            return res
            
        return None

    # --- Phase 1: Try DPLL ---
    # Set recursion limit higher just in case, though 50 vars is small
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    dpll_start_time = time.time()
    initial_assignment = {}
    
    # We give DPLL a generous chunk of time, say 60s. If it doesn't solve, we switch to WalkSAT.
    # For 50 vars, it should be much faster.
    dpll_deadline = dpll_start_time + 60.0
    # Override global deadline temporarily for DPLL logic? No, dpll checks global deadline.
    # We want to stop DPLL earlier to allow WalkSAT time.
    # Let's modify dpll to take a deadline? Or just let it run and check time in main loop?
    # The dpll function checks `deadline`. If we want to stop it at 60s, we need a separate deadline.
    
    # Let's create a wrapper that respects a local deadline for DPLL
    local_deadline = dpll_start_time + 60.0
    original_deadline = deadline
    deadline = local_deadline # Hack: update global variable used by dpll
    
    solution = dpll(cnf, initial_assignment)
    
    # Restore deadline
    deadline = original_deadline
    sys.setrecursionlimit(old_limit)
    
    if solution:
        return solution

    # --- Phase 2: WalkSAT Local Search ---
    # If DPLL failed or timed out, use WalkSAT.
    # We can use the partial assignment from DPLL if it's useful, but it's in a local scope.
    # Let's just restart with a good initialization.
    
    # Build indices for WalkSAT
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi, sign = lit
            var_to_clauses[vi].append(ci)
            positions.append((vi, sign))
        clause_lit_pos.append(positions)
        
    m = len(cnf)
    
    def compute_clause_sat_counts(assignment):
        sat_counts = [0] * m
        for ci in range(m):
            sc = 0
            for vi, sign in clause_lit_pos[ci]:
                if assignment[vi] == sign:
                    sc += 1
            sat_counts[ci] = sc
        return sat_counts

    def smooth_clause_score(sat_count, clause_len):
        alpha = 3.0
        return math.exp(-alpha * sat_count / clause_len)

    def walksat(assign, max_steps, p_bias, seed):
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        sat_counts = compute_clause_sat_counts(cur)
        cur_unsat = sum(1 for sc in sat_counts if sc == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        clause_weights = [smooth_clause_score(sc, len(clause_lit_pos[ci])) for ci, sc in enumerate(sat_counts)]
        clause_select_count = [0] * m
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            unsat_clauses = [ci for ci in range(m) if sat_counts[ci] == 0]
            if not unsat_clauses:
                return True, cur, 0
            
            for uci in unsat_clauses:
                clause_select_count[uci] += 1
                hist_factor = 1.0 + 0.1 * min(clause_select_count[uci], 20)
                clause_weights[uci] = smooth_clause_score(0, len(clause_lit_pos[uci])) * hist_factor
            
            total_weight = sum(clause_weights[ci] for ci in unsat_clauses)
            if total_weight <= 0:
                ci = rng.choice(unsat_clauses)
            else:
                r = rng.random() * total_weight
                cum = 0
                ci = unsat_clauses[-1]
                for uci in unsat_clauses:
                    cum += clause_weights[uci]
                    if r <= cum:
                        ci = uci
                        break
            
            clause = cnf[ci]
            weights = []
            for lit in clause:
                vi, sign = lit
                w = 1.0 / (1 + len(var_to_clauses[vi]))
                if cur[vi] == sign:
                    w *= 0.3
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
            vi, sign = literal
            
            if rng.random() < p_bias:
                cur[vi] = sign
            else:
                cur[vi] = not cur[vi]
            
            delta_unsat = 0
            for aci in var_to_clauses[vi]:
                old_sc = sat_counts[aci]
                new_sc = 0
                for vi2, sign2 in clause_lit_pos[aci]:
                    if cur[vi2] == sign2:
                        new_sc += 1
                sat_counts[aci] = new_sc
                if old_sc == 0 and new_sc > 0:
                    delta_unsat -= 1
                elif old_sc > 0 and new_sc == 0:
                    delta_unsat += 1
                clause_weights[aci] = smooth_clause_score(new_sc, len(clause_lit_pos[aci]))
            
            cur_unsat += delta_unsat
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
        
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    # Multi-restart WalkSAT
    best_assign = [random.Random(0).random() < 0.5 for _ in range(n)]
    best_unsat = count_unsat(best_assign)
    
    if is_satisfied(best_assign):
        return best_assign

    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    base_steps = 10000
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(base_steps + attempt * 2000, 100000)
        
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

    if is_satisfied(best_assign):
        return best_assign
        
    # Last resort random flips
    rng = random.Random(0)
    for _ in range(2000):
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

