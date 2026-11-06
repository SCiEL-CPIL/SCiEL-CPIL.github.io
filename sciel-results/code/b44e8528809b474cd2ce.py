"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def solve(instance):
    """
    Solve a 3-CNF SAT instance using GF(2) linear algebra, DPLL with unit propagation
    and pure literal elimination. If DPLL fails or times out, fallback to WalkSAT.
    Returns a list of booleans representing the satisfying assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 300s for safety
    start_time = time.time()
    deadline = start_time + 300.0
    
    def check_time():
        return time.time() > deadline
    
    def get_var(lit):
        return abs(lit) - 1
    
    def is_satisfied_clause(clause, assign):
        for lit in clause:
            v = get_var(lit)
            if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                return True
        return False
    
    # --- GF(2) Linear Algebra Approach ---
    # For each clause (l1 ∨ l2 ∨ l3), the unsatisfied condition is ¬l1 ∧ ¬l2 ∧ ¬l3
    # Over GF(2), we can extract linear constraints from the polynomial expansion.
    # Key insight: for planted instances, the linear terms may uniquely determine the solution.
    
    def try_gf2_solve(clauses, n, deadline):
        """Try to solve via GF(2) Gaussian elimination on linearized constraints."""
        if time.time() > deadline - 2.0:
            return None
        
        # Build linear system: for each clause, collect linear coefficients
        # Clause (l1 ∨ l2 ∨ l3) is unsatisfied iff all literals are false
        # In GF(2): let x_v = 1 if var v is True, 0 otherwise
        # Literal l: if l>0, value = x_v; if l<0, value = 1 + x_v (mod 2)
        # Clause satisfied iff at least one literal is 1
        # Unsat: product of (1 + lit_value) = 1 for all literals (all lit_values are 0)
        # This gives: ∏(1 + x_{v_i} + (1 if lit_i < 0 else 0)) = 1
        
        # Linear approximation: keep only degree-0 and degree-1 terms
        # For clause with literals l1, l2, l3:
        # Constant term: product of (1 + sign_offset) where sign_offset = 1 if neg, 0 if pos
        # Linear terms: sum over each literal of its coefficient
        
        # Build augmented matrix for GF(2)
        # Each clause gives one linear equation
        num_clauses = len(clauses)
        if num_clauses == 0:
            return [False] * n
        
        # Use bit arrays for efficiency
        # Each row: n+1 bits (n variables + 1 constant)
        rows = []
        
        for clause in clauses:
            if len(clause) != 3:
                continue
            
            row = [0] * (n + 1)  # last element is RHS
            
            # For clause (l1 ∨ l2 ∨ l3), the polynomial for "all false" is:
            # ∏_{i=1}^{3} (1 - lit_i_value) = 1
            # Over GF(2): ∏(1 + lit_i_value) = 1
            # where lit_i_value = x_v if lit>0, (1+x_v) if lit<0
            
            # Expand: constant + linear terms
            # For 3 literals with signs s1, s2, s3 (s_i = 0 if pos, 1 if neg):
            # lit_i_value = x_{v_i} + s_i
            # 1 + lit_i_value = 1 + x_{v_i} + s_i
            
            # Product of three (1 + x_{v_i} + s_i):
            # Constant: (1+s1)(1+s2)(1+s3)
            # Linear in x_{v1}: (1+s2)(1+s3)
            # Linear in x_{v2}: (1+s1)(1+s3)  
            # Linear in x_{v3}: (1+s1)(1+s2)
            
            lits = list(clause)
            signs = [1 if l < 0 else 0 for l in lits]
            vars_ = [abs(l) - 1 for l in lits]
            
            # Check for duplicate variables (degenerate clause)
            if len(set(vars_)) < 3:
                continue
            
            s1, s2, s3 = signs
            v1, v2, v3 = vars_
            
            # Constant term: (1+s1)(1+s2)(1+s3) mod 2
            const = ((1 + s1) * (1 + s2) * (1 + s3)) % 2
            
            # Linear coefficients
            coef1 = ((1 + s2) * (1 + s3)) % 2
            coef2 = ((1 + s1) * (1 + s3)) % 2
            coef3 = ((1 + s1) * (1 + s2)) % 2
            
            # The equation is: const + coef1*x_{v1} + coef2*x_{v2} + coef3*x_{v3} = 1 (mod 2)
            # So: coef1*x_{v1} + coef2*x_{v2} + coef3*x_{v3} = 1 + const (mod 2)
            
            row[v1] = coef1
            row[v2] = coef2
            row[v3] = coef3
            row[n] = (1 + const) % 2
            
            rows.append(row)
        
        if not rows:
            return None
        
        # Gaussian elimination over GF(2)
        m = len(rows)
        rank = 0
        pivot_cols = []
        
        for col in range(n):
            if time.time() > deadline - 2.0:
                return None
            
            # Find pivot
            pivot = -1
            for i in range(rank, m):
                if rows[i][col] == 1:
                    pivot = i
                    break
            
            if pivot == -1:
                continue
            
            # Swap
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            pivot_cols.append(col)
            
            # Eliminate
            for i in range(m):
                if i != rank and rows[i][col] == 1:
                    for j in range(col, n + 1):
                        rows[i][j] ^= rows[rank][j]
            
            rank += 1
        
        # Check for consistency
        for i in range(rank, m):
            # Row should be all zeros if consistent
            all_zero = all(rows[i][j] == 0 for j in range(n))
            if not all_zero and rows[i][n] == 1:
                return None  # Inconsistent
        
        # If we have a unique solution (rank == n), extract it
        if rank == n:
            solution = [0] * n
            for i in range(n):
                solution[pivot_cols[i]] = rows[i][n]
            
            # Convert to booleans and verify
            assignment = [bool(x) for x in solution]
            if all(is_satisfied_clause(c, assignment) for c in clauses):
                return assignment
        
        return None
    
    # --- DPLL Implementation ---
    
    def simplify_clauses(clauses, assign):
        new_clauses = []
        for clause in clauses:
            new_clause = []
            satisfied = False
            for lit in clause:
                v = get_var(lit)
                if v in assign:
                    val = assign[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                else:
                    new_clause.append(lit)
            
            if satisfied:
                continue
            
            if len(new_clause) == 0:
                return None # Contradiction
            
            new_clauses.append(new_clause)
        
        return new_clauses
    
    def unit_propagate(clauses, assign):
        changed = True
        while changed:
            changed = False
            unit_clause = None
            for c in clauses:
                if len(c) == 1:
                    unit_clause = c
                    break
            
            if not unit_clause:
                break
            
            lit = unit_clause[0]
            v = get_var(lit)
            
            if v in assign:
                val = assign[v]
                if (lit > 0 and not val) or (lit < 0 and val):
                    return False, None, None
                clauses = [c for c in clauses if c != unit_clause]
                continue
            
            assign[v] = (lit > 0)
            changed = True
            clauses = simplify_clauses(clauses, assign)
            if clauses is None:
                return False, None, None
            
            if check_time():
                return True, clauses, assign
        
        return True, clauses, assign
    
    def find_pure_literal(clauses):
        if not clauses:
            return None, None
        
        pos_vars = set()
        neg_vars = set()
        
        for clause in clauses:
            for lit in clause:
                v = get_var(lit)
                if lit > 0:
                    pos_vars.add(v)
                else:
                    neg_vars.add(v)
        
        for v in range(n):
            in_pos = v in pos_vars
            in_neg = v in neg_vars
            
            if in_pos and not in_neg:
                return v + 1, v
            if in_neg and not in_pos:
                return -(v + 1), v
                
        return None, None
    
    def select_variable(clauses, assign):
        counts = {}
        for c in clauses:
            for lit in c:
                v = get_var(lit)
                if v not in assign:
                    counts[v] = counts.get(v, 0) + 1
        
        if not counts:
            return None, None
            
        max_count = max(counts.values())
        candidates = [v for v, c in counts.items() if c == max_count]
        v = random.choice(candidates)
        
        for c in clauses:
            for lit in c:
                if get_var(lit) == v:
                    return v, lit
        return None, None
    
    def dpll(clauses, assign):
        if check_time():
            return assign
        
        if not clauses:
            return assign
        
        success, clauses, assign = unit_propagate(clauses, assign)
        if not success:
            return None
        if check_time():
            return assign
            
        if not clauses:
            return assign
        
        lit, v = find_pure_literal(clauses)
        if lit is not None:
            assign[v] = (lit > 0)
            clauses = simplify_clauses(clauses, assign)
            if clauses is None:
                return None
            if check_time():
                return assign
            return dpll(clauses, assign)
        
        v, lit = select_variable(clauses, assign)
        if v is None:
            return None

        assign_copy = assign.copy()
        assign_copy[v] = (lit > 0)
        clauses_1 = simplify_clauses(clauses, assign_copy)
        if clauses_1 is not None:
            result = dpll(clauses_1, assign_copy)
            if result is not None:
                return result
        
        assign_copy = assign.copy()
        assign_copy[v] = not (lit > 0)
        clauses_2 = simplify_clauses(clauses, assign_copy)
        if clauses_2 is not None:
            result = dpll(clauses_2, assign_copy)
            if result is not None:
                return result
        
        return None

    # Step 1: Try GF(2) linear algebra (fast, may solve planted instances directly)
    gf2_result = try_gf2_solve(clauses, n, deadline)
    if gf2_result is not None:
        return gf2_result
    
    # Step 2: Try DPLL
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    result_assign = None
    try:
        current_clauses = simplify_clauses(clauses, {})
        if current_clauses is not None:
            result_assign = dpll(current_clauses, {})
    finally:
        sys.setrecursionlimit(old_limit)

    # Prepare assignment for WalkSAT
    if result_assign is not None and len(result_assign) == n:
        final_assign = [False] * n
        for v, val in result_assign.items():
            if 0 <= v < n:
                final_assign[v] = val
        if all(is_satisfied_clause(c, final_assign) for c in clauses):
            return final_assign

    # Fallback: WalkSAT with efficient delta tracking
    if result_assign is not None:
        assignment = [False] * n
        for v, val in result_assign.items():
            if 0 <= v < n:
                assignment[v] = val
    else:
        assignment = [random.random() < 0.5 for _ in range(n)]
    
    rng = random.Random(42)
    
    # Precompute clause metadata
    num_clauses = len(clauses)
    clause_vars = []  # list of variable indices per clause
    clause_satisfied = [False] * num_clauses
    var_to_clauses = {}
    
    for i, c in enumerate(clauses):
        vars_in_clause = []
        for lit in c:
            v = get_var(lit)
            vars_in_clause.append(v)
            if v not in var_to_clauses:
                var_to_clauses[v] = []
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_clause)
        clause_satisfied[i] = is_satisfied_clause(c, assignment)
    
    unsat_count = sum(1 for s in clause_satisfied if not s)
    unsat_set = set(i for i, s in enumerate(clause_satisfied) if not s)
    
    # Track number of false literals per clause (0 means satisfied)
    clause_false_count = [0] * num_clauses
    for i, c in enumerate(clauses):
        fc = 0
        for lit in c:
            v = get_var(lit)
            if (lit > 0 and not assignment[v]) or (lit < 0 and assignment[v]):
                fc += 1
        clause_false_count[i] = fc
    
    def flip_var(v, assignment, clause_satisfied, clause_false_count, unsat_set, unsat_count):
        """Flip variable v and update all tracked structures. Returns new unsat_count."""
        old_val = assignment[v]
        assignment[v] = not old_val
        
        new_unsat_count = unsat_count
        
        for ci in var_to_clauses.get(v, []):
            c = clauses[ci]
            was_sat = clause_satisfied[ci]
            
            # Update false count for this clause
            for lit in c:
                if get_var(lit) == v:
                    if (lit > 0 and not old_val) or (lit < 0 and old_val):
                        # literal was false before flip
                        clause_false_count[ci] -= 1
                    else:
                        # literal was true before flip
                        clause_false_count[ci] += 1
            
            is_now_sat = clause_false_count[ci] == 0
            clause_satisfied[ci] = is_now_sat
            
            if was_sat and not is_now_sat:
                new_unsat_count += 1
                unsat_set.add(ci)
            elif not was_sat and is_now_sat:
                new_unsat_count -= 1
                unsat_set.discard(ci)
        
        return new_unsat_count
    
    # WalkSAT with restarts
    max_steps_per_restart = 5000
    best_unsat_ever = unsat_count
    best_assignment = assignment.copy()
    
    while time.time() < deadline - 2.0:
        if unsat_count == 0:
            return assignment
        
        # Restart if stuck
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline - 2.0:
            if unsat_count == 0:
                return assignment
            
            if not unsat_set:
                break
            
            idx = rng.choice(list(unsat_set))
            clause = clauses[idx]
            
            best_var = None
            best_delta = float('inf')
            
            for lit in clause:
                v = get_var(lit)
                
                # Compute delta efficiently using clause_false_count
                delta = 0
                for ci in var_to_clauses.get(v, []):
                    c = clauses[ci]
                    cur_sat = clause_satisfied[ci]
                    cur_fc = clause_false_count[ci]
                    
                    # After flipping v, how does this clause change?
                    # Check if v appears in this clause
                    v_in_clause = False
                    lit_was_false = False
                    for l2 in c:
                        if get_var(l2) == v:
                            v_in_clause = True
                            if (l2 > 0 and not assignment[v]) or (l2 < 0 and assignment[v]):
                                lit_was_false = True
                            break
                    
                    if not v_in_clause:
                        continue
                    
                    if cur_sat:
                        # Currently satisfied (fc == 0). After flip, if the literal was true, it becomes false
                        # fc increases by 1, so still satisfied unless fc becomes > 0... wait
                        # If cur_sat, fc == 0. The literal must be true (since fc==0 means all true).
                        # After flip, literal becomes false, so fc becomes 1. Clause becomes unsat.
                        delta += 1
                    else:
                        # Currently unsat (fc > 0). 
                        # If literal was false, after flip it becomes true, fc decreases by 1.
                        # If fc was 1, it becomes 0 (satisfied). Delta -= 1.
                        # If fc > 1, still unsat. Delta unchanged.
                        # If literal was true (but clause unsat means at least one other is false),
                        # after flip it becomes false, fc increases by 1. Delta unchanged (still unsat).
                        if lit_was_false:
                            if cur_fc == 1:
                                delta -= 1
                        # else: literal was true, fc increases, no change in sat status
                    
                    # Note: if v appears multiple times in clause, handle carefully
                    # For 3-CNF, assume no duplicates
                
                noisy_delta = delta + rng.random() * 0.5
                if noisy_delta < best_delta:
                    best_delta = noisy_delta
                    best_var = v
            
            if best_var is not None:
                unsat_count = flip_var(best_var, assignment, clause_satisfied, clause_false_count, unsat_set, unsat_count)
                
                if unsat_count < best_unsat_ever:
                    best_unsat_ever = unsat_count
                    best_assignment = assignment.copy()
            
            steps += 1
        
        # Restart: randomize
        if time.time() < deadline - 2.0:
            # Partial restart: flip some random variables
            for _ in range(min(n // 2, 10)):
                v = rng.randrange(n)
                unsat_count = flip_var(v, assignment, clause_satisfied, clause_false_count, unsat_set, unsat_count)
            
            # If we've been stuck, full restart
            if best_unsat_ever > 0 and rng.random() < 0.3:
                assignment = [random.random() < 0.5 for _ in range(n)]
                for i in range(num_clauses):
                    clause_satisfied[i] = is_satisfied_clause(clauses[i], assignment)
                    fc = 0
                    for lit in clauses[i]:
                        v = get_var(lit)
                        if (lit > 0 and not assignment[v]) or (lit < 0 and assignment[v]):
                            fc += 1
                    clause_false_count[i] = fc
                unsat_set = set(i for i, s in enumerate(clause_satisfied) if not s)
                unsat_count = len(unsat_set)
    
    # Final polish: try to fix remaining unsat clauses
    if unsat_count > 0 and time.time() < deadline - 0.5:
        # Try flipping variables in unsat clauses
        for _ in range(100):
            if time.time() > deadline - 0.5:
                break
            if not unsat_set:
                break
            idx = rng.choice(list(unsat_set))
            clause = clauses[idx]
            for lit in clause:
                v = get_var(lit)
                # Try flipping v
                old_unsat = unsat_count
                unsat_count = flip_var(v, assignment, clause_satisfied, clause_false_count, unsat_set, unsat_count)
                if unsat_count < old_unsat:
                    break
                else:
                    # Revert
                    unsat_count = flip_var(v, assignment, clause_satisfied, clause_false_count, unsat_set, unsat_count)
    
    if unsat_count == 0:
        return assignment
    
    # Return best found
    if best_unsat_ever < unsat_count:
        return best_assignment
    
    return assignment

# EVOLVE-BLOCK-END
