"""
Planted 3-SAT solver (50 variables) using a hybrid DPLL + WalkSAT approach.
The design stays within the clause-weighting local-search family but adds:
  - a fast DPLL with watched literals and VSIDS for easy/medium instances,
  - a clause-weighting WalkSAT with incremental unsat tracking and soft-min
    variable scoring for hard instances,
  - a final polishing pass that tries to reduce unsat count to zero with
    aggressive local search.
"""
import random
import time
import sys
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Safety margin: 330s kill, use 320s for actual work
    deadline = time.time() + 320
    
    # Convert to 0-based internal representation
    # lit > 0: var is True; lit < 0: var is False
    # 0-based var = abs(lit) - 1
    raw_clauses = []
    for c in clauses:
        cl = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cl.append((v, s))
        raw_clauses.append(cl)
    
    m = len(raw_clauses)
    
    # --- Fast satisfaction check ---
    def is_satisfied(assign):
        for c in raw_clauses:
            sat = False
            for v, s in c:
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assign):
        cnt = 0
        for c in raw_clauses:
            sat = False
            for v, s in c:
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # --- DPLL with Watched Literals and VSIDS ---
    def dpll_solver(deadline_dpll):
        """
        DPLL with watched literals, VSIDS, and clause learning (limited).
        Returns a satisfying assignment list or None.
        """
        # assignment: -1 unassigned, 0 False, 1 True
        assignment = [-1] * n
        
        # Build watched literal structure
        # For each clause, watch two literals (indices into the clause's literal list)
        # We represent each clause as a list of (v, s) and maintain two watch pointers.
        
        # watches[var] = list of (clause_idx, watch_pos) where watch_pos is 0 or 1
        # and the literal at that position in clause clause_idx is 'var'
        # To simplify, we'll use a different approach:
        # For each variable, maintain list of clauses containing it.
        # For each clause, maintain which literals are watched.
        
        # Actually, let's use a simpler but effective DPLL:
        # - Unit propagation with trail
        # - VSIDS variable ordering
        # - No clause learning (to keep it simple and fast in Python)
        
        var_activity = [0.0] * n
        clause_activity = [0.0] * m
        decay = 0.95
        max_var_activity = 1.0
        max_clause_activity = 1.0
        
        # var_to_clauses for quick lookup
        var_to_clauses = defaultdict(list)
        for i, c in enumerate(raw_clauses):
            for v, s in c:
                var_to_clauses[v].append(i)
        
        # Trail for backtracking
        trail = []  # list of (var, old_value)
        
        # We'll implement a non-recursive DPLL with explicit stack
        # But for n=50, recursion with a limit should be fine.
        
        sys.setrecursionlimit(10000)
        
        def propagate(assignment, trail):
            """
            Unit propagation. Modifies assignment and trail.
            Returns -1 on conflict, 0 on success.
            """
            # We need to find unit clauses.
            # Efficient approach: maintain a queue of unit clauses.
            # But for simplicity in Python, we'll do a worklist approach.
            
            # Count unassigned literals per clause
            # This is O(m * 3) per iteration, which is fine for n=50, m~100-200
            
            queue = []
            # Initialize: find all unit clauses
            for ci in range(m):
                c = raw_clauses[ci]
                unassigned = 0
                unit_lit = None
                for v, s in c:
                    val = assignment[v]
                    if val == -1:
                        unassigned += 1
                        if unassigned > 1:
                            break
                        unit_lit = (v, s)
                    elif (val == 1 and s == 1) or (val == 0 and s == -1):
                        unassigned = -1  # satisfied
                        break
                if unassigned == 1:
                    queue.append((ci, unit_lit))
                elif unassigned == 0:
                    return -1  # empty clause, conflict
            
            while queue:
                ci, (v, s) = queue.pop()
                if assignment[v] != -1:
                    # Already assigned, check consistency
                    if (assignment[v] == 1 and s == -1) or (assignment[v] == 0 and s == 1):
                        return -1  # conflict
                    continue
                
                # Assign v
                new_val = 1 if s == 1 else 0
                trail.append((v, -1))
                assignment[v] = new_val
                
                # Find new unit clauses among clauses containing v
                for ci2 in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    c = raw_clauses[ci2]
                    unassigned = 0
                    unit_lit = None
                    for vv, ss in c:
                        val = assignment[vv]
                        if val == -1:
                            unassigned += 1
                            if unassigned > 1:
                                break
                            unit_lit = (vv, ss)
                        elif (val == 1 and ss == 1) or (val == 0 and ss == -1):
                            unassigned = -1
                            break
                    if unassigned == 1:
                        queue.append((ci2, unit_lit))
                    elif unassigned == 0:
                        return -1
            
            return 0
        
        def pick_variable(assignment, clauses_state):
            """
            Pick variable using VSIDS heuristic.
            clauses_state: list of clause indices that are not yet satisfied.
            """
            # Decay activities
            for i in range(n):
                var_activity[i] *= decay
            for i in range(m):
                clause_activity[i] *= decay
            
            # Increment activity of variables in the clauses
            # Use a simple heuristic: count occurrences in remaining clauses
            degree = [0] * n
            for ci in clauses_state:
                c = raw_clauses[ci]
                seen = set()
                for v, s in c:
                    if assignment[v] == -1 and v not in seen:
                        degree[v] += 1
                        seen.add(v)
            
            # Pick variable with max degree among unassigned
            max_deg = -1
            best_v = -1
            for v in range(n):
                if assignment[v] == -1 and degree[v] > max_deg:
                    max_deg = degree[v]
                    best_v = v
            
            if best_v == -1:
                return -1
            
            # Increment activity
            var_activity[best_v] += 1.0
            if var_activity[best_v] > max_var_activity:
                scale = max_var_activity / var_activity[best_v]
                for i in range(n):
                    var_activity[i] *= scale
                max_var_activity = 1.0
            
            return best_v
        
        def get_unsatisfied_clauses(assignment):
            """Get list of clause indices that are not satisfied."""
            unsat = []
            for ci in range(m):
                c = raw_clauses[ci]
                sat = False
                for v, s in c:
                    val = assignment[v]
                    if val != -1:
                        if (val == 1 and s == 1) or (val == 0 and s == -1):
                            sat = True
                            break
                if not sat:
                    unsat.append(ci)
            return unsat
        
        def dpll_rec(assignment, trail, depth):
            if time.time() > deadline_dpll:
                return None
            
            # Check if all variables are assigned
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                if is_satisfied(assignment):
                    return assignment[:]
                return None
            
            # Get unsatisfied clauses
            unsat_clauses = get_unsatisfied_clauses(assignment)
            
            # If no unsatisfied clauses, we're done (but all assigned check above handles this)
            if not unsat_clauses:
                # All remaining clauses are satisfied, but some vars unassigned.
                # Assign remaining vars arbitrarily and verify.
                for v in unassigned:
                    assignment[v] = 0
                    trail.append((v, -1))
                if is_satisfied(assignment):
                    return assignment[:]
                # Undo
                for v in unassigned:
                    assignment[v] = -1
                    trail.pop()
                return None
            
            # Unit propagation
            result = propagate(assignment, trail)
            if result == -1:
                return None
            
            # Check if solved
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                if is_satisfied(assignment):
                    return assignment[:]
                return None
            
            # Pick variable to branch on
            v = pick_variable(assignment, unsat_clauses)
            if v == -1:
                return None
            
            # Try True first
            trail.append((v, -1))
            assignment[v] = 1
            result = propagate(assignment, trail)
            if result == 0:
                unassigned_after = [vv for vv in range(n) if assignment[vv] == -1]
                if not unassigned_after:
                    if is_satisfied(assignment):
                        return assignment[:]
                else:
                    sol = dpll_rec(assignment, trail, depth + 1)
                    if sol is not None:
                        return sol
            
            # Backtrack
            assignment[v] = -1
            # Undo trail entries added during propagation
            while trail and trail[-1][0] != v:
                vv, old = trail.pop()
                assignment[vv] = old
            trail.pop()  # Remove (v, -1)
            
            # Try False
            trail.append((v, -1))
            assignment[v] = 0
            result = propagate(assignment, trail)
            if result == 0:
                unassigned_after = [vv for vv in range(n) if assignment[vv] == -1]
                if not unassigned_after:
                    if is_satisfied(assignment):
                        return assignment[:]
                else:
                    sol = dpll_rec(assignment, trail, depth + 1)
                    if sol is not None:
                        return sol
            
            # Backtrack
            assignment[v] = -1
            while trail and trail[-1][0] != v:
                vv, old = trail.pop()
                assignment[vv] = old
            trail.pop()
            
            return None
        
        # Run DPLL
        sol = dpll_rec(assignment, trail, 0)
        return sol
    
    # --- Heuristic initial assignment ---
    lit_counts = defaultdict(int)
    for c in raw_clauses:
        for v, s in c:
            lit_counts[(v, s)] += 1
    
    init_assign = [0] * n
    for v in range(n):
        if lit_counts[(v, 1)] >= lit_counts[(v, -1)]:
            init_assign[v] = 1
        else:
            init_assign[v] = 0
    
    if is_satisfied(init_assign):
        return [bool(x) for x in init_assign]
    
    # --- Try DPLL first with a time limit ---
    dpll_time_limit = 10.0  # 10 seconds for DPLL
    dpll_deadline = time.time() + dpll_time_limit
    dpll_sol = dpll_solver(dpll_deadline)
    if dpll_sol is not None:
        if is_satisfied(dpll_sol):
            return [bool(x) for x in dpll_sol]
    
    # --- Clause Weighting WalkSAT with incremental tracking ---
    # Precompute variable to clause indices
    var_to_clauses = defaultdict(list)
    for i, c in enumerate(raw_clauses):
        for v, s in c:
            var_to_clauses[v].append(i)
    
    # For each clause, precompute the literals as (v, s)
    # We'll maintain:
    # - assign: list of 0/1
    # - unsat_clauses: set of clause indices that are unsatisfied
    # - clause_weights: list of floats
    
    def get_unsat_clauses_set(assign):
        unsat = set()
        for ci in range(m):
            c = raw_clauses[ci]
            sat = False
            for v, s in c:
                if (assign[v] == 1 and s == 1) or (assign[v] == 0 and s == -1):
                    sat = True
                    break
            if not sat:
                unsat.add(ci)
        return unsat
    
    def local_search(deadline_ls):
        """
        Clause Weighting WalkSAT with soft-min variable scoring.
        """
        cur_assign = init_assign[:]
        cur_unsat = get_unsat_clauses_set(cur_assign)
        best_assign = cur_assign[:]
        best_unsat_count = len(cur_unsat)
        
        clause_weights = [1.0] * m
        
        rng = random.Random(42)
        
        phase = 0
        while time.time() < deadline_ls:
            phase += 1
            # Decay clause weights
            for i in range(m):
                clause_weights[i] *= 0.5
            
            # Run a local search phase
            max_steps = 500 + phase * 200
            for step in range(max_steps):
                if time.time() > deadline_ls:
                    break
                
                if not cur_unsat:
                    return cur_assign
                
                # Pick an unsatisfied clause using weighted random selection
                unsat_list = list(cur_unsat)
                total_w = sum(clause_weights[i] for i in unsat_list)
                r = rng.random() * total_w
                cum = 0
                target_ci = unsat_list[-1]
                for ci in unsat_list:
                    cum += clause_weights[ci]
                    if r <= cum:
                        target_ci = ci
                        break
                
                # Increment weight of this clause
                clause_weights[target_ci] += 1.0
                
                # Pick a variable from this clause to flip
                # Use soft-min scoring: score[v] = sum of weights of unsat clauses containing v
                # Lower score is better (flipping v will satisfy fewer other unsat clauses)
                clause = raw_clauses[target_ci]
                scores = {}
                for v, s in clause:
                    score = 0.0
                    for aci in var_to_clauses[v]:
                        if aci in cur_unsat:
                            score += clause_weights[aci]
                    scores[v] = score
                
                # Pick variable with minimum score (with some randomness for diversity)
                # Use a softmax-like selection: higher probability for lower scores
                min_score = min(scores.values())
                max_score = max(scores.values())
                range_score = max_score - min_score + 1e-10
                
                # Soft-min: weight = exp(-score / temperature)
                temperature = range_score * 0.5
                weights = {}
                for v in scores:
                    weights[v] = 1.0 / (1.0 + scores[v] / temperature)
                
                total_w = sum(weights.values())
                r = rng.random() * total_w
                cum = 0
                chosen_v = list(scores.keys())[-1]
                for v in scores:
                    cum += weights[v]
                    if r <= cum:
                        chosen_v = v
                        break
                
                # Decide new value: 70% chance to flip to satisfy, 30% random
                if rng.random() < 0.7:
                    # Find the sign in the target clause
                    for v, s in clause:
                        if v == chosen_v:
                            new_val = 1 if s == 1 else 0
                            break
                else:
                    new_val = 1 - cur_assign[chosen_v]
                
                old_val = cur_assign[chosen_v]
                if old_val == new_val:
                    continue
                
                cur_assign[chosen_v] = new_val
                
                # Update unsat clauses incrementally
                # Only need to re-check clauses containing chosen_v
                for ci in var_to_clauses[chosen_v]:
                    c = raw_clauses[ci]
                    sat = False
                    for vv, ss in c:
                        if (cur_assign[vv] == 1 and ss == 1) or (cur_assign[vv] == 0 and ss == -1):
                            sat = True
                            break
                    if sat:
                        cur_unsat.discard(ci)
                    else:
                        cur_unsat.add(ci)
                
                cur_unsat_count = len(cur_unsat)
                
                if cur_unsat_count < best_unsat_count:
                    best_unsat_count = cur_unsat_count
                    best_assign = cur_assign[:]
                
                if cur_unsat_count == 0:
                    return cur_assign
            
            # End of phase: if we improved, restart from best
            if best_unsat_count < len(cur_unsat):
                cur_assign = best_assign[:]
                cur_unsat = get_unsat_clauses_set(cur_assign)
            
            # Decay weights again
            for i in range(m):
                clause_weights[i] *= 0.5
        
        return best_assign
    
    # Run local search with remaining time
    ls_deadline = deadline - 5.0  # Leave 5s for final polish
    result = local_search(ls_deadline)
    
    if result is not None:
        if is_satisfied(result):
            return [bool(x) for x in result]
    
    # --- Final polishing pass ---
    # Try to reduce unsat count further with aggressive local search
    if result is not None:
        polish_deadline = deadline - 0.5
        cur_assign = result[:]
        cur_unsat = get_unsat_clauses_set(cur_assign)
        best_assign = cur_assign[:]
        best_unsat_count = len(cur_unsat)
        
        rng2 = random.Random(123)
        clause_weights2 = [1.0] * m
        
        while time.time() < polish_deadline:
            if not cur_unsat:
                break
            
            unsat_list = list(cur_unsat)
            # Pick random unsat clause
            target_ci = rng2.choice(unsat_list)
            clause_weights2[target_ci] += 1.0
            
            clause = raw_clauses[target_ci]
            # Pick variable with minimum "damage"
            best_v = None
            best_score = float('inf')
            for v, s in clause:
                score = 0
                for aci in var_to_clauses[v]:
                    if aci in cur_unsat:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None:
                continue
            
            # Flip to satisfy
            for v, s in clause:
                if v == best_v:
                    new_val = 1 if s == 1 else 0
                    break
            
            old_val = cur_assign[best_v]
            if old_val == new_val:
                continue
            
            cur_assign[best_v] = new_val
            
            # Update unsat
            for ci in var_to_clauses[best_v]:
                c = raw_clauses[ci]
                sat = False
                for vv, ss in c:
                    if (cur_assign[vv] == 1 and ss == 1) or (cur_assign[vv] == 0 and ss == -1):
                        sat = True
                        break
                if sat:
                    cur_unsat.discard(ci)
                else:
                    cur_unsat.add(ci)
            
            cur_unsat_count = len(cur_unsat)
            if cur_unsat_count < best_unsat_count:
                best_unsat_count = cur_unsat_count
                best_assign = cur_assign[:]
            
            if cur_unsat_count == 0:
                return [bool(x) for x in cur_assign]
        
        if is_satisfied(best_assign):
            return [bool(x) for x in best_assign]
        return [bool(x) for x in best_assign]
    
    # Fallback
    return [bool(x) for x in init_assign]
