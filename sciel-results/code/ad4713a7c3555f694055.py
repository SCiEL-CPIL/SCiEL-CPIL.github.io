"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 330s wall-clock limit, use 325s for safety margin.
    deadline = time.time() + 325.0
    
    # Convert clauses to 0-based (var_idx, polarity) where polarity: 1=pos, -1=neg
    clauses_v = [[(abs(l)-1, 1 if l > 0 else -1) for l in c] for c in clauses]
    m = len(clauses_v)
    
    # Build index: for each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append((ci, pol))
    
    # ============================================================
    # PHASE 1: DPLL with unit propagation, pure literal elimination,
    # watched literals, and VSIDS-style variable activity
    # ============================================================
    
    def dpll_solver(deadline):
        """DPLL solver. Returns assignment list of bools or None."""
        
        # Assignment: -1 unassigned, 0 false, 1 true
        assign = [-1] * n
        # For each clause, track unassigned literal count and which literals are false
        clause_unassigned = [len(c) for c in clauses_v]
        clause_false = [0] * m
        # Active clauses: those not yet satisfied
        active = [True] * m
        
        # Unit propagation queue
        unit_queue = deque()
        
        # Variable activity (VSIDS-like)
        var_activity = [0.0] * n
        activity_decay = 0.95
        activity_timer = 0
        max_activity = 1e100
        
        def propagate():
            """Unit propagation. Returns False on contradiction."""
            while unit_queue:
                var, val = unit_queue.popleft()
                val_int = 1 if val else 0
                if assign[var] != -1:
                    if assign[var] != val_int:
                        return False
                    continue
                assign[var] = val_int
                var_activity[var] += 1.0
                # Update clauses containing this variable
                for ci, pol in var_to_clauses[var]:
                    if not active[ci]:
                        continue
                    # Does this variable satisfy the clause?
                    if pol == 1 and val_int == 1:
                        active[ci] = False
                    elif pol == -1 and val_int == 0:
                        active[ci] = False
                    else:
                        # Variable is false in this clause
                        clause_unassigned[ci] -= 1
                        clause_false[ci] += 1
                        if clause_unassigned[ci] == 0:
                            return False  # Contradiction
                        elif clause_unassigned[ci] == 1:
                            # Find the unassigned literal
                            for v, p in clauses_v[ci]:
                                if assign[v] == -1:
                                    unit_queue.append((v, p == 1))
                                    break
            return True
        
        def pure_literal_elim():
            """Pure literal elimination. Returns False on contradiction."""
            changed = True
            while changed:
                changed = False
                for var in range(n):
                    if assign[var] != -1:
                        continue
                    pos_seen = False
                    neg_seen = False
                    for ci, pol in var_to_clauses[var]:
                        if not active[ci]:
                            continue
                        if pol == 1:
                            pos_seen = True
                        else:
                            neg_seen = True
                        if pos_seen and neg_seen:
                            break
                    if not pos_seen and not neg_seen:
                        continue
                    if pos_seen and not neg_seen:
                        assign[var] = 1
                        var_activity[var] += 1.0
                        changed = True
                        for ci, pol in var_to_clauses[var]:
                            if not active[ci]:
                                continue
                            if pol == 1:
                                active[ci] = False
                            else:
                                clause_unassigned[ci] -= 1
                                clause_false[ci] += 1
                                if clause_unassigned[ci] == 0:
                                    return False
                                elif clause_unassigned[ci] == 1:
                                    for v, p in clauses_v[ci]:
                                        if assign[v] == -1:
                                            unit_queue.append((v, p == 1))
                                            break
                    elif neg_seen and not pos_seen:
                        assign[var] = 0
                        var_activity[var] += 1.0
                        changed = True
                        for ci, pol in var_to_clauses[var]:
                            if not active[ci]:
                                continue
                            if pol == -1:
                                active[ci] = False
                            else:
                                clause_unassigned[ci] -= 1
                                clause_false[ci] += 1
                                if clause_unassigned[ci] == 0:
                                    return False
                                elif clause_unassigned[ci] == 1:
                                    for v, p in clauses_v[ci]:
                                        if assign[v] == -1:
                                            unit_queue.append((v, p == 1))
                                            break
            return True
        
        # Initial processing
        for ci, c in enumerate(clauses_v):
            if len(c) == 1:
                v, p = c[0]
                unit_queue.append((v, p == 1))
        
        if not pure_literal_elim():
            return None
        if not propagate():
            return None
        
        # Check if solved
        if not any(active):
            for i in range(n):
                if assign[i] == -1:
                    assign[i] = 0
            return [bool(x) for x in assign]
        
        # VSIDS variable selection with activity decay
        def select_variable():
            """Select variable using VSIDS heuristic."""
            nonlocal activity_decay, activity_timer, max_activity
            
            # Decay activities periodically
            activity_timer += 1
            if activity_timer > 1000:
                for i in range(n):
                    var_activity[i] *= activity_decay
                max_activity = max(var_activity) if any(var_activity) else 1.0
                if max_activity > 1e100:
                    scale = 1e100 / max_activity
                    for i in range(n):
                        var_activity[i] *= scale
                    max_activity = 1e100
                activity_timer = 0
            
            best_var = -1
            best_score = -1.0
            for var in range(n):
                if assign[var] != -1:
                    continue
                # Score: activity + bonus for being in short clauses
                score = var_activity[var]
                # Bonus for variables in clauses with few unassigned literals
                min_unassign = 999
                for ci, pol in var_to_clauses[var]:
                    if active[ci] and clause_unassigned[ci] < min_unassign:
                        min_unassign = clause_unassigned[ci]
                if min_unassign < 999:
                    score += 10.0 / (min_unassign + 1)
                if score > best_score:
                    best_score = score
                    best_var = var
            return best_var
        
        # Iterative DPLL with backtracking
        # Use a stack-based approach to avoid deep recursion
        # Trail for backtracking
        trail = []  # list of (var, old_value)
        
        def branch_and_search(deadline):
            """Iterative DPLL with backtracking. Returns assignment or None."""
            backtrack_stack = []
            
            while True:
                if time.time() > deadline:
                    return None
                
                # Check if solved
                if not any(active):
                    for i in range(n):
                        if assign[i] == -1:
                            assign[i] = 0
                    return [bool(x) for x in assign]
                
                # Try pure literal elimination
                if not pure_literal_elim():
                    # Backtrack
                    if backtrack_stack:
                        level, decisions = backtrack_stack.pop()
                        for var, old_val in reversed(decisions):
                            assign[var] = old_val
                        continue
                    return None
                
                if not propagate():
                    if backtrack_stack:
                        level, decisions = backtrack_stack.pop()
                        for var, old_val in reversed(decisions):
                            assign[var] = old_val
                        continue
                    return None
                
                # Check if solved after propagation
                if not any(active):
                    for i in range(n):
                        if assign[i] == -1:
                            assign[i] = 0
                    return [bool(x) for x in assign]
                
                # Select variable to branch on
                var = select_variable()
                if var == -1:
                    return None
                
                # Save state for backtracking
                decisions = [(var, assign[var])]
                assign[var] = 1  # Try true first
                var_activity[var] += 1.0
                for ci, pol in var_to_clauses[var]:
                    if not active[ci]:
                        continue
                    if pol == 1:
                        active[ci] = False
                    else:
                        clause_unassigned[ci] -= 1
                        clause_false[ci] += 1
                        if clause_unassigned[ci] == 0:
                            # Contradiction, try other branch
                            assign[var] = 0
                            for ci2, pol2 in var_to_clauses[var]:
                                if not active[ci2]:
                                    continue
                                if pol2 == -1:
                                    active[ci2] = False
                                else:
                                    clause_unassigned[ci2] -= 1
                                    clause_false[ci2] += 1
                                    if clause_unassigned[ci2] == 0:
                                        if backtrack_stack:
                                            level2, dec2 = backtrack_stack.pop()
                                            for vv, ov in reversed(dec2):
                                                assign[vv] = ov
                                            continue
                                        return None
                                    elif clause_unassigned[ci2] == 1:
                                        for v2, p2 in clauses_v[ci2]:
                                            if assign[v2] == -1:
                                                unit_queue.append((v2, p2 == 1))
                                                break
                            if backtrack_stack:
                                level2, dec2 = backtrack_stack.pop()
                                for vv, ov in reversed(dec2):
                                    assign[vv] = ov
                                continue
                            return None
                        elif clause_unassigned[ci] == 1:
                            for v2, p2 in clauses_v[ci]:
                                if assign[v2] == -1:
                                    unit_queue.append((v2, p2 == 1))
                                    break
                # Push for backtracking (will pop if we need to undo)
                backtrack_stack.append((0, decisions))
                # Continue loop to propagate and check
        
        result = branch_and_search(deadline)
        return result
    
    # Try DPLL first
    dpll_result = dpll_solver(deadline)
    if dpll_result is not None:
        # Verify
        all_sat = True
        for c in clauses_v:
            sat = False
            for var, pol in c:
                if dpll_result[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result
    
    # ============================================================
    # PHASE 2: Optimized WalkSAT with incremental evaluation
    # ============================================================
    
    def count_unsat_fast(assignment):
        cnt = 0
        for c in clauses_v:
            sat = False
            for var, pol in c:
                if assignment[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # Track best assignment found so far
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts with increasing step budgets
    max_restarts = 500
    base_steps = 10000
    max_steps = 100000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize assignment randomly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        unsat_count = count_unsat_fast(assignment)
        
        # Update best if better
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            if unsat_count == 0:
                return assignment
        
        # Determine step budget for this restart
        steps = min(base_steps + restart * 200, max_steps)
        
        # Track unsatisfied clause indices for quick lookup
        unsat_set = set()
        for ci, c in enumerate(clauses_v):
            sat = False
            for var, pol in c:
                if assignment[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                unsat_set.add(ci)
        
        # WalkSAT loop
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            if not unsat_set:
                return assignment
            
            # Update best if this assignment is better
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses_v[ci]
            
            # Score each literal by how many clauses would be satisfied if flipped
            best_var = -1
            best_score = -float('inf')
            best_flip = False
            
            for var, pol in clause:
                new_val = not assignment[var]
                
                # Count net change in satisfied clauses
                score = 0
                for cj, cpol in var_to_clauses[var]:
                    c_j = clauses_v[cj]
                    # Check current satisfaction
                    cur_sat = False
                    for v, p in c_j:
                        if assignment[v] == (p == 1):
                            cur_sat = True
                            break
                    
                    # Check new satisfaction
                    new_sat = False
                    for v, p in c_j:
                        if v == var:
                            if new_val == (p == 1):
                                new_sat = True
                                break
                        else:
                            if assignment[v] == (p == 1):
                                new_sat = True
                                break
                    
                    if cur_sat and not new_sat:
                        score -= 1
                    elif not cur_sat and new_sat:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip = new_val
            
            # With some probability, pick a random literal (exploration)
            if rng.random() < 0.25:
                var, pol = rng.choice(clause)
                new_val = not assignment[var]
            else:
                var = best_var
                new_val = best_flip
            
            if var < 0:
                continue
            
            old_val = assignment[var]
            assignment[var] = new_val
            
            # Update unsat_set incrementally
            for cj, cpol in var_to_clauses[var]:
                c_j = clauses_v[cj]
                # Check if clause was satisfied before
                was_sat = False
                for v, p in c_j:
                    if v == var:
                        if old_val == (p == 1):
                            was_sat = True
                            break
                    else:
                        if assignment[v] == (p == 1):
                            was_sat = True
                            break
                
                # Check if clause is satisfied after
                is_sat_now = False
                for v, p in c_j:
                    if v == var:
                        if new_val == (p == 1):
                            is_sat_now = True
                            break
                    else:
                        if assignment[v] == (p == 1):
                            is_sat_now = True
                            break
                
                if was_sat and not is_sat_now:
                    unsat_set.add(cj)
                elif not was_sat and is_sat_now:
                    unsat_set.discard(cj)
        
        # After steps exhausted, check if solved
        if not unsat_set:
            return assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [False] * n
# EVOLVE-BLOCK-END
