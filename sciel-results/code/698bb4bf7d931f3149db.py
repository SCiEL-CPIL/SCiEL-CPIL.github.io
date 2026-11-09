"""Improved solver for planted 3-SAT with weighted Max-SAT local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque, defaultdict


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 330s wall-clock limit, use 325s for safety margin.
    deadline = time.time() + 325.0
    
    # Convert clauses to 0-based signed literals
    clauses_v = [[(abs(l)-1, 1 if l > 0 else -1) for l in c] for c in clauses]
    m = len(clauses_v)
    
    # Build index: for each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append((ci, pol))
    
    # Track satisfaction count per clause: how many literals are true
    # clause_sat_count[ci] = number of true literals in clause ci under current assignment
    
    # ============================================================
    # PHASE 1: DPLL with unit propagation and pure literal elimination
    # ============================================================
    
    def dpll_solve(deadline):
        """DPLL solver with unit propagation and pure literal elimination.
        Returns assignment list or None if not solved within deadline."""
        
        # assignment[var] = True/False/None (unassigned)
        assignment = [None] * n
        # Track which clauses are still active (not yet satisfied or falsified)
        active_clauses = set(range(m))
        # For each clause, track which literals are still unassigned
        clause_unassigned = [list(c) for c in clauses_v]
        
        # Unit propagation queue: list of (var, value) to assign
        unit_queue = deque()
        
        def propagate_units():
            """Process unit clauses. Returns False if contradiction."""
            while unit_queue:
                var, val = unit_queue.popleft()
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return False
                    continue
                assignment[var] = val
                for ci, pol in var_to_clauses[var]:
                    if ci not in active_clauses:
                        continue
                    # Check if this clause is already satisfied
                    satisfied = False
                    for v, p in clause_unassigned[ci]:
                        if assignment[v] is not None and assignment[v] == (p == 1):
                            satisfied = True
                            break
                    if satisfied:
                        active_clauses.discard(ci)
                        continue
                    # Check if this clause is falsified
                    all_false = True
                    new_unassigned = []
                    for v, p in clause_unassigned[ci]:
                        if assignment[v] is None:
                            all_false = False
                            new_unassigned.append((v, p))
                        elif assignment[v] != (p == 1):
                            pass
                        else:
                            all_false = False
                            break
                    if all_false:
                        return False
                    clause_unassigned[ci] = new_unassigned
                    if len(new_unassigned) == 1:
                        unit_queue.append((new_unassigned[0][0], new_unassigned[0][1] == 1))
            return True
        
        def pure_literal_elimination():
            """Assign pure literals. Returns number of assignments made, or -1 on contradiction."""
            count = 0
            for var in range(n):
                if assignment[var] is not None:
                    continue
                pols = set()
                for ci, pol in var_to_clauses[var]:
                    if ci in active_clauses:
                        pols.add(pol)
                if len(pols) == 1:
                    pol = pols.pop()
                    val = (pol == 1)
                    if assignment[var] is None:
                        assignment[var] = val
                        count += 1
                        for ci, p in var_to_clauses[var]:
                            if ci not in active_clauses:
                                continue
                            satisfied = False
                            for v, pp in clause_unassigned[ci]:
                                if assignment[v] is not None and assignment[v] == (pp == 1):
                                    satisfied = True
                                    break
                            if satisfied:
                                active_clauses.discard(ci)
                                continue
                            new_unassigned = [(v, p) for v, p in clause_unassigned[ci] 
                                              if assignment[v] is None]
                            clause_unassigned[ci] = new_unassigned
                            if not new_unassigned:
                                return -1
                            if len(new_unassigned) == 1:
                                unit_queue.append((new_unassigned[0][0], new_unassigned[0][1] == 1))
            return count
        
        # Initial pure literal elimination
        if pure_literal_elimination() == -1:
            return None
        if not propagate_units():
            return None
        
        if not active_clauses:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
        
        # Recursive DPLL with restarts
        def dpll_recursive(assignment, active_clauses, clause_unassigned, deadline):
            if time.time() > deadline:
                return None
            
            # Try pure literal elimination
            pl_count = 0
            for var in range(n):
                if assignment[var] is not None:
                    continue
                pols = set()
                for ci, pol in var_to_clauses[var]:
                    if ci in active_clauses:
                        pols.add(pol)
                if len(pols) == 1:
                    pol = pols.pop()
                    val = (pol == 1)
                    assignment[var] = val
                    pl_count += 1
                    for ci, p in var_to_clauses[var]:
                        if ci not in active_clauses:
                            continue
                        satisfied = False
                        for v, pp in clause_unassigned[ci]:
                            if assignment[v] is not None and assignment[v] == (pp == 1):
                                satisfied = True
                                break
                        if satisfied:
                            active_clauses.discard(ci)
                            continue
                        new_unassigned = [(v, p) for v, p in clause_unassigned[ci] 
                                          if assignment[v] is None]
                        clause_unassigned[ci] = new_unassigned
                        if not new_unassigned:
                            return None
            if pl_count > 0:
                for ci in list(active_clauses):
                    if len(clause_unassigned[ci]) == 1:
                        v, p = clause_unassigned[ci][0]
                        val = (p == 1)
                        if assignment[v] is not None:
                            if assignment[v] != val:
                                return None
                        else:
                            assignment[v] = val
                            for cj, pp in var_to_clauses[v]:
                                if cj not in active_clauses:
                                    continue
                                satisfied = False
                                for vv, ppp in clause_unassigned[cj]:
                                    if assignment[vv] is not None and assignment[vv] == (ppp == 1):
                                        satisfied = True
                                        break
                                if satisfied:
                                    active_clauses.discard(cj)
                                    continue
                                new_un = [(vv, ppp) for vv, ppp in clause_unassigned[cj] 
                                          if assignment[vv] is None]
                                clause_unassigned[cj] = new_un
                                if not new_un:
                                    return None
                if not active_clauses:
                    for i in range(n):
                        if assignment[i] is None:
                            assignment[i] = False
                    return assignment
                if time.time() > deadline:
                    return None
            
            # Pick a variable to branch on using VSIDS-like heuristic
            # Count occurrences in active clauses
            var_count = [0] * n
            for ci in active_clauses:
                for v, p in clause_unassigned[ci]:
                    if assignment[v] is None:
                        var_count[v] += 1
            
            branch_var = -1
            max_count = -1
            for v in range(n):
                if assignment[v] is None and var_count[v] > max_count:
                    max_count = var_count[v]
                    branch_var = v
            
            if branch_var == -1:
                return None
            
            # Try True
            assignment_backup = assignment[:]
            active_backup = set(active_clauses)
            clause_unassigned_backup = [list(c) for c in clause_unassigned]
            
            assignment[branch_var] = True
            contradiction = False
            for ci, pol in var_to_clauses[branch_var]:
                if ci not in active_clauses:
                    continue
                satisfied = False
                for v, p in clause_unassigned[ci]:
                    if assignment[v] is not None and assignment[v] == (p == 1):
                        satisfied = True
                        break
                if satisfied:
                    active_clauses.discard(ci)
                    continue
                new_unassigned = [(v, p) for v, p in clause_unassigned[ci] 
                                  if assignment[v] is None]
                clause_unassigned[ci] = new_unassigned
                if not new_unassigned:
                    contradiction = True
                    break
            if not contradiction:
                if not active_clauses:
                    for i in range(n):
                        if assignment[i] is None:
                            assignment[i] = False
                    return assignment
                result = dpll_recursive(assignment, active_clauses, clause_unassigned, deadline)
                if result is not None:
                    return result
                assignment = assignment_backup
                active_clauses = active_backup
                clause_unassigned = clause_unassigned_backup
            
            # Try False
            assignment[branch_var] = False
            for ci, pol in var_to_clauses[branch_var]:
                if ci not in active_clauses:
                    continue
                satisfied = False
                for v, p in clause_unassigned[ci]:
                    if assignment[v] is not None and assignment[v] == (p == 1):
                        satisfied = True
                        break
                if satisfied:
                    active_clauses.discard(ci)
                    continue
                new_unassigned = [(v, p) for v, p in clause_unassigned[ci] 
                                  if assignment[v] is None]
                clause_unassigned[ci] = new_unassigned
                if not new_unassigned:
                    return None
            if not active_clauses:
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = False
                return assignment
            result = dpll_recursive(assignment, active_clauses, clause_unassigned, deadline)
            if result is not None:
                return result
            return None
        
        dpll_deadline = time.time() + 20.0
        result = dpll_recursive(assignment, active_clauses, clause_unassigned, dpll_deadline)
        return result
    
    # Try DPLL first
    dpll_result = dpll_solve(deadline)
    if dpll_result is not None:
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
    # PHASE 2: Weighted Max-SAT local search with dynamic penalties
    # ============================================================
    
    def is_satisfied(assignment, clause):
        for var, pol in clause:
            if assignment[var] == (pol == 1):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses_v:
            if not is_satisfied(assignment, c):
                cnt += 1
        return cnt
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(42)
    
    # Weighted local search: each clause gets a penalty weight that increases
    # when it remains unsatisfied, guiding search toward hard clauses
    
    max_restarts = 2000
    base_steps = 5000
    max_steps = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize assignment
        if restart == 0:
            # Start with all False, then do a quick greedy pass
            assignment = [False] * n
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        unsat_count = count_unsat(assignment)
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            if unsat_count == 0:
                return assignment
        
        # Determine step budget
        steps = min(base_steps + restart * 50, max_steps)
        
        # Track unsatisfied clause indices
        unsat_set = set()
        for ci, c in enumerate(clauses_v):
            if not is_satisfied(assignment, c):
                unsat_set.add(ci)
        
        # Dynamic penalty weights for unsatisfied clauses
        # Weight increases the longer a clause stays unsatisfied
        clause_weights = [1.0] * m
        # Track how long each clause has been unsatisfied
        clause_unsat_duration = [0] * m
        
        # Variable activity for better branching
        var_activity = [0.0] * n
        
        # WalkSAT loop with weighted clause selection
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            if not unsat_set:
                return assignment
            
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Update clause unsat durations and weights
            for ci in unsat_set:
                clause_unsat_duration[ci] += 1
                # Exponential growth of penalty, capped
                clause_weights[ci] = min(1.0 + clause_unsat_duration[ci] * 0.1, 100.0)
            
            # Pick a random unsatisfied clause, weighted by penalty
            # Use weighted random selection
            if len(unsat_set) > 1:
                # Weighted selection
                unsat_list = list(unsat_set)
                weights = [clause_weights[ci] for ci in unsat_list]
                total_w = sum(weights)
                if total_w == 0:
                    ci = rng.choice(unsat_list)
                else:
                    r = rng.random() * total_w
                    cumulative = 0
                    ci = unsat_list[0]
                    for uci, w in zip(unsat_list, weights):
                        cumulative += w
                        if r <= cumulative:
                            ci = uci
                            break
            else:
                ci = next(iter(unsat_set))
            
            clause = clauses_v[ci]
            
            # Score each literal by weighted net change in satisfied clauses
            best_var = -1
            best_score = -float('inf')
            best_flip = False
            
            for var, pol in clause:
                new_val = not assignment[var]
                
                # Count net weighted change
                score = 0.0
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
                        score -= clause_weights[cj]
                    elif not cur_sat and new_sat:
                        score += clause_weights[cj]
                
                # Add small activity bonus
                score += var_activity[var] * 0.01
                
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip = new_val
            
            # With some probability, pick a random literal (exploration)
            if rng.random() < 0.2:
                var, pol = rng.choice(clause)
                new_val = not assignment[var]
            else:
                var = best_var
                new_val = best_flip
            
            if var < 0:
                continue
            
            old_val = assignment[var]
            assignment[var] = new_val
            
            # Update variable activity
            var_activity[var] = var_activity[var] * 0.99 + 1.0
            
            # Update unsat_set incrementally
            for cj, cpol in var_to_clauses[var]:
                c_j = clauses_v[cj]
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
                    clause_unsat_duration[cj] = 0
                    clause_weights[cj] = 1.0
                elif not was_sat and is_sat_now:
                    unsat_set.discard(cj)
                    clause_unsat_duration[cj] = 0
                    clause_weights[cj] = 1.0
        
        if not unsat_set:
            return assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    return [False] * n
# EVOLVE-BLOCK-END
