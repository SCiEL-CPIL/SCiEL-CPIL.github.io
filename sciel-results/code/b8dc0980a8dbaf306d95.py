"""Hybrid DPLL + WalkSAT 3-SAT solver for planted instances."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # 10s safety margin before 300s limit

    # Preprocess
    clauses = [list(c) for c in clauses]
    m = len(clauses)

    # Build occurrence index
    occ = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            occ[v].append((ci, sign))

    def dpll_watched():
        """Complete DPLL with watched literals, unit propagation, and pure literal elimination."""
        # Watched literals: for each clause, watch two literals
        # assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        
        # For each clause, store the two watched literals and the clause index
        # watched[ci] = (lit1, lit2) where lit1, lit2 are literals in clause ci
        watched = [None] * m
        for ci in range(m):
            if len(clauses[ci]) >= 2:
                watched[ci] = (clauses[ci][0], clauses[ci][1])
            elif len(clauses[ci]) == 1:
                watched[ci] = (clauses[ci][0], None)
            else:
                return None  # empty clause, UNSAT
        
        # active_clauses tracks which clauses are still active (not satisfied)
        active = [True] * m
        
        # For unit propagation, we need to find unit clauses efficiently
        # We'll use a worklist approach
        
        def get_clause_state(cl):
            """Return (num_unassigned, num_false) for a clause given current assignment."""
            num_unassigned = 0
            num_false = 0
            for lit in cl:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    num_unassigned += 1
                else:
                    expected = 1 if lit > 0 else 0
                    if assignment[v] != expected:
                        num_false += 1
            return num_unassigned, num_false
        
        def unit_propagate():
            """Perform unit propagation using watched literals. Returns False on conflict."""
            changed = True
            while changed:
                changed = False
                # Find unit clauses
                for ci in range(m):
                    if not active[ci]:
                        continue
                    cl = clauses[ci]
                    num_unassigned, num_false = get_clause_state(cl)
                    
                    if num_unassigned == 0:
                        if num_false == len(cl):
                            return False  # conflict: all literals false
                        else:
                            active[ci] = False  # clause is satisfied
                            continue
                    
                    if num_unassigned == 1:
                        # Unit clause: find the unassigned literal
                        for lit in cl:
                            v = abs(lit) - 1
                            if assignment[v] == -1:
                                val = 1 if lit > 0 else 0
                                assignment[v] = val
                                changed = True
                                # Update watched literals for affected clauses
                                neg_lit = -lit
                                for cj in range(m):
                                    if cj == ci or not active[cj]:
                                        continue
                                    w = watched[cj]
                                    if w is None:
                                        continue
                                    # Check if either watched literal is the negation
                                    if w[0] == neg_lit or (w[1] is not None and w[1] == neg_lit):
                                        # Need to find a new watch
                                        new_watch = None
                                        for lit2 in clauses[cj]:
                                            v2 = abs(lit2) - 1
                                            if assignment[v2] == -1:
                                                new_watch = lit2
                                                break
                                        if new_watch is None:
                                            # No unassigned literal to watch
                                            # Check if clause is satisfied
                                            num_un2, num_false2 = get_clause_state(clauses[cj])
                                            if num_false2 == len(clauses[cj]):
                                                return False
                                            else:
                                                active[cj] = False
                                        else:
                                            # Replace the negated watch
                                            if w[0] == neg_lit:
                                                watched[cj] = (new_watch, w[1] if w[1] != neg_lit else None)
                                                if watched[cj][1] is None and w[0] != neg_lit:
                                                    watched[cj] = (w[0], new_watch)
                                            elif w[1] is not None and w[1] == neg_lit:
                                                watched[cj] = (w[0], new_watch)
                                break
            return True
        
        def pure_literal_elimination():
            """Eliminate pure literals. Returns True if any were eliminated."""
            eliminated = False
            for v in range(n):
                if assignment[v] != -1:
                    continue
                # Check if literal v or -v appears
                pos_occ = False
                neg_occ = False
                for (ci, sign) in occ[v]:
                    if not active[ci]:
                        continue
                    if sign > 0:
                        pos_occ = True
                    else:
                        neg_occ = True
                if pos_occ and not neg_occ:
                    assignment[v] = 1
                    eliminated = True
                elif neg_occ and not pos_occ:
                    assignment[v] = 0
                    eliminated = True
            return eliminated
        
        # Initial pure literal elimination
        pure_literal_elimination()
        
        # Initial unit propagation
        if not unit_propagate():
            return None
        
        # Check if satisfied
        for ci in range(m):
            if not active[ci]:
                continue
            cl = clauses[ci]
            for lit in cl:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    break
                expected = 1 if lit > 0 else 0
                if assignment[v] == expected:
                    active[ci] = False
                    break
            else:
                # All assigned, check if satisfied
                num_unassigned, num_false = get_clause_state(cl)
                if num_false == len(cl):
                    return None
                active[ci] = False
        
        # Check if all clauses satisfied
        all_sat = all(not active[ci] for ci in range(m))
        if all_sat:
            # Fill remaining unassigned vars
            for i in range(n):
                if assignment[i] == -1:
                    assignment[i] = random.random() < 0.5
            return [bool(a) for a in assignment]
        
        # DPLL search with watched literals
        def dpll_search(depth=0):
            if time.time() > deadline:
                return None
            
            # Pure literal elimination
            pure_literal_elimination()
            
            # Unit propagation
            if not unit_propagate():
                return None
            
            # Check if satisfied
            for ci in range(m):
                if not active[ci]:
                    continue
                cl = clauses[ci]
                num_unassigned, num_false = get_clause_state(cl)
                if num_unassigned == 0:
                    if num_false == len(cl):
                        return None
                    active[ci] = False
            
            all_sat = all(not active[ci] for ci in range(m))
            if all_sat:
                for i in range(n):
                    if assignment[i] == -1:
                        assignment[i] = random.random() < 0.5
                return [bool(a) for a in assignment]
            
            # Choose a variable to branch on (pick one from an active clause)
            branch_var = -1
            for ci in range(m):
                if not active[ci]:
                    continue
                cl = clauses[ci]
                for lit in cl:
                    v = abs(lit) - 1
                    if assignment[v] == -1:
                        branch_var = v
                        break
                if branch_var != -1:
                    break
            
            if branch_var == -1:
                # All assigned but not all satisfied? Shouldn't happen
                for i in range(n):
                    if assignment[i] == -1:
                        assignment[i] = random.random() < 0.5
                return [bool(a) for a in assignment]
            
            # Try both values
            for val in [1, 0]:
                # Save state
                saved_assignment = assignment[:]
                saved_active = active[:]
                saved_watched = [w[:] if w else None for w in watched]
                
                assignment[branch_var] = val
                
                # Unit propagation
                if unit_propagate():
                    result = dpll_search(depth + 1)
                    if result is not None:
                        return result
                
                # Restore state
                assignment[:] = saved_assignment
                active[:] = saved_active
                watched[:] = saved_watched
            
            return None
        
        result = dpll_search()
        if result is not None:
            return result
        
        return None

    def walk_sat_trial(rng, max_steps, noise_weight=1.0):
        """WalkSAT with adaptive noise. Returns (unsat_count, assignment)."""
        assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial energy
        clause_unsat = [False] * m
        for ci, cl in enumerate(clauses):
            sat = False
            for lit in cl:
                v = abs(lit) - 1
                if assign[v] == (lit > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
        unsat_count = sum(clause_unsat)
        
        if unsat_count == 0:
            return 0, assign

        best_unsat = unsat_count
        best_assign = assign[:]
        
        # Precompute clause literal positions for faster delta computation
        # For each variable, store which clauses it appears in and at what position
        var_clauses = [[] for _ in range(n)]
        for ci, cl in enumerate(clauses):
            for pos, lit in enumerate(cl):
                v = abs(lit) - 1
                var_clauses[v].append((ci, pos, lit))
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return 0, assign

            # Adaptive noise: increase noise when stuck
            # Count how many steps since last improvement
            if step % 100 == 0 and unsat_count >= best_unsat:
                noise_weight = min(3.0, noise_weight * 1.2)
            else:
                noise_weight = max(0.5, noise_weight * 0.98)

            # Pick a random unsatisfied clause
            unsat_clauses = [ci for ci in range(m) if clause_unsat[ci]]
            if not unsat_clauses:
                return 0, assign
            ci = rng.choice(unsat_clauses)
            cl = clauses[ci]
            
            # Choose variable to flip using weight-based heuristic
            # Weight = number of satisfied literals that would be broken - noise
            best_var = -1
            best_weight = float('inf')
            for pos, lit in enumerate(cl):
                v = abs(lit) - 1
                # Count how many other clauses would be broken
                weight = 0
                for (cidx, pos2, lit2) in var_clauses[v]:
                    if cidx == ci:
                        continue
                    # Check if this clause is currently satisfied by this literal
                    if clause_unsat[cidx]:
                        continue
                    # Check if flipping v would break this clause
                    # The clause is satisfied, so at least one literal is true
                    # If this literal is the only true one, flipping breaks it
                    is_only_true = True
                    for lit3 in clauses[cidx]:
                        v3 = abs(lit3) - 1
                        if v3 == v:
                            continue
                        if assign[v3] == (lit3 > 0):
                            is_only_true = False
                            break
                    if is_only_true:
                        weight += 1
                
                weight += rng.random() * noise_weight
                
                if weight < best_weight:
                    best_weight = weight
                    best_var = v
            
            if best_var == -1:
                break
            
            # Flip the variable
            v = best_var
            old_val = assign[v]
            assign[v] = not old_val
            
            # Update clause_unsat for affected clauses
            for (cidx, pos2, lit2) in var_clauses[v]:
                cl2 = clauses[cidx]
                sat = False
                for lit in cl2:
                    v2 = abs(lit) - 1
                    if assign[v2] == (lit > 0):
                        sat = True
                        break
                new_unsat = not sat
                if new_unsat and not clause_unsat[cidx]:
                    unsat_count += 1
                elif not new_unsat and clause_unsat[cidx]:
                    unsat_count -= 1
                clause_unsat[cidx] = new_unsat
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign[:]
        
        return best_unsat, best_assign

    # Strategy: First try DPLL with watched literals, then fall back to WalkSAT
    result = dpll_watched()
    if result is not None:
        return result

    # DPLL failed, use WalkSAT with restarts
    rng = random.Random(42)
    best_unsat = m + 1
    best_assignment = None
    trial_num = 0

    while time.time() < deadline:
        # Adaptive step budget
        max_steps = 5000 + trial_num * 1000
        if trial_num > 5:
            max_steps = 15000
        if trial_num > 15:
            max_steps = 20000

        unsat_count, assign = walk_sat_trial(rng, max_steps)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assign[:]
        if unsat_count == 0:
            return assign

        trial_num += 1

    return best_assignment if best_assignment is not None else [False] * n
