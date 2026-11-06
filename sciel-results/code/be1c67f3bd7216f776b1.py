"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    # var_clauses[v] = list of (clause_idx, is_pos_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def calculate_sat_counts(assignment):
        """Calculate number of satisfied literals for each clause."""
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            count = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    count += 1
            sat_counts[i] = count
        return sat_counts
    
    def get_unsat_clauses(sat_counts):
        """Get list of indices of unsatisfied clauses."""
        return [i for i, c in enumerate(sat_counts) if c == 0]
    
    def walk_sat(assignment, rng, max_flips, sat_counts, unsat_clauses):
        """
        WalkSAT with incremental sat_counts update.
        Returns True if solution found, False otherwise.
        """
        flip_count = 0
        
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            noise = 0.5 * max(0, 1 - flip_count / max_flips)
            
            best_var = -1
            best_delta = float('inf')
            
            # Evaluate each variable in the chosen clause
            for var_idx, is_pos in cv:
                delta = 0
                old_val = assignment[var_idx]
                new_val = not old_val
                
                # Iterate through all clauses containing this variable
                for ci2, is_pos2 in var_clauses[var_idx]:
                    # Determine if the literal (var_idx, is_pos2) is currently satisfied
                    currently_sat = (old_val == is_pos2)
                    
                    # Determine if it will be satisfied after flip
                    post_sat = (new_val == is_pos2)
                    
                    if currently_sat and not post_sat:
                        # Literal goes from SAT to UNSAT
                        if sat_counts[ci2] == 1:
                            delta += 1
                    elif not currently_sat and post_sat:
                        # Literal goes from UNSAT to SAT
                        if sat_counts[ci2] == 0:
                            delta -= 1
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip from the clause
                best_var = rng.choice(cv)[0]
            
            # Perform the flip
            assignment[best_var] = not assignment[best_var]
            
            # Update sat_counts and unsat_clauses incrementally
            new_val = assignment[best_var]
            for ci2, is_pos2 in var_clauses[best_var]:
                if new_val == is_pos2:
                    # Literal became SAT
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        try:
                            unsat_clauses.remove(ci2)
                        except ValueError:
                            pass
                else:
                    # Literal is UNSAT
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        unsat_clauses.append(ci2)
            
            flip_count += 1
            
            if not unsat_clauses:
                return True
        
        return len(unsat_clauses) == 0
    
    def dpll_solve(clauses_list, n_vars):
        """
        DPLL with unit propagation and pure literal elimination.
        Returns assignment dict {var_idx: bool} or None if UNSAT.
        """
        # clauses_list: list of lists of (var_idx, is_positive)
        # assignment: dict mapping var_idx to bool
        
        if not clauses_list:
            return {}
        
        # Unit propagation
        assignment = {}
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses_list:
                # Check for satisfied clause
                sat = False
                unit_var = None
                unit_is_pos = None
                remaining = []
                
                for var_idx, is_pos in clause:
                    if var_idx in assignment:
                        if assignment[var_idx] == is_pos:
                            sat = True
                            break
                    else:
                        remaining.append((var_idx, is_pos))
                
                if sat:
                    continue
                
                if not remaining:
                    return None  # UNSAT
                
                if len(remaining) == 1:
                    unit_var, unit_is_pos = remaining[0]
                    assignment[unit_var] = unit_is_pos
                    changed = True
                    continue
                
                new_clauses.append(remaining)
            
            clauses_list = new_clauses
        
        if not clauses_list:
            return assignment
        
        # Pure literal elimination
        changed = True
        while changed:
            changed = False
            # Count occurrences of each literal
            pos_count = [0] * n_vars
            neg_count = [0] * n_vars
            
            for clause in clauses_list:
                seen_vars = set()
                for var_idx, is_pos in clause:
                    if var_idx in assignment:
                        continue
                    if var_idx not in seen_vars:
                        if is_pos:
                            pos_count[var_idx] += 1
                        else:
                            neg_count[var_idx] += 1
                        seen_vars.add(var_idx)
            
            # Check for pure literals
            for var_idx in range(n_vars):
                if var_idx in assignment:
                    continue
                if pos_count[var_idx] > 0 and neg_count[var_idx] == 0:
                    assignment[var_idx] = True
                    changed = True
                elif neg_count[var_idx] > 0 and pos_count[var_idx] == 0:
                    assignment[var_idx] = False
                    changed = True
            
            if not changed:
                break
            
            # Remove satisfied clauses
            new_clauses = []
            for clause in clauses_list:
                sat = False
                remaining = []
                for var_idx, is_pos in clause:
                    if var_idx in assignment:
                        if assignment[var_idx] == is_pos:
                            sat = True
                            break
                    else:
                        remaining.append((var_idx, is_pos))
                
                if not sat:
                    if not remaining:
                        return None
                    new_clauses.append(remaining)
            
            clauses_list = new_clauses
        
        if not clauses_list:
            return assignment
        
        # Choose a variable to branch on
        # Use frequency heuristic: pick variable appearing in most clauses
        var_freq = [0] * n_vars
        for clause in clauses_list:
            seen_vars = set()
            for var_idx, is_pos in clause:
                if var_idx not in assignment and var_idx not in seen_vars:
                    var_freq[var_idx] += 1
                    seen_vars.add(var_idx)
        
        # Pick variable with highest frequency
        max_freq = -1
        branch_var = -1
        for var_idx in range(n_vars):
            if var_idx not in assignment and var_freq[var_idx] > max_freq:
                max_freq = var_freq[var_idx]
                branch_var = var_idx
        
        if branch_var == -1:
            return assignment
        
        # Branch: try True first
        assignment[branch_var] = True
        result = dpll_solve(clauses_list, n_vars)
        if result is not None:
            return result
        
        # Branch: try False
        assignment[branch_var] = False
        result = dpll_solve(clauses_list, n_vars)
        if result is not None:
            return result
        
        # UNSAT
        del assignment[branch_var]
        return None
    
    def vsids_dpll(clauses_list, n_vars, deadline):
        """
        DPLL with VSIDS variable activity scoring and watched literals.
        Returns assignment list of booleans or None if UNSAT/timeout.
        """
        # Variable activities
        activities = [1.0] * n_vars
        var_inc = 1.0
        
        # Assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n_vars
        
        # Trail: list of (var_idx, value)
        trail = []
        
        # Watched literals: for each clause, watch two literals
        # watched[var_idx] = list of (clause_idx, is_pos)
        watched = [[] for _ in range(n_vars)]
        
        # Clause status: 0 = active, 1 = satisfied, 2 = false
        clause_status = [0] * len(clauses_list)
        
        # Initialize watches
        for ci, clause in enumerate(clauses_list):
            if len(clause) >= 2:
                v1, p1 = clause[0]
                v2, p2 = clause[1]
                watched[v1].append((ci, p1))
                watched[v2].append((ci, p2))
            elif len(clause) == 1:
                v1, p1 = clause[0]
                watched[v1].append((ci, p1))
        
        # Unit propagation queue
        unit_queue = []
        
        def propagate():
            """Process unit clauses via watched literals."""
            while unit_queue:
                var_idx, value = unit_queue.pop()
                
                if assignment[var_idx] != -1:
                    continue
                
                # Assign variable
                assignment[var_idx] = value
                trail.append((var_idx, value))
                
                # Increase activities
                for v in range(n_vars):
                    if assignment[v] == -1:
                        activities[v] += var_inc
                
                # Process watched literals for this variable
                new_watches = []
                for ci, is_pos in watched[var_idx]:
                    if clause_status[ci] != 0:
                        new_watches.append((ci, is_pos))
                        continue
                    
                    # Check if clause is satisfied by another literal
                    clause = clauses_list[ci]
                    satisfied = False
                    for v2, p2 in clause:
                        if v2 == var_idx:
                            continue
                        if assignment[v2] != -1:
                            if assignment[v2] == p2:
                                satisfied = True
                                break
                        else:
                            # Add as new watch
                            new_watches.append((ci, p2))
                    
                    if not satisfied:
                        # Check if clause is now false
                        all_false = True
                        for v2, p2 in clause:
                            if v2 == var_idx:
                                continue
                            if assignment[v2] == -1:
                                all_false = False
                                break
                            elif assignment[v2] != p2:
                                pass  # This literal is false
                        # Actually, we need to check if all literals are false
                        # Let's redo this
                        all_false = True
                        for v2, p2 in clause:
                            if assignment[v2] == -1:
                                all_false = False
                                break
                            elif assignment[v2] != p2:
                                pass
                        # Simpler: check if any unassigned or satisfied
                        has_unassigned_or_sat = False
                        for v2, p2 in clause:
                            if assignment[v2] == -1:
                                has_unassigned_or_sat = True
                                break
                            elif assignment[v2] == p2:
                                has_unassigned_or_sat = True
                                break
                        
                        if not has_unassigned_or_sat:
                            # Clause is false -> UNSAT
                            return False
                        else:
                            # Unit clause: add to queue
                            for v2, p2 in clause:
                                if assignment[v2] == -1:
                                    unit_queue.append((v2, p2))
                                    break
                            # Don't add to new_watches since it's unit
                            continue
                    
                    # Clause is satisfied or has unassigned literals
                    # Keep watching
                    new_watches.append((ci, is_pos))
                
                watched[var_idx] = new_watches
            
            return True
        
        # Initial unit propagation
        for ci, clause in enumerate(clauses_list):
            if len(clause) == 1:
                v, p = clause[0]
                unit_queue.append((v, p))
        
        if not propagate():
            return None
        
        # Check if solved
        all_assigned = all(a != -1 for a in assignment)
        if all_assigned:
            return [bool(a) for a in assignment]
        
        # DPLL search
        def search(level):
            if time.time() > deadline:
                return None
            
            if not propagate():
                return None
            
            all_assigned = all(a != -1 for a in assignment)
            if all_assigned:
                return [bool(a) for a in assignment]
            
            # Choose variable using VSIDS
            best_var = -1
            best_activity = -1
            for v in range(n_vars):
                if assignment[v] == -1 and activities[v] > best_activity:
                    best_activity = activities[v]
                    best_var = v
            
            if best_var == -1:
                return [bool(a) for a in assignment]
            
            # Branch: try False first (often better for SAT)
            assignment[best_var] = 0
            trail.append((best_var, 0))
            
            result = search(level + 1)
            if result is not None:
                return result
            
            # Backtrack
            while trail and trail[-1][0] != best_var:
                v, val = trail.pop()
                assignment[v] = -1
            if trail:
                trail.pop()
            assignment[best_var] = -1
            
            # Branch: try True
            assignment[best_var] = 1
            trail.append((best_var, 1))
            
            result = search(level + 1)
            if result is not None:
                return result
            
            # Backtrack
            while trail and trail[-1][0] != best_var:
                v, val = trail.pop()
                assignment[v] = -1
            if trail:
                trail.pop()
            assignment[best_var] = -1
            
            # Increase activity of best_var
            activities[best_var] += var_inc
            if activities[best_var] > 1e100:
                # Rescale
                for v in range(n_vars):
                    activities[v] /= 1e100
                var_inc /= 1e100
            
            return None
        
        return search(0)
    
    # Multiple restarts with mixed strategies
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Alternate between WalkSAT and DPLL-based approaches
        if restart % 3 == 0:
            # WalkSAT
            rng = random.Random(42 + restart)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            sat_counts = calculate_sat_counts(assignment)
            unsat_clauses = get_unsat_clauses(sat_counts)
            
            if not unsat_clauses:
                return assignment
            
            max_flips = 10000 + restart * 1000
            
            if walk_sat(assignment, rng, max_flips, sat_counts, unsat_clauses):
                return assignment
            
            unsat = len(unsat_clauses)
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = assignment[:]
        
        elif restart % 3 == 1:
            # DPLL with unit propagation
            result = dpll_solve(clause_vars, n)
            if result is not None:
                # Convert dict to list
                assignment = [result.get(v, False) for v in range(n)]
                return assignment
        
        else:
            # VSIDS DPLL
            result = vsids_dpll(clause_vars, n, deadline - 10)
            if result is not None:
                return result
        
        restart += 1
        
        if restart > 10000:
            break
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
