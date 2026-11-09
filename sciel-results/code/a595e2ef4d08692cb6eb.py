import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause structure for fast access
    # clause_data[i] = list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            cd.append((var, pos))
        clause_data.append(cd)
    
    # Build variable-to-clause index for fast scoring
    # var_clauses[var] = list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for var, pos in cd:
            var_clauses[var].append((ci, pos))
    
    deadline = time.time() + 290  # Safety margin from 330s limit
    rng = random.Random(42)
    
    def is_clause_sat(clause_idx, assignment):
        for var, pos in clause_data[clause_idx]:
            if assignment[var] == pos:
                return True
        return False
        
    def count_satisfied(assignment):
        count = 0
        for i in range(num_clauses):
            if is_clause_sat(i, assignment):
                count += 1
        return count

    def dpll_solver(deadline):
        """
        DPLL with unit propagation.
        Returns assignment if solved, None if failed or timed out.
        """
        # Copy clauses to avoid modification issues, though we track assignments
        # We will use a recursive backtracking approach.
        # For 50 vars, recursion depth is safe.
        
        # To make it fast, we'll implement a simple DPLL.
        # State: assignment (list of None/True/False), clauses (list of clauses, updated by unit prop)
        
        # Initialize assignment
        assignment = [None] * n
        
        # We need to manage the set of active clauses.
        # For efficiency, we can just check satisfaction against the original clauses 
        # but that's slow. 
        # Better: maintain a list of active clauses.
        
        # Simple DPLL implementation
        def unit_propagate(clauses, assignment):
            """
            Returns None if conflict, else (new_clauses, new_assignment)
            """
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    # Check if clause is satisfied
                    sat = False
                    # Check if clause has unit literal
                    unassigned = []
                    for var, pos in clause:
                        if assignment[var] is not None:
                            if assignment[var] == pos:
                                sat = True
                                break
                        else:
                            unassigned.append((var, pos))
                    
                    if sat:
                        continue
                    
                    if not unassigned:
                        # Conflict: clause is empty and not satisfied
                        return None, None
                    
                    if len(unassigned) == 1:
                        var, pos = unassigned[0]
                        if assignment[var] is not None:
                            if assignment[var] != pos:
                                return None, None
                        else:
                            assignment[var] = pos
                            changed = True
                        # Clause is satisfied by the unit literal, so it can be dropped
                        # Actually, if we set the var, the clause is satisfied.
                        # So we don't add it to new_clauses.
                    else:
                        new_clauses.append(unassigned)
                
                clauses = new_clauses
            return clauses, assignment

        def dpll_recursive(clauses, assignment, depth=0):
            if time.time() > deadline:
                return None
            
            # Check for conflicts and unit propagation
            res = unit_propagate(clauses, assignment)
            if res[0] is None:
                return None # Conflict
            clauses, assignment = res
            
            if not clauses:
                # All clauses satisfied
                # Fill remaining unassigned vars arbitrarily (e.g., False)
                final_assignment = [False if a is None else a for a in assignment]
                return final_assignment
            
            # Choose a variable
            # Heuristic: Pick variable from the shortest clause
            min_len = len(clauses[0])
            min_clause = clauses[0]
            for c in clauses[1:]:
                if len(c) < min_len:
                    min_len = len(c)
                    min_clause = c
            
            # Pick first variable in that clause
            var, pos = min_clause[0]
            
            # Branch 1: Set var to pos (satisfies this literal)
            assignment[var] = pos
            # Filter clauses: remove satisfied ones, remove var from others
            new_clauses_pos = []
            for c in clauses:
                # Check if satisfied by assignment
                sat = False
                new_c = []
                for v, p in c:
                    if v == var:
                        if assignment[v] == p:
                            sat = True
                        # Don't add this literal to new clause
                    else:
                        if assignment[v] is not None:
                            if assignment[v] == p:
                                sat = True
                        else:
                            new_c.append((v, p))
                if not sat:
                    if not new_c:
                        return None # Conflict in this branch
                    new_clauses_pos.append(new_c)
            
            sol = dpll_recursive(new_clauses_pos, assignment, depth + 1)
            if sol:
                return sol
            
            # Branch 2: Set var to not pos
            assignment[var] = not pos
            new_clauses_neg = []
            for c in clauses:
                sat = False
                new_c = []
                for v, p in c:
                    if v == var:
                        if assignment[v] == p:
                            sat = True
                    else:
                        if assignment[v] is not None:
                            if assignment[v] == p:
                                sat = True
                        else:
                            new_c.append((v, p))
                if not sat:
                    if not new_c:
                        return None # Conflict in this branch
                    new_clauses_neg.append(new_c)
            
            sol = dpll_recursive(new_clauses_neg, assignment, depth + 1)
            if sol:
                return sol
            
            # Backtrack
            assignment[var] = None
            return None

        # Start DPLL
        initial_clauses = [list(c) for c in clause_data]
        return dpll_recursive(initial_clauses, [None]*n)

    def walk_sat_trial(assignment, max_iters, deadline):
        """Run WalkSAT with score-based literal selection."""
        cur_count = count_satisfied(assignment)
        
        if cur_count == num_clauses:
            return assignment, cur_count
        
        no_improve = 0
        restart_threshold = 1000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(num_clauses):
                if not is_clause_sat(i, assignment):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment, num_clauses
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clause_data[clause_idx]
            
            # Score each literal: how many clauses would become satisfied
            # if we flip this variable to make this literal true
            scores = []
            for var, pos in clause:
                # If we set assignment[var] = pos, which clauses gain satisfaction?
                # We use var_clauses for speed
                gain = 0
                lose = 0
                for ci, polarity in var_clauses[var]:
                    # Check if clause is currently satisfied
                    cur_sat = False
                    for v, p in clause_data[ci]:
                        if assignment[v] == p:
                            cur_sat = True
                            break
                    
                    # After setting var=pos, is this clause satisfied?
                    if polarity == pos:
                        new_sat = True
                    else:
                        new_sat = False
                        for v, p in clause_data[ci]:
                            if v == var:
                                continue
                            if assignment[v] == p:
                                new_sat = True
                                break
                    
                    if new_sat and not cur_sat:
                        gain += 1
                    elif not new_sat and cur_sat:
                        lose += 1
                scores.append(gain - lose)
            
            # Pick literal with highest score, with some randomization
            max_score = max(scores)
            # Use a soft selection: prefer higher scores but allow some randomness
            # Weighted random selection
            total_weight = sum(s + 1 for s in scores)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_idx = 0
            for i, s in enumerate(scores):
                cumulative += s + 1
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            var, pos = clause[chosen_idx]
            assignment[var] = pos
            
            new_count = count_satisfied(assignment)
            if new_count > cur_count:
                cur_count = new_count
                no_improve = 0
                if cur_count == num_clauses:
                    return assignment, cur_count
            else:
                no_improve += 1
            
            if no_improve >= restart_threshold:
                break
        
        return assignment, cur_count

    # Phase 1: Try DPLL with a short timeout
    # DPLL is very fast on easy instances. 
    dpll_deadline = time.time() + 5
    sol = dpll_solver(dpll_deadline)
    if sol:
        return sol

    # Phase 2: WalkSAT with restarts
    best_assignment = [False] * n
    best_count = -1
    
    max_restarts = 100
    iters_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        assignment, cur_count = walk_sat_trial(assignment, iters_per_restart, deadline)
        
        if cur_count > best_count:
            best_count = cur_count
            best_assignment = assignment[:]
        
        if best_count == num_clauses:
            return best_assignment
    
    return best_assignment
