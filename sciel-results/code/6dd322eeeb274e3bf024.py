"""Synthetic planted 3-SAT solver (50 variables) with optimized incremental local search."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 280  # Safety margin before 300s kill

    # Precompute clause structure for speed
    # For each clause, store the variable indices (0-based) and polarities
    # polarity: 1 means literal is positive (var must be True), -1 means negative (var must be False)
    
    # Build variable-to-clauses index for efficient damage calculation
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []  # list of lists of (var_idx, polarity)
    for c in clauses:
        cvars = []
        for l in c:
            v = abs(l) - 1
            p = 1 if l > 0 else -1
            cvars.append((v, p))
            var_to_clauses[v].append(len(clause_vars))
        clause_vars.append(cvars)
    
    def is_satisfied(clause_idx, assignment):
        for v, p in clause_vars[clause_idx]:
            if assignment[v] == (p == 1):
                return True
        return False

    def count_satisfied(assignment):
        """Count number of satisfied clauses."""
        cnt = 0
        for ci in range(num_clauses):
            if is_satisfied(ci, assignment):
                cnt += 1
        return cnt

    def get_unsat_clauses(assignment):
        return [ci for ci in range(num_clauses) if not is_satisfied(ci, assignment)]

    def walksat_step(assignment, rng, unsat_clauses):
        """WalkSAT: pick random unsat clause, flip literal that minimizes damage."""
        if not unsat_clauses:
            return assignment
        
        clause_idx = rng.choice(unsat_clauses)
        
        best_var = None
        best_polarity = None
        best_damage = float('inf')
        
        for v, p in clause_vars[clause_idx]:
            # Calculate damage: how many clauses would be broken if we flip variable v
            damage = 0
            for ci in var_to_clauses[v]:
                if ci == clause_idx:
                    continue
                # Check if clause ci is currently satisfied only by variable v
                # with the current value
                # If we flip v, will clause ci become unsatisfied?
                # Clause ci is broken if:
                # 1. It's currently satisfied
                # 2. It's satisfied ONLY by variable v (no other literal is satisfied)
                # 3. After flipping v, v no longer satisfies clause ci
                
                # Check if clause ci is currently satisfied
                currently_sat = False
                satisfied_by_v = False
                satisfied_by_other = False
                
                for v2, p2 in clause_vars[ci]:
                    if assignment[v2] == (p2 == 1):
                        currently_sat = True
                        if v2 == v:
                            satisfied_by_v = True
                        else:
                            satisfied_by_other = True
                
                if currently_sat and satisfied_by_v and not satisfied_by_other:
                    # Flipping v would break this clause
                    damage += 1
            
            if damage < best_damage:
                best_damage = damage
                best_var = v
                best_polarity = p
        
        if best_var is not None:
            # Flip the variable to satisfy the chosen clause
            assignment[best_var] = (best_polarity == 1)
        
        return assignment

    def simulated_annealing(assignment, rng, start_temp, end_temp, max_steps, deadline):
        """Simulated annealing for Max-SAT."""
        current_score = count_satisfied(assignment)
        best_assignment = assignment[:]
        best_score = current_score
        
        if current_score == num_clauses:
            return assignment, best_assignment, True
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            # Temperature schedule (geometric cooling)
            progress = step / max_steps
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Pick a random variable to flip
            var_idx = rng.randint(0, n - 1)
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            new_score = count_satisfied(assignment)
            delta = new_score - current_score
            
            if delta >= 0:
                # Accept improving or equal moves
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_assignment = assignment[:]
                    if best_score == num_clauses:
                        return assignment, best_assignment, True
            else:
                # Accept worsening moves with probability exp(delta/temp)
                if temp > 0:
                    prob = math.exp(delta / temp)
                    if rng.random() < prob:
                        current_score = new_score
                    else:
                        # Reject: undo the flip
                        assignment[var_idx] = old_val
        
        return assignment, best_assignment, best_score == num_clauses

    def walksat_search(assignment, rng, max_steps, deadline):
        """Standard WalkSAT search."""
        unsat = get_unsat_clauses(assignment)
        if not unsat:
            return assignment, True
        
        best_assignment = assignment[:]
        best_unsat = len(unsat)
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if not unsat:
                return assignment, True
            
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            # Use WalkSAT step
            assignment = walksat_step(assignment, rng, unsat)
            unsat = get_unsat_clauses(assignment)
        
        return best_assignment, best_unsat == 0

    def flipsat_step(assignment, rng, unsat_clauses):
        """FlipSAT: pick a random unsat clause, flip one of its literals randomly."""
        if not unsat_clauses:
            return assignment
        clause_idx = rng.choice(unsat_clauses)
        v, p = rng.choice(clause_vars[clause_idx])
        assignment[v] = (p == 1)
        return assignment

    def flipsat_search(assignment, rng, max_steps, deadline):
        """FlipSAT search."""
        unsat = get_unsat_clauses(assignment)
        if not unsat:
            return assignment, True
        
        best_assignment = assignment[:]
        best_unsat = len(unsat)
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if not unsat:
                return assignment, True
            
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            assignment = flipsat_step(assignment, rng, unsat)
            unsat = get_unsat_clauses(assignment)
        
        return best_assignment, best_unsat == 0

    # Main search loop with multiple strategies and restarts
    best_assignment = None
    best_score = -1
    
    # Strategy 1: Simulated Annealing with multiple restarts
    sa_restarts = 20
    sa_steps = 5000
    sa_start_temp = 1.0
    sa_end_temp = 0.01
    
    for restart in range(sa_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(1000 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if count_satisfied(assignment) == num_clauses:
            return assignment
        
        _, best_sa, solved = simulated_annealing(
            assignment, rng, sa_start_temp, sa_end_temp, sa_steps, deadline
        )
        
        if solved:
            return best_sa
        
        sa_score = count_satisfied(best_sa)
        if sa_score > best_score:
            best_score = sa_score
            best_assignment = best_sa[:]
    
    # Strategy 2: WalkSAT with multiple restarts
    ws_restarts = 25
    ws_steps = 8000
    
    for restart in range(ws_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(5000 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        if count_satisfied(assignment) == num_clauses:
            return assignment
        
        best_ws, solved = walksat_search(assignment, rng, ws_steps, deadline)
        
        if solved:
            return best_ws
        
        ws_score = count_satisfied(best_ws)
        if ws_score > best_score:
            best_score = ws_score
            best_assignment = best_ws[:]
    
    # Strategy 3: FlipSAT with multiple restarts
    fs_restarts = 15
    fs_steps = 5000
    
    for restart in range(fs_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(9000 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        if count_satisfied(assignment) == num_clauses:
            return assignment
        
        best_fs, solved = flipsat_search(assignment, rng, fs_steps, deadline)
        
        if solved:
            return best_fs
        
        fs_score = count_satisfied(best_fs)
        if fs_score > best_score:
            best_score = fs_score
            best_assignment = best_fs[:]
    
    # Final polish: try local search from the best assignment found
    if best_assignment is not None and time.time() < deadline - 15:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        unsat = get_unsat_clauses(assignment)
        
        if not unsat:
            return assignment
        
        # Try WalkSAT polish from best point
        for step in range(30000):
            if time.time() > deadline - 5:
                break
            if not unsat:
                return assignment
            assignment = walksat_step(assignment, rng, unsat)
            unsat = get_unsat_clauses(assignment)
        
        if not unsat:
            return assignment
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [False] * n
