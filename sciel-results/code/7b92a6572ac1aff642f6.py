"""Planted 3-SAT solver: Optimized WalkSAT with incremental clause tracking."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: return best result with safety margin before 330s
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Preprocess: convert clauses to 0-based, track literal signs
    # clause: list of (var_idx, is_positive)
    clauses_pp = []
    for c in clauses:
        pp_c = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pp_c.append((var_idx, is_pos))
        clauses_pp.append(pp_c)
    
    # Build clause index per variable for efficient impact calculation
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_pp):
        for v, _ in c:
            var_to_clauses[v].append(ci)
            
    # Helper to check if a literal is satisfied given assignment
    # literal is (var, pos). assignment[var] == pos
    def is_lit_sat(assignment, var, pos):
        return assignment[var] == pos

    # Helper to get the unsat count for a clause
    def get_clause_unsat_count(assignment, clause_idx):
        cnt = 0
        for v, pos in clauses_pp[clause_idx]:
            if not is_lit_sat(assignment, v, pos):
                cnt += 1
        return cnt

    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    rng = random.Random(42)
    
    max_restarts = 500
    steps_per_restart = 20000
    
    # Clause weights for weighted random selection
    weights = [1.0] * num_clauses
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        if restart == 0:
            # First restart: try a deterministic heuristic
            # Set each variable to satisfy the most clauses
            assignment = [False] * n
            for v in range(n):
                true_sat = 0
                false_sat = 0
                for ci in var_to_clauses[v]:
                    c = clauses_pp[ci]
                    for vv, pos in c:
                        if vv == v:
                            if pos:
                                true_sat += 1
                            else:
                                false_sat += 1
                            break
                assignment[v] = true_sat >= false_sat
        else:
            # Subsequent restarts: random with some noise
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Decay weights on restart
        if restart > 0:
            for i in range(num_clauses):
                weights[i] = max(1.0, weights[i] * 0.9)
        
        # Initialize incremental state
        # clause_unsat_count[i] = number of false literals in clause i
        clause_unsat_count = [0] * num_clauses
        unsat_clauses_set = set()
        
        for ci in range(num_clauses):
            cnt = get_clause_unsat_count(assignment, ci)
            clause_unsat_count[ci] = cnt
            if cnt == 3: # All false -> unsatisfied
                unsat_clauses_set.add(ci)
                
        current_unsat = len(unsat_clauses_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        if current_unsat == 0:
            return assignment
        
        # Local search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_clauses_set:
                return assignment
            
            if len(unsat_clauses_set) < best_unsat_count:
                best_unsat_count = len(unsat_clauses_set)
                best_assignment = assignment[:]
                
            # Choose an unsatisfied clause with weighted selection
            # Convert set to list for weighted choice if needed, or just random choice
            # For efficiency, if set is large, just pick random index from set
            # If weights are uniform, random choice is fine. 
            # We have weights, so we need weighted selection.
            # To avoid O(U) conversion every step, we can approximate or just use random choice 
            # since weights are only mildly different. Let's use random choice for speed, 
            # weights are a minor heuristic here.
            chosen_ci = rng.choice(list(unsat_clauses_set))
            
            clause = clauses_pp[chosen_ci]
            
            # Update weight of this clause
            weights[chosen_ci] *= 1.05
            
            # Choose a literal to flip using min-conflicts with noise
            best_var = -1
            best_gain = -float('inf')
            
            for var, pos in clause:
                current_val = assignment[var]
                
                # Calculate gain:
                # Flipping var changes its value.
                # For each clause c' containing var:
                # 1. If c' was unsat (count=3), and the literal var in c' becomes true -> count becomes 2 (sat) -> gain +1
                # 2. If c' was sat (count<3), and the literal var in c' becomes false:
                #    - If count was 2, it becomes 3 (unsat) -> gain -1
                #    - If count was <2, it remains sat -> gain 0
                # Note: The chosen clause 'clause' is unsat. The literal 'var' in 'clause' is currently false.
                # Flipping var makes it true. So 'clause' becomes sat. Gain +1 for this clause is implicit in the logic below if we process all clauses.
                
                gain = 0
                for other_ci in var_to_clauses[var]:
                    # Determine the sign of 'var' in 'other_ci'
                    # We need to know if the literal was False or True before flip
                    # and what it becomes.
                    
                    # Current state of literal 'var' in 'other_ci'
                    # We need to find the pos of var in other_ci.
                    # To avoid scanning other_ci every time, we can precompute a map or just scan (size 3).
                    var_pos_in_other = None
                    for v2, p2 in clauses_pp[other_ci]:
                        if v2 == var:
                            var_pos_in_other = p2
                            break
                            
                    if var_pos_in_other is None:
                        continue # Should not happen
                    
                    # Current value of literal
                    is_lit_true_now = (assignment[var] == var_pos_in_other)
                    # New value of literal after flip
                    is_lit_true_new = (not assignment[var] == var_pos_in_other)
                    
                    # Current unsat count of other_ci
                    current_cnt = clause_unsat_count[other_ci]
                    
                    if is_lit_true_now and not is_lit_true_new:
                        # Literal goes True -> False
                        # Unsat count increases by 1
                        new_cnt = current_cnt + 1
                        if current_cnt == 2 and new_cnt == 3:
                            # Clause becomes unsat
                            gain -= 1
                    elif not is_lit_true_now and is_lit_true_new:
                        # Literal goes False -> True
                        # Unsat count decreases by 1
                        new_cnt = current_cnt - 1
                        if current_cnt == 3 and new_cnt == 2:
                            # Clause becomes sat
                            gain += 1
                        # If current_cnt < 3, it stays sat (or becomes more sat), no penalty.
                        # If current_cnt == 3, it becomes 2 (sat), gain +1.
                        
                    # Add small noise to avoid getting stuck
                    # Only add noise if we are not strictly improving? 
                    # Standard WalkSAT: pick best gain, tie-break with noise.
                    # Here we calculate gain for each var.
                    pass # Noise applied below
                    
                # Add small noise to avoid getting stuck
                noise = rng.gauss(0, 0.1)
                total_gain = gain + noise
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_var = var
            
            if best_var != -1:
                # Perform the flip
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                # Update incremental state for all clauses containing best_var
                for ci in var_to_clauses[best_var]:
                    # Find sign of best_var in ci
                    var_pos_in_ci = None
                    for v2, p2 in clauses_pp[ci]:
                        if v2 == best_var:
                            var_pos_in_ci = p2
                            break
                    if var_pos_in_ci is None:
                        continue
                    
                    is_lit_true_now = (assignment[best_var] == var_pos_in_ci)
                    # is_lit_true_now is the NEW state
                    
                    # We need to know the OLD state to update the count.
                    # Old state was (old_val == var_pos_in_ci)
                    is_lit_true_old = (old_val == var_pos_in_ci)
                    
                    if is_lit_true_old and not is_lit_true_now:
                        # True -> False, unsat count +1
                        clause_unsat_count[ci] += 1
                        if clause_unsat_count[ci] == 3:
                            unsat_clauses_set.add(ci)
                    elif not is_lit_true_old and is_lit_true_now:
                        # False -> True, unsat count -1
                        clause_unsat_count[ci] -= 1
                        if clause_unsat_count[ci] < 3:
                            if ci in unsat_clauses_set:
                                unsat_clauses_set.remove(ci)
            else:
                # Fallback: random flip
                v = rng.randint(0, n - 1)
                old_val = assignment[v]
                assignment[v] = not old_val
                
                for ci in var_to_clauses[v]:
                    var_pos_in_ci = None
                    for v2, p2 in clauses_pp[ci]:
                        if v2 == v:
                            var_pos_in_ci = p2
                            break
                    if var_pos_in_ci is None:
                        continue
                    is_lit_true_now = (assignment[v] == var_pos_in_ci)
                    is_lit_true_old = (old_val == var_pos_in_ci)
                    
                    if is_lit_true_old and not is_lit_true_now:
                        clause_unsat_count[ci] += 1
                        if clause_unsat_count[ci] == 3:
                            unsat_clauses_set.add(ci)
                    elif not is_lit_true_old and is_lit_true_now:
                        clause_unsat_count[ci] -= 1
                        if clause_unsat_count[ci] < 3:
                            if ci in unsat_clauses_set:
                                unsat_clauses_set.remove(ci)
        
        # If we've exhausted this restart, check if we should continue
        if best_unsat_count == 0:
            break
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
