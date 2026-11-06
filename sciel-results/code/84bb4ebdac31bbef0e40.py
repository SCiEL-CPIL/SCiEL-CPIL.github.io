# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin

    # Convert clauses to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build data structures
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # === PREPROCESSING: Pure Literal Elimination + Unit Propagation ===
    fixed = {}
    remaining_vars = set(range(n))
    remaining_clauses = list(clauses_0based)
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        
        pos_count = {}
        neg_count = {}
        for clause in remaining_clauses:
            for var, sign in clause:
                if var not in remaining_vars:
                    continue
                if sign > 0:
                    pos_count[var] = pos_count.get(var, 0) + 1
                else:
                    neg_count[var] = neg_count.get(var, 0) + 1
        
        for var in list(remaining_vars):
            if var not in pos_count and var not in neg_count:
                continue
            if var in pos_count and var not in neg_count:
                fixed[var] = True
                remaining_vars.discard(var)
                changed = True
            elif var not in pos_count and var in neg_count:
                fixed[var] = False
                remaining_vars.discard(var)
                changed = True
        
        if changed:
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
            continue
        
        unit_clauses = []
        for clause in remaining_clauses:
            unassigned = [(var, sign) for var, sign in clause if var in remaining_vars]
            if len(unassigned) == 1:
                unit_clauses.append(unassigned[0])
        
        if unit_clauses:
            for var, sign in unit_clauses:
                if var in remaining_vars:
                    fixed[var] = (sign > 0)
                    remaining_vars.discard(var)
            changed = True
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
    
    n_reduced = len(remaining_vars)
    m_reduced = len(remaining_clauses)
    
    if n_reduced == 0:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            else:
                full_assignment[var] = False
        return full_assignment

    # Map reduced variables to indices
    reduced_var_list = list(remaining_vars)
    var_to_reduced_idx = {var: i for i, var in enumerate(reduced_var_list)}
    
    # Convert remaining clauses to use reduced indices
    reduced_clauses = []
    for clause in remaining_clauses:
        reduced_clause = []
        for var, sign in clause:
            if var in remaining_vars:
                reduced_clause.append((var_to_reduced_idx[var], sign))
        if reduced_clause:
            reduced_clauses.append(reduced_clause)
    
    n_r = n_reduced
    m_r = len(reduced_clauses)
    
    # Build var_clauses for reduced instance
    reduced_var_clauses = [[] for _ in range(n_r)]
    for ci, clause in enumerate(reduced_clauses):
        for var, sign in clause:
            reduced_var_clauses[var].append((ci, sign))

    def _run_walksat(rng, max_time, walk_prob=0.3):
        """
        Highly optimized WalkSAT using vectorized satisfaction tracking.
        Returns (assignment, is_solved)
        """
        if m_r == 0:
            return [True] * n_r, True
            
        assignment = np.array([rng.random() < 0.5 for _ in range(n_r)], dtype=bool)
        
        # Precompute clause data for fast access
        # clause_vars[i] = array of var indices in clause i
        # clause_signs[i] = array of signs in clause i
        clause_vars = []
        clause_signs = []
        for clause in reduced_clauses:
            vs, ss = zip(*clause) if clause else ((), ())
            clause_vars.append(np.array(vs, dtype=np.int32))
            clause_signs.append(np.array(ss, dtype=np.int32))
        
        # Initialize sat_count
        # sat_count[i] = number of literals in clause i that are true
        sat_count = np.zeros(m_r, dtype=np.int32)
        unsat_count = 0
        unsat_indices = []
        
        for i in range(m_r):
            # assignment[var] == (sign > 0)
            # sign > 0 is True if sign==1, False if sign==-1
            # We want assignment[var] == (sign == 1)
            # Let's precompute a boolean mask for "literal is true"
            # lit_true = assignment[var] if sign==1 else ~assignment[var]
            # This is equivalent to assignment[var] == (sign > 0)
            
            # Vectorized check:
            # signs are 1 or -1. 
            # If sign is 1, we want assignment[var].
            # If sign is -1, we want ~assignment[var].
            # This can be done by: (assignment[var] == (sign > 0))
            
            cv = clause_vars[i]
            cs = clause_signs[i]
            # cs > 0 is a boolean array
            # assignment[cv] is a boolean array
            # match = assignment[cv] == (cs > 0)
            # count = sum(match)
            
            match = assignment[cv] == (cs > 0)
            cnt = np.sum(match)
            sat_count[i] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_indices.append(i)
        
        if unsat_count == 0:
            return assignment.tolist(), True

        best_assignment = assignment.copy()
        best_unsat = unsat_count
        
        current_time = time.time()
        while time.time() < current_time + max_time:
            if unsat_count == 0:
                return assignment.tolist(), True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
                
            # Pick an unsatisfied clause
            if not unsat_indices:
                return assignment.tolist(), True
                
            ci = rng.choice(unsat_indices)
            
            # Heuristic: Pick variable to flip
            # 1. With probability walk_prob, pick random variable from clause
            # 2. Otherwise, pick variable that minimizes "damage" (maximizes gain)
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            if rng.random() < walk_prob:
                idx = rng.randrange(len(cv))
                var_to_flip = cv[idx]
            else:
                best_score = -float('inf')
                var_to_flip = cv[0]
                
                for k in range(len(cv)):
                    var = cv[k]
                    sign = cs[k]
                    
                    # Calculate score: 
                    # If we flip var, how many other clauses become satisfied? 
                    # How many become unsatisfied?
                    
                    # Current state of var: assignment[var]
                    # If we flip, new state is ~assignment[var]
                    
                    # For each clause containing var:
                    # If the literal in that clause is currently TRUE, it becomes FALSE.
                    #   If that clause has sat_count == 1, it becomes UNSAT. (Damage +1)
                    #   Else, sat_count decreases, stays SAT.
                    # If the literal in that clause is currently FALSE, it becomes TRUE.
                    #   If that clause has sat_count == 0, it becomes SAT. (Gain +1)
                    #   Else, sat_count increases, stays SAT.
                    
                    score = 0
                    for (other_ci, other_sign) in reduced_var_clauses[var]:
                        if other_ci == ci:
                            continue
                            
                        # Check current truth value of this literal
                        # Literal is true if assignment[var] == (other_sign > 0)
                        is_true = (assignment[var] == (other_sign > 0))
                        
                        if is_true:
                            # Becomes false. Check if clause becomes unsat.
                            if sat_count[other_ci] == 1:
                                score -= 1
                        else:
                            # Becomes true. Check if clause becomes sat.
                            if sat_count[other_ci] == 0:
                                score += 1
                            
                    if score > best_score:
                        best_score = score
                        var_to_flip = var
            
            # Perform Flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update sat_count and unsat_count
            for (other_ci, other_sign) in reduced_var_clauses[var_to_flip]:
                # Determine if the literal in other_ci was True or False
                # Literal is true if assignment[var] == (sign > 0)
                # We just flipped, so we use old_val to determine previous state
                was_true = (old_val == (other_sign > 0))
                
                if was_true:
                    # Literal becomes False. sat_count decreases.
                    sat_count[other_ci] -= 1
                    if sat_count[other_ci] == 0:
                        # Clause becomes unsat
                        unsat_count += 1
                        unsat_indices.append(other_ci)
                else:
                    # Literal becomes True. sat_count increases.
                    sat_count[other_ci] += 1
                    if sat_count[other_ci] == 1:
                        # Clause becomes sat (was 0)
                        unsat_count -= 1
                        # Remove from unsat_indices efficiently?
                        # Using a list and removing is O(N). 
                        # For 150 vars, m is likely < 1000. 
                        # A set or boolean mask is better.
                        pass
                        
            # Optimization: Rebuild unsat_indices if it gets too dirty?
            # Or use a boolean mask `is_unsat` and filter periodically.
            # Given the small size, let's use a boolean mask for `is_unsat` to make removal O(1) or O(M).
            # Actually, maintaining a list and removing specific indices is O(M).
            # Let's switch to a boolean mask `unsat_mask` and `np.where` for selection.
            # But we are in a tight loop.
            
            # Let's refine the update logic to use a boolean mask `unsat_mask`
            # This requires changing the initialization and selection logic.
            
            # For now, the list approach might be slow if m is large.
            # Let's assume m is manageable (< 2000).
            # To speed up removal:
            # Instead of `unsat_indices.remove(other_ci)`, we can just leave it and check validity?
            # No, `rng.choice` on a list with stale entries is bad.
            
            # Let's re-implement the inner loop with a boolean mask `unsat_mask`
            # and `unsat_count`.
            # Selection: `np.where(unsat_mask)[0]` then `rng.choice`.
            # This is O(M) per step.
            
            # Given the time limit, the current list approach with O(M) removal is likely acceptable for 150 vars.
            # But let's optimize the removal.
            # We can't easily remove from a list in O(1).
            
            # Alternative: Don't maintain `unsat_indices` as a list.
            # Just maintain `unsat_mask` (boolean array) and `unsat_count`.
            # Selection: `unsat_idx_array = np.nonzero(unsat_mask)[0]`; `ci = rng.choice(unsat_idx_array)`.
            
            # Let's rewrite the solver to use `unsat_mask`.
            pass # Placeholder for logic change
            
        return best_assignment.tolist(), False

    def _run_walksat_v2(rng, max_time, walk_prob=0.3):
        """
        Optimized WalkSAT using boolean mask for unsat clauses.
        """
        if m_r == 0:
            return [True] * n_r, True
            
        assignment = np.array([rng.random() < 0.5 for _ in range(n_r)], dtype=bool)
        
        clause_vars = []
        clause_signs = []
        for clause in reduced_clauses:
            vs, ss = zip(*clause) if clause else ((), ())
            clause_vars.append(np.array(vs, dtype=np.int32))
            clause_signs.append(np.array(ss, dtype=np.int32))
        
        sat_count = np.zeros(m_r, dtype=np.int32)
        unsat_mask = np.zeros(m_r, dtype=bool)
        unsat_count = 0
        
        for i in range(m_r):
            cv = clause_vars[i]
            cs = clause_signs[i]
            match = assignment[cv] == (cs > 0)
            cnt = np.sum(match)
            sat_count[i] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_mask[i] = True
        
        if unsat_count == 0:
            return assignment.tolist(), True

        best_assignment = assignment.copy()
        best_unsat = unsat_count
        
        current_time = time.time()
        while time.time() < current_time + max_time:
            if unsat_count == 0:
                return assignment.tolist(), True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
                
            # Pick an unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist(), True
            ci = rng.choice(unsat_indices)
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            if rng.random() < walk_prob:
                idx = rng.randrange(len(cv))
                var_to_flip = cv[idx]
            else:
                best_score = -float('inf')
                var_to_flip = cv[0]
                
                for k in range(len(cv)):
                    var = cv[k]
                    sign = cs[k]
                    
                    score = 0
                    for (other_ci, other_sign) in reduced_var_clauses[var]:
                        if other_ci == ci:
                            continue
                            
                        is_true = (assignment[var] == (other_sign > 0))
                        
                        if is_true:
                            if sat_count[other_ci] == 1:
                                score -= 1
                        else:
                            if sat_count[other_ci] == 0:
                                score += 1
                            
                    if score > best_score:
                        best_score = score
                        var_to_flip = var
            
            # Perform Flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update sat_count and unsat_mask
            for (other_ci, other_sign) in reduced_var_clauses[var_to_flip]:
                was_true = (old_val == (other_sign > 0))
                
                if was_true:
                    sat_count[other_ci] -= 1
                    if sat_count[other_ci] == 0:
                        if not unsat_mask[other_ci]:
                            unsat_mask[other_ci] = True
                            unsat_count += 1
                else:
                    sat_count[other_ci] += 1
                    if sat_count[other_ci] == 1:
                        if unsat_mask[other_ci]:
                            unsat_mask[other_ci] = False
                            unsat_count -= 1
            
        return best_assignment.tolist(), False

    # Run Solvers
    rng = random.Random(42)
    
    # Allocate time: 60% to WalkSAT, 40% to Simulated Annealing (or second WalkSAT)
    total_time = deadline - time.time()
    if total_time <= 0:
        return [False] * n
        
    walk_time = total_time * 0.6
    sa_time = total_time * 0.4
    
    result, solved = _run_walksat_v2(rng, walk_time, walk_prob=0.3)
    
    if solved:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assignment[var] = result[var_to_reduced_idx[var]]
            else:
                full_assignment[var] = False
        return full_assignment

    # If not solved, try Simulated Annealing with the same fast update logic
    def _run_sa(rng, max_time):
        if m_r == 0:
            return [True] * n_r, True
            
        assignment = np.array([rng.random() < 0.5 for _ in range(n_r)], dtype=bool)
        
        clause_vars = []
        clause_signs = []
        for clause in reduced_clauses:
            vs, ss = zip(*clause) if clause else ((), ())
            clause_vars.append(np.array(vs, dtype=np.int32))
            clause_signs.append(np.array(ss, dtype=np.int32))
        
        sat_count = np.zeros(m_r, dtype=np.int32)
        unsat_mask = np.zeros(m_r, dtype=bool)
        unsat_count = 0
        
        for i in range(m_r):
            cv = clause_vars[i]
            cs = clause_signs[i]
            match = assignment[cv] == (cs > 0)
            cnt = np.sum(match)
            sat_count[i] = cnt
            if cnt == 0:
                unsat_count += 1
                unsat_mask[i] = True
        
        if unsat_count == 0:
            return assignment.tolist(), True

        best_assignment = assignment.copy()
        best_unsat = unsat_count
        
        T0 = 1.0
        T_end = 0.005
        # Estimate steps
        # Assume ~1000 steps/sec? 
        # We don't know exact speed, so just run until time limit.
        
        current_time = time.time()
        T = T0
        # Decay T based on time? Or fixed steps?
        # Fixed steps is risky if speed varies.
        # Let's use a large number of steps and break on time.
        # Or, decay T exponentially with time.
        
        while time.time() < current_time + max_time:
            if unsat_count == 0:
                return assignment.tolist(), True
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Pick random variable
            var = rng.randrange(n_r)
            old_val = assignment[var]
            assignment[var] = not old_val
            
            delta = 0
            for (other_ci, other_sign) in reduced_var_clauses[var]:
                was_true = (old_val == (other_sign > 0))
                
                if was_true:
                    sat_count[other_ci] -= 1
                    if sat_count[other_ci] == 0:
                        if not unsat_mask[other_ci]:
                            unsat_mask[other_ci] = True
                            unsat_count += 1
                            delta += 1
                else:
                    sat_count[other_ci] += 1
                    if sat_count[other_ci] == 1:
                        if unsat_mask[other_ci]:
                            unsat_mask[other_ci] = False
                            unsat_count -= 1
                            delta -= 1
            
            new_unsat = unsat_count
            
            if new_unsat < unsat_count:
                pass # Accept
            else:
                if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                    pass # Accept
                else:
                    # Reject: Revert
                    assignment[var] = old_val
                    for (other_ci, other_sign) in reduced_var_clauses[var]:
                        was_true = (old_val == (other_sign > 0))
                        if was_true:
                            sat_count[other_ci] += 1
                            if sat_count[other_ci] == 1:
                                unsat_mask[other_ci] = False
                                unsat_count -= 1
                        else:
                            sat_count[other_ci] -= 1
                            if sat_count[other_ci] == 0:
                                unsat_mask[other_ci] = True
                                unsat_count += 1
                    unsat_count = int(np.sum(unsat_mask)) # Resync to be safe
                    
            # Cool down
            # Simple linear decay in time?
            elapsed = time.time() - current_time
            T = T0 * (T_end / T0) ** (elapsed / max_time)
            
        return best_assignment.tolist(), False

    result2, solved2 = _run_sa(random.Random(123), sa_time)
    
    if solved2:
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assignment[var] = result2[var_to_reduced_idx[var]]
            else:
                full_assignment[var] = False
        return full_assignment

    # Return best result found
    full_assignment = [False] * n
    for var in range(n):
        if var in fixed:
            full_assignment[var] = fixed[var]
        elif var in var_to_reduced_idx:
            full_assignment[var] = result[var_to_reduced_idx[var]]
        else:
            full_assignment[var] = False
    return full_assignment

# EVOLVE-BLOCK-END
