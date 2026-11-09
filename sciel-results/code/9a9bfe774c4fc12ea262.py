import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 295
    
    # Precomputation
    var_clauses = [[] for _ in range(n)]
    clauses_store = []
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append((i, sign))
        clauses_store.append(cl_store)
    
    m = len(clauses)
    
    def is_satisfied(assignment, clause_idx):
        for v, sign in clauses_store[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt
    
    def get_unsat_clauses(assignment):
        return [i for i in range(m) if not is_satisfied(assignment, i)]
    
    def maxsat_local_search(rng, max_steps=100000, penalty_start=1.0, penalty_end=50.0, walk_prob_start=0.5, walk_prob_end=0.1):
        """
        Max-SAT formulation with dynamic penalty weight.
        Objective: minimize sum(penalty_i * unsat_i) where penalty_i increases over time.
        This guides search toward satisfying all clauses by making unsatisfied clauses
        progressively more expensive to leave unsatisfied.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track per-clause unsat status and penalty
        unsat_flags = [False] * m
        for i in range(m):
            unsat_flags[i] = not is_satisfied(assignment, i)
        
        unsat_count = sum(unsat_flags)
        if unsat_count == 0:
            return assignment
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        # Per-clause penalty weights (dynamic)
        base_penalty = [1.0] * m
        
        # Track how long each clause has been unsatisfied (for dynamic penalty)
        unsat_duration = [0] * m
        
        total_steps = max_steps
        no_improve = 0
        max_no_improve = 8000
        
        for step in range(total_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            # Compute current penalty scale (increases over time)
            progress = step / total_steps
            penalty_scale = penalty_start + (penalty_end - penalty_start) * progress
            
            # Dynamic walk probability: start high (explore), end low (exploit)
            current_walk_prob = walk_prob_start + (walk_prob_end - walk_prob_start) * progress
            
            # Update unsat durations
            for i in range(m):
                if unsat_flags[i]:
                    unsat_duration[i] += 1
            
            # Pick a random unsatisfied clause
            unsat_indices = [i for i in range(m) if unsat_flags[i]]
            if not unsat_indices:
                return assignment
            
            # Weighted selection: clauses that have been unsat longer get higher penalty
            # Use soft-max sampling for smooth selection
            weights = []
            for i in unsat_indices:
                w = base_penalty[i] * penalty_scale * (1.0 + 0.1 * unsat_duration[i])
                weights.append(w)
            
            total_w = sum(weights)
            if total_w <= 0:
                cl_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            else:
                r = rng.random() * total_w
                cumsum = 0.0
                cl_idx = unsat_indices[-1]
                for idx, w in zip(unsat_indices, weights):
                    cumsum += w
                    if r <= cumsum:
                        cl_idx = idx
                        break
            
            clause_vars = clauses_store[cl_idx]
            
            # Choose variable to flip
            if rng.random() < (1.0 - current_walk_prob):
                # Greedy: pick variable with best weighted delta
                best_var = -1
                best_delta = float('inf')
                for v, sign in clause_vars:
                    delta = 0.0
                    # This clause is currently unsat
                    if assignment[v] != (sign == 1):
                        # Flipping satisfies this clause: gain = penalty for this clause
                        delta -= base_penalty[cl_idx] * penalty_scale
                    else:
                        delta += base_penalty[cl_idx] * penalty_scale
                    
                    # Check other clauses containing v
                    for c_idx, c_sign in var_clauses[v]:
                        if c_idx == cl_idx:
                            continue
                        currently_satisfies = (assignment[v] == (c_sign == 1))
                        if currently_satisfies:
                            # Flipping may make this clause unsat
                            other_satisfy = False
                            for v2, s2 in clauses_store[c_idx]:
                                if v2 == v:
                                    continue
                                if assignment[v2] == (s2 == 1):
                                    other_satisfy = True
                                    break
                            if not other_satisfy:
                                delta += base_penalty[c_idx] * penalty_scale
                        else:
                            # Flipping satisfies this clause
                            delta -= base_penalty[c_idx] * penalty_scale
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                v_to_flip = best_var if best_var != -1 else clause_vars[rng.randint(0, 2)][0]
            else:
                # Random walk
                v_to_flip = clause_vars[rng.randint(0, 2)][0]
            
            # Apply flip and update
            old_unsat_count = unsat_count
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            new_unsat_count = old_unsat_count
            for c_idx, c_sign in var_clauses[v_to_flip]:
                was_unsat = unsat_flags[c_idx]
                now_sat = is_satisfied(assignment, c_idx)
                if was_unsat and now_sat:
                    unsat_flags[c_idx] = False
                    unsat_duration[c_idx] = 0
                    new_unsat_count -= 1
                elif not was_unsat and not now_sat:
                    unsat_flags[c_idx] = True
                    new_unsat_count += 1
            
            unsat_count = new_unsat_count
            delta = new_unsat_count - old_unsat_count
            
            # Acceptance: always accept improvements; accept worsening with probability
            # based on the weighted penalty change
            if delta <= 0:
                if new_unsat_count < best_unsat:
                    best_unsat = new_unsat_count
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Probability of accepting a worsening move decreases with penalty_scale
                # and the magnitude of the worsening
                accept_prob = math.exp(-delta * penalty_scale * 0.3)
                if rng.random() < accept_prob:
                    no_improve += 1
                else:
                    # Revert
                    assignment[v_to_flip] = not assignment[v_to_flip]
                    for c_idx, c_sign in var_clauses[v_to_flip]:
                        was_unsat_after_flip = unsat_flags[c_idx]
                        now_sat_after_revert = is_satisfied(assignment, c_idx)
                        if was_unsat_after_flip and now_sat_after_revert:
                            unsat_flags[c_idx] = False
                            unsat_duration[c_idx] = 0
                        elif not was_unsat_after_flip and not now_sat_after_revert:
                            unsat_flags[c_idx] = True
                    no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        if best_unsat == 0:
            return best_assignment
        return None
    

    
    def dpll_with_propagation(rng, depth_limit=25):
        """
        DPLL with unit propagation and pure literal elimination.
        """
        def unit_propagate(assignment, clauses_to_check):
            """
            Perform unit propagation. Returns (new_assignment, remaining_clauses, contradiction).
            """
            new_assignment = assignment.copy()
            remaining = list(clauses_to_check)
            changed = True
            while changed:
                changed = False
                new_remaining = []
                for clause in remaining:
                    sat = False
                    unassigned = []
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                        else:
                            unassigned.append((v, sign))
                    
                    if sat:
                        continue
                    
                    if len(unassigned) == 0:
                        return None, None, True
                    
                    if len(unassigned) == 1:
                        v, sign = unassigned[0]
                        new_assignment[v] = (sign == 1)
                        changed = True
                    else:
                        new_remaining.append(clause)
                
                remaining = new_remaining
            
            return new_assignment, remaining, False
        
        def pure_literal_eliminate(assignment, clauses_to_check):
            """
            Eliminate pure literals. Returns (new_assignment, remaining_clauses, contradiction).
            """
            new_assignment = assignment.copy()
            remaining = list(clauses_to_check)
            changed = True
            while changed:
                changed = False
                # Find pure literals
                var_polarity = {}
                for clause in remaining:
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            continue
                        if v not in var_polarity:
                            var_polarity[v] = set()
                        var_polarity[v].add(sign)
                
                for v, signs in var_polarity.items():
                    if len(signs) == 1:
                        # Pure literal
                        sign = signs.pop()
                        new_assignment[v] = (sign == 1)
                        changed = True
                
                if not changed:
                    break
                
                # Re-filter clauses
                new_remaining = []
                for clause in remaining:
                    sat = False
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        new_remaining.append(clause)
                remaining = new_remaining
            
            return new_assignment, remaining, False
        
        def dpll_recursive(assignment, clauses_remaining, depth):
            if depth > depth_limit:
                return None
            
            # Check if all clauses are satisfied
            all_sat = True
            for clause in clauses_remaining:
                sat = False
                for v, sign in clause:
                    if assignment[v] is not None:
                        if assignment[v] == (sign == 1):
                            sat = True
                            break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                return assignment
            
            # Unit propagation
            new_assignment, remaining_clauses, contradiction = unit_propagate(assignment, clauses_remaining)
            if contradiction:
                return None
            
            if remaining_clauses is None or len(remaining_clauses) == 0:
                all_sat = True
                for clause in clauses_remaining:
                    sat = False
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    return new_assignment
                else:
                    return None
            
            # Pure literal elimination
            new_assignment, remaining_clauses, contradiction = pure_literal_eliminate(new_assignment, remaining_clauses)
            if contradiction:
                return None
            
            if remaining_clauses is None or len(remaining_clauses) == 0:
                all_sat = True
                for clause in remaining_clauses:
                    sat = False
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    return new_assignment
                else:
                    return None
            
            # Pick a variable to branch on
            var_counts = {}
            for clause in remaining_clauses:
                for v, sign in clause:
                    if new_assignment[v] is None:
                        var_counts[v] = var_counts.get(v, 0) + 1
            
            if not var_counts:
                return None
            
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Try both assignments
            for val in [True, False]:
                test_assignment = new_assignment.copy()
                test_assignment[var] = val
                
                filtered_clauses = []
                for clause in remaining_clauses:
                    sat = False
                    for v, sign in clause:
                        if test_assignment[v] is not None:
                            if test_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        filtered_clauses.append(clause)
                
                result = dpll_recursive(test_assignment, filtered_clauses, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        initial_assignment = [None] * n
        result = dpll_recursive(initial_assignment, clauses_store, 0)
        
        if result is not None:
            final = []
            for v in range(n):
                if result[v] is None:
                    final.append(rng.random() < 0.5)
                else:
                    final.append(result[v])
            return final
        
        return None
    
    # Strategy 1: Try DPLL with limited depth first (fast for easy instances)
    rng = random.Random(42)
    
    sol = dpll_with_propagation(rng, depth_limit=20)
    if sol is not None:
        if count_unsat(sol) == 0:
            return sol
    
    # Strategy 2: Max-SAT local search with dynamic penalty weights
    # Multiple restarts with varying penalty schedules
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary penalty schedule across restarts
        penalty_start = 0.5 + r * 0.1
        penalty_end = 20.0 + r * 1.0
        walk_prob_start = 0.5 - (r % 5) * 0.05
        walk_prob_end = 0.1 + (r % 3) * 0.03
        max_steps = 80000 + r * 500
        
        sol = maxsat_local_search(
            rng, 
            max_steps=max_steps,
            penalty_start=penalty_start,
            penalty_end=penalty_end,
            walk_prob_start=walk_prob_start,
            walk_prob_end=walk_prob_end
        )
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol
    
    # Strategy 3: Higher penalty end values for harder instances
    for r in range(30):
        if time.time() > deadline:
            break
        
        penalty_start = 1.0 + r * 0.2
        penalty_end = 50.0 + r * 2.0
        walk_prob_start = 0.4
        walk_prob_end = 0.05
        max_steps = 60000
        
        sol = maxsat_local_search(
            rng,
            max_steps=max_steps,
            penalty_start=penalty_start,
            penalty_end=penalty_end,
            walk_prob_start=walk_prob_start,
            walk_prob_end=walk_prob_end
        )
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol
    
    # Last resort: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
