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
    
    def walk_sat_increased(rng, max_steps=80000, initial_temp=8.0, cooling=0.9997, min_temp=0.005, walk_prob=0.3):
        """
        WalkSAT with simulated annealing using incremental unsat tracking.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat set
        unsat_set = set(get_unsat_clauses(assignment))
        if not unsat_set:
            return assignment
        
        best_unsat = len(unsat_set)
        best_assignment = assignment.copy()
        
        temp = initial_temp
        no_improve = 0
        max_no_improve = 5000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick a random unsatisfied clause
            cl_idx = next(iter(unsat_set))
            # Random pick from unsat set
            unsat_list = list(unsat_set)
            cl_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Choose a variable to flip from this clause
            clause_vars = clauses_store[cl_idx]
            
            if rng.random() < (1.0 - walk_prob):
                # Best improvement heuristic: pick variable with best delta
                best_var = -1
                best_delta = 1000000
                for v, sign in clause_vars:
                    delta = 0
                    # This clause is currently unsat
                    if assignment[v] != (sign == 1):
                        delta -= 1  # Flipping satisfies this clause
                    else:
                        delta += 1  # Flipping unsatisfies (but clause already unsat)
                    
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
                                delta += 1
                        else:
                            # Flipping satisfies this clause
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                v_to_flip = best_var if best_var != -1 else clause_vars[rng.randint(0, 2)][0]
            else:
                # Random walk: pick random variable from clause
                v_to_flip = clause_vars[rng.randint(0, 2)][0]
            
            # Compute delta by checking affected clauses
            old_unsat_count = len(unsat_set)
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat set incrementally
            new_unsat_count = old_unsat_count
            # Check all clauses containing v_to_flip
            for c_idx, c_sign in var_clauses[v_to_flip]:
                was_unsat = c_idx in unsat_set
                now_sat = is_satisfied(assignment, c_idx)
                if was_unsat and now_sat:
                    unsat_set.discard(c_idx)
                    new_unsat_count -= 1
                elif not was_unsat and not now_sat:
                    unsat_set.add(c_idx)
                    new_unsat_count += 1
            
            delta = new_unsat_count - old_unsat_count
            
            # Simulated annealing acceptance
            if delta <= 0:
                if new_unsat_count < best_unsat:
                    best_unsat = new_unsat_count
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if temp > min_temp:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        no_improve += 1
                    else:
                        # Reject, revert
                        assignment[v_to_flip] = not assignment[v_to_flip]
                        # Revert unsat set
                        for c_idx, c_sign in var_clauses[v_to_flip]:
                            was_unsat_after_flip = c_idx in unsat_set
                            now_sat_after_revert = is_satisfied(assignment, c_idx)
                            if was_unsat_after_flip and now_sat_after_revert:
                                unsat_set.discard(c_idx)
                            elif not was_unsat_after_flip and not now_sat_after_revert:
                                unsat_set.add(c_idx)
                        no_improve += 1
                else:
                    # Reject, revert
                    assignment[v_to_flip] = not assignment[v_to_flip]
                    for c_idx, c_sign in var_clauses[v_to_flip]:
                        was_unsat_after_flip = c_idx in unsat_set
                        now_sat_after_revert = is_satisfied(assignment, c_idx)
                        if was_unsat_after_flip and now_sat_after_revert:
                            unsat_set.discard(c_idx)
                        elif not was_unsat_after_flip and not now_sat_after_revert:
                            unsat_set.add(c_idx)
                    no_improve += 1
            
            # Cool down
            temp = max(temp * cooling, min_temp)
            
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
    
    # Strategy 2: WalkSAT with simulated annealing (incremental tracking)
    max_restarts = 40
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        initial_temp = 4.0 + r * 0.4
        cooling = 0.9997 - r * 0.000005
        max_steps = 50000 + r * 500
        walk_prob = 0.25 + (r % 5) * 0.05
        
        sol = walk_sat_increased(rng, max_steps=max_steps, initial_temp=initial_temp, cooling=cooling, walk_prob=walk_prob)
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol
    
    # Strategy 3: More aggressive WalkSAT with higher walk probability
    for r in range(20):
        if time.time() > deadline:
            break
        
        initial_temp = 6.0 + r * 0.3
        cooling = 0.9998
        max_steps = 40000
        walk_prob = 0.4 + r * 0.02
        
        sol = walk_sat_increased(rng, max_steps=max_steps, initial_temp=initial_temp, cooling=cooling, walk_prob=walk_prob)
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol
    
    # Last resort: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
