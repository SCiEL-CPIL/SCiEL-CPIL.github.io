import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    # Safety margin: 330s kill limit, use 315s for solving
    deadline = start_time + 315

    # Precomputation: Convert to 0-based indices and store signs
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

    def fast_dpll(rng, depth_limit=50):
        """
        DPLL with watched literals for efficient unit propagation.
        Returns assignment list or None.
        """
        assignment = [None] * n
        
        # Watched literals: for each clause, track two unassigned literals
        # watches[clause_idx] = (lit1, lit2) where lit is (var, sign)
        # active_clauses: set of clause indices that are not yet satisfied
        watches = {}
        active_clauses = set()
        
        # Initialize watches
        for i in range(m):
            cl = clauses_store[i]
            watches[i] = (cl[0], cl[1]) if len(cl) >= 2 else (cl[0], cl[0])
            active_clauses.add(i)
        
        # Clause status check
        def clause_status(clause, assignment):
            unassigned_count = 0
            for v, sign in clause:
                if assignment[v] is not None:
                    if assignment[v] == (sign == 1):
                        return 1
                else:
                    unassigned_count += 1
            if unassigned_count == 0:
                return -1
            return 0

        def find_new_watch(clause_idx, assignment, exclude_var):
            """Find a new unassigned literal in the clause, excluding exclude_var."""
            for v, sign in clauses_store[clause_idx]:
                if v != exclude_var and assignment[v] is None:
                    return (v, sign)
            return None

        def propagate_unit():
            """
            Perform unit propagation using watched literals.
            Returns True if success, False if contradiction.
            Modifies assignment and active_clauses in place.
            """
            queue = []
            
            # Find all unit clauses initially
            for ci in list(active_clauses):
                cl = clauses_store[ci]
                unassigned = []
                for v, sign in cl:
                    if assignment[v] is None:
                        unassigned.append((v, sign))
                    elif assignment[v] == (sign == 1):
                        # Satisfied, remove from active
                        active_clauses.discard(ci)
                        break
                else:
                    if len(unassigned) == 0:
                        return False  # Contradiction
                    if len(unassigned) == 1:
                        queue.append((ci, unassigned[0]))
            
            while queue:
                ci, unit_lit = queue.pop()
                v, sign = unit_lit
                assignment[v] = (sign == 1)
                
                # Update watches for all clauses containing v
                for other_ci, other_sign in var_clauses[v]:
                    if other_ci not in active_clauses:
                        continue
                    
                    # Check if v is a watched literal for other_ci
                    w = watches[other_ci]
                    if w[0][0] == v or w[1][0] == v:
                        # Check if other_ci is satisfied
                        is_sat = False
                        for vv, ss in clauses_store[other_ci]:
                            if assignment[vv] is not None and assignment[vv] == (ss == 1):
                                is_sat = True
                                break
                        
                        if is_sat:
                            active_clauses.discard(other_ci)
                            continue
                        
                        # Need to find a new watch
                        new_watch = find_new_watch(other_ci, assignment, v)
                        if new_watch is None:
                            # This clause is now unit or empty
                            unassigned = []
                            for vv, ss in clauses_store[other_ci]:
                                if assignment[vv] is None:
                                    unassigned.append((vv, ss))
                            if len(unassigned) == 0:
                                return False  # Contradiction
                            elif len(unassigned) == 1:
                                queue.append((other_ci, unassigned[0]))
                            else:
                                watches[other_ci] = (unassigned[0], unassigned[1])
                        else:
                            # Replace the watched literal that was assigned
                            if w[0][0] == v:
                                watches[other_ci] = (new_watch, w[1])
                            else:
                                watches[other_ci] = (w[0], new_watch)
            
            return True

        def pure_literal_elimination():
            """
            Apply pure literal elimination.
            Returns True if any pure literals were assigned.
            """
            var_signs = {}
            for ci in active_clauses:
                for v, sign in clauses_store[ci]:
                    if assignment[v] is None:
                        if v not in var_signs:
                            var_signs[v] = set()
                        var_signs[v].add(sign)
            
            pure_vars = []
            for v, signs in var_signs.items():
                if len(signs) == 1:
                    pure_vars.append((v, list(signs)[0]))
            
            if not pure_vars:
                return False
            
            for v, sign in pure_vars:
                assignment[v] = (sign == 1)
            
            # Remove satisfied clauses from active set
            for v, sign in pure_vars:
                for ci, c_sign in var_clauses[v]:
                    if ci in active_clauses:
                        active_clauses.discard(ci)
            
            return True

        def dpll_recursive(depth):
            if depth > depth_limit:
                return None
            
            if time.time() > deadline:
                return None

            # Propagate
            if not propagate_unit():
                return None
            
            if not active_clauses:
                final_assignment = assignment.copy()
                for i in range(n):
                    if final_assignment[i] is None:
                        final_assignment[i] = rng.random() < 0.5
                return final_assignment

            # Pure literal elimination
            if pure_literal_elimination():
                # Re-check after pure literal assignment
                if not propagate_unit():
                    return None
                if not active_clauses:
                    final_assignment = assignment.copy()
                    for i in range(n):
                        if final_assignment[i] is None:
                            final_assignment[i] = rng.random() < 0.5
                    return final_assignment

            # Variable Selection Heuristic: Pick variable appearing in most active clauses
            var_freq = {}
            for ci in active_clauses:
                for v, sign in clauses_store[ci]:
                    if assignment[v] is None:
                        var_freq[v] = var_freq.get(v, 0) + 1
            
            if not var_freq:
                final_assignment = assignment.copy()
                for i in range(n):
                    if final_assignment[i] is None:
                        final_assignment[i] = rng.random() < 0.5
                return final_assignment

            max_freq = max(var_freq.values())
            candidates = [v for v, c in var_freq.items() if c == max_freq]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Branch
            for val in [True, False]:
                # Save state
                saved_assignment = assignment.copy()
                saved_active = set(active_clauses)
                
                assignment[var] = val
                
                # Remove satisfied clauses
                for ci, c_sign in var_clauses[var]:
                    if ci in active_clauses:
                        if assignment[var] == (c_sign == 1):
                            active_clauses.discard(ci)
                
                # Check for contradictions (empty clauses)
                contradiction = False
                for ci in list(active_clauses):
                    cl = clauses_store[ci]
                    unassigned = 0
                    for v, sign in cl:
                        if assignment[v] is None:
                            unassigned += 1
                        elif assignment[v] != (sign == 1):
                            pass
                    if unassigned == 0:
                        contradiction = True
                        break
                
                if not contradiction:
                    result = dpll_recursive(depth + 1)
                    if result is not None:
                        return result
                
                # Restore state
                assignment[:] = saved_assignment
                active_clauses.clear()
                active_clauses.update(saved_active)
            
            return None

        return dpll_recursive(0)

    def min_conflicts_walksat(rng, max_steps=100000, initial_temp=5.0, cooling=0.9998, min_temp=0.01):
        """
        Min-Conflicts WalkSAT with incremental clause satisfaction tracking.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute clause satisfaction status
        clause_sat = [False] * m
        unsat_clauses = set()
        for i in range(m):
            if is_satisfied(assignment, i):
                clause_sat[i] = True
            else:
                unsat_clauses.add(i)
        
        current_unsat_count = len(unsat_clauses)
        if current_unsat_count == 0:
            return assignment
        
        best_unsat_count = current_unsat_count
        best_assignment = assignment.copy()
        
        temp = initial_temp
        no_improve = 0
        max_no_improve = 3000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            unsat_list = list(unsat_clauses)
            cl_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Find the variable in this clause that minimizes the increase in unsat count
            best_var = -1
            best_delta = 1000000
            best_delta_var_list = []
            
            for v, sign in clauses_store[cl_idx]:
                # Calculate delta if we flip v
                delta = 0
                
                # The chosen clause is currently unsat, flipping v satisfies it
                delta -= 1
                
                # Other clauses containing v
                for c_idx, c_sign in var_clauses[v]:
                    if c_idx == cl_idx:
                        continue
                    
                    if clause_sat[c_idx]:
                        # Currently satisfied. Check if v is the unique satisfier.
                        unique_satisfier = True
                        for v2, s2 in clauses_store[c_idx]:
                            if v2 == v:
                                continue
                            if assignment[v2] == (s2 == 1):
                                unique_satisfier = False
                                break
                        if unique_satisfier:
                            delta += 1
                    else:
                        # Currently unsat. Flipping v satisfies it.
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                    best_delta_var_list = [v]
                elif delta == best_delta:
                    best_delta_var_list.append(v)
            
            # Pick a variable from the best candidates
            v_to_flip = best_delta_var_list[rng.randint(0, len(best_delta_var_list) - 1)]
            
            # Perform Flip and update clause satisfaction
            old_count = current_unsat_count
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            for c_idx, c_sign in var_clauses[v_to_flip]:
                was_sat = clause_sat[c_idx]
                now_sat = is_satisfied(assignment, c_idx)
                
                if was_sat and not now_sat:
                    clause_sat[c_idx] = False
                    unsat_clauses.add(c_idx)
                elif not was_sat and now_sat:
                    clause_sat[c_idx] = True
                    unsat_clauses.discard(c_idx)
            
            new_count = len(unsat_clauses)
            delta = new_count - old_count
            
            # Acceptance Criteria (Simulated Annealing)
            if delta <= 0:
                current_unsat_count = new_count
                if new_count < best_unsat_count:
                    best_unsat_count = new_count
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if temp > min_temp:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_unsat_count = new_count
                        no_improve += 1
                    else:
                        # Reject
                        assignment[v_to_flip] = not assignment[v_to_flip]
                        for c_idx, c_sign in var_clauses[v_to_flip]:
                            was_sat_after = clause_sat[c_idx]
                            now_sat_after_revert = is_satisfied(assignment, c_idx)
                            if was_sat_after and not now_sat_after_revert:
                                clause_sat[c_idx] = False
                                unsat_clauses.add(c_idx)
                            elif not was_sat_after and now_sat_after_revert:
                                clause_sat[c_idx] = True
                                unsat_clauses.discard(c_idx)
                        current_unsat_count = old_count
                        no_improve += 1
                else:
                    # Reject
                    assignment[v_to_flip] = not assignment[v_to_flip]
                    for c_idx, c_sign in var_clauses[v_to_flip]:
                        was_sat_after = clause_sat[c_idx]
                        now_sat_after_revert = is_satisfied(assignment, c_idx)
                        if was_sat_after and not now_sat_after_revert:
                            clause_sat[c_idx] = False
                            unsat_clauses.add(c_idx)
                        elif not was_sat_after and now_sat_after_revert:
                            clause_sat[c_idx] = True
                            unsat_clauses.discard(c_idx)
                    current_unsat_count = old_count
                    no_improve += 1
            
            # Cooling
            temp = max(temp * cooling, min_temp)
            
            if no_improve > max_no_improve:
                break
        
        if best_unsat_count == 0:
            return best_assignment
        return None

    # Strategy Execution
    rng = random.Random(42)
    
    # 1. Try DPLL first with watched literals (fast for easy/structured instances)
    dpll_deadline = start_time + 15
    old_deadline = deadline
    deadline = dpll_deadline
    
    sol = fast_dpll(rng, depth_limit=50)
    
    deadline = old_deadline
    
    if sol is not None:
        if count_unsat(sol) == 0:
            return sol
    
    # 2. Try Min-Conflicts WalkSAT with multiple restarts
    remaining_time = deadline - time.time()
    if remaining_time <= 0:
        return [rng.random() < 0.5 for _ in range(n)]
    
    # Adaptive restart strategy
    restarts = 15
    time_per_restart = remaining_time / restarts
    
    for r in range(restarts):
        if time.time() > deadline - 2:
            break
        
        # Adaptive temperature: start high, decrease over restarts
        base_temp = 10.0
        temp = base_temp * (1.0 - 0.7 * r / restarts) + 0.5
        cooling = 0.9997
        max_steps = int(time_per_restart * 8000)
        
        sol = min_conflicts_walksat(rng, max_steps=max_steps, initial_temp=temp, cooling=cooling, min_temp=0.01)
        
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol

    # 3. Final fallback
    return [rng.random() < 0.5 for _ in range(n)]

if __name__ == "__main__":
    pass
