"""Simulated-annealing WalkSAT with incremental clause tracking and adaptive restarts."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s wall clock, keep 15s margin
    deadline = time.time() + 285.0
    
    # Precompute clause data for speed
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_in_c)
        clause_signs.append(signs)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in clause_vars[ci]:
            var_to_clauses[v].append(ci)
    
    # For each clause, precompute which position each variable occupies
    # clause_var_pos[ci][v] = position of v in clause ci (or -1 if not present)
    # This is useful for fast satisfaction checks
    
    def is_clause_satisfied(ci, assignment):
        """Check if clause ci is satisfied by assignment."""
        vs = clause_vars[ci]
        ss = clause_signs[ci]
        for i in range(len(vs)):
            if assignment[vs[i]] == (ss[i] > 0):
                return True
        return False
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for ci in range(m):
            if not is_clause_satisfied(ci, assignment):
                cnt += 1
        return cnt
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Simulated annealing parameters
    # Temperature schedule: T = T0 * cooling_factor^(step)
    # Accept worse moves with probability exp(-delta_cost / T)
    
    max_restarts = 1000
    base_max_steps = 20000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track unsatisfied clauses incrementally
        # unsat_clauses: set of clause indices that are currently unsatisfied
        unsat_clauses = set()
        for ci in range(m):
            if not is_clause_satisfied(ci, assignment):
                unsat_clauses.add(ci)
        
        unsat_count = len(unsat_clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Simulated annealing parameters
        # Higher temperature early, cooling over time
        T0 = 3.0
        cooling_factor = 0.9995
        # Adaptive: if stuck, increase temperature
        T = T0
        
        # Number of steps for this restart
        # Scale with problem size and restart number
        max_steps = base_max_steps
        if restart > 0:
            # Gradually increase search effort
            max_steps = min(base_max_steps * (1 + restart * 0.05), 100000)
        
        stuck_counter = 0
        best_in_restart = unsat_count
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            # Update temperature
            T *= cooling_factor
            
            # If stuck, reheat
            if unsat_count > best_in_restart:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            if stuck_counter > 500:
                # Reheat
                T = T0 * 0.5
                stuck_counter = 0
            
            # Pick an unsatisfied clause
            if unsat_count == 0:
                return assignment
            
            # Randomly pick from unsatisfied clauses
            # For efficiency, pick a random clause and check
            # If satisfied, try a few more times, then scan
            ci = None
            for _ in range(5):
                cand = random.randint(0, m - 1)
                if cand in unsat_clauses:
                    ci = cand
                    break
            
            if ci is None:
                # Pick from unsat_clauses directly
                unsat_list = list(unsat_clauses)
                if not unsat_list:
                    return assignment
                ci = random.choice(unsat_list)
            
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            
            # Choose a variable to flip
            # Use a mix of greedy and random
            # Probability of random flip
            p_random = 0.4 + 0.2 * (T / T0)  # More random at high temp
            
            if random.random() < p_random:
                # Random choice from clause variables
                idx = random.randint(0, len(vs) - 1)
            else:
                # Greedy: pick variable that minimizes unsat_count after flip
                # For each variable in the clause, compute the change in unsat count
                best_gain = -1000
                best_idx = 0
                for i in range(len(vs)):
                    v = vs[i]
                    # Compute gain from flipping v
                    gain = 0
                    for cj in var_to_clauses[v]:
                        # Check if clause cj is currently satisfied
                        cur_sat = (cj not in unsat_clauses)
                        
                        # Check if clause cj would be satisfied after flipping v
                        # If v is in clause cj, flipping v changes its satisfaction status
                        # Find position of v in clause cj
                        vpos = -1
                        for j in range(len(clause_vars[cj])):
                            if clause_vars[cj][j] == v:
                                vpos = j
                                break
                        
                        if vpos == -1:
                            # v not in this clause (shouldn't happen since v in var_to_clauses[cj])
                            new_sat = cur_sat
                        else:
                            # After flip, v has new value
                            new_val = not assignment[v]
                            # Check if clause cj is satisfied with new v value
                            new_sat = False
                            for j in range(len(clause_vars[cj])):
                                v2 = clause_vars[cj][j]
                                s2 = clause_signs[cj][j]
                                if v2 == v:
                                    if new_val == (s2 > 0):
                                        new_sat = True
                                        break
                                else:
                                    if assignment[v2] == (s2 > 0):
                                        new_sat = True
                                        break
                        
                        if not cur_sat and new_sat:
                            gain += 1
                        elif cur_sat and not new_sat:
                            gain -= 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                idx = best_idx
            
            v = vs[idx]
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            
            # Update unsat_clauses incrementally
            # Only clauses containing v can change satisfaction status
            for cj in var_to_clauses[v]:
                was_sat = (cj not in unsat_clauses)
                is_now_sat = is_clause_satisfied(cj, assignment)
                if was_sat and not is_now_sat:
                    unsat_clauses.add(cj)
                    unsat_count += 1
                elif not was_sat and is_now_sat:
                    unsat_clauses.discard(cj)
                    unsat_count -= 1
            
            if unsat_count == 0:
                return assignment
            
            # Simulated annealing acceptance
            if unsat_count > best_in_restart:
                # This is a worse move
                delta = unsat_count - best_in_restart
                if T > 0.01:
                    # Accept with probability exp(-delta / T)
                    if random.random() < math.exp(-delta / max(T, 0.001)):
                        # Accept
                        pass
                    else:
                        # Reject: undo the flip
                        assignment[v] = old_val
                        # Revert unsat_clauses
                        for cj in var_to_clauses[v]:
                            was_sat_before_undo = (cj not in unsat_clauses)
                            # After undo, v has old_val again
                            is_sat_after_undo = is_clause_satisfied(cj, assignment)
                            if was_sat_before_undo and not is_sat_after_undo:
                                unsat_clauses.add(cj)
                                unsat_count += 1
                            elif not was_sat_before_undo and is_sat_after_undo:
                                unsat_clauses.discard(cj)
                                unsat_count -= 1
                # If T is very low, effectively reject (already undone above if not accepted)
            else:
                # Better or equal move, always accept
                if unsat_count < best_in_restart:
                    best_in_restart = unsat_count
                    if unsat_count < best_unsat_count:
                        best_unsat_count = unsat_count
                        best_assignment = assignment[:]
        
        # After max_steps, check final state
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
