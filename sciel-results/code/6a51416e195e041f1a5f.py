import random
import time
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Explicit deadline guard: 300s wall-clock limit, use 285s as safety margin
    start_time = time.time()
    TIME_LIMIT = 285.0
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    if time_remaining() <= 0:
        return [True] * n

    m = len(clauses)
    
    # Precompute clause literals as (var_index, is_positive)
    # Also build variable-to-clauses index for fast gain calculation
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cd = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            cd.append((v, p))
            var_clauses[v].append(i)
        clause_data.append(cd)

    # State variables
    # assignment[v] = True/False
    # clause_sat[i] = number of satisfied literals in clause i (0 means unsatisfied)
    
    def init_state(assignment):
        clause_sat = [0] * m
        for i, cd in enumerate(clause_data):
            sat = 0
            for v, p in cd:
                if assignment[v] == p:
                    sat += 1
            clause_sat[i] = sat
        return clause_sat

    def count_unsat(clause_sat):
        return sum(1 for s in clause_sat if s == 0)

    def full_satisfied(clause_sat):
        # Check if any clause is unsatisfied
        for s in clause_sat:
            if s == 0:
                return False
        return True

    def gsat_search(seed, max_restarts, steps_per_restart, use_weighting=False):
        rng = random.Random(seed)
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        # Clause weights
        if use_weighting:
            weights = [1.0] * m
        
        for restart in range(max_restarts):
            if time_remaining() < 1.0:
                break
            
            # Random initialization
            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_sat = init_state(assignment)
            current_unsat = count_unsat(clause_sat)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            if current_unsat == 0:
                return assignment[:]

            if use_weighting:
                # Decay weights slightly on restart
                for i in range(m):
                    weights[i] = max(1.0, weights[i] * 0.95)

            for step in range(steps_per_restart):
                if time_remaining() < 1.0:
                    break
                
                if current_unsat == 0:
                    return assignment[:]

                # Find unsatisfied clauses
                unsat_indices = [i for i, s in enumerate(clause_sat) if s == 0]
                
                if not unsat_indices:
                    return assignment[:]

                # Select a clause to focus on
                # Weighted selection if enabled, else random
                if use_weighting:
                    unsat_weights = [weights[i] for i in unsat_indices]
                    total_w = sum(unsat_weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen_idx = unsat_indices[-1]
                    for j, w in enumerate(unsat_weights):
                        cum += w
                        if r <= cum:
                            chosen_idx = unsat_indices[j]
                            break
                else:
                    chosen_idx = rng.choice(unsat_indices)
                
                if use_weighting:
                    weights[chosen_idx] *= 1.05

                clause = clause_data[chosen_idx]
                
                # Select variable to flip
                # Strategy: Min-Conflicts with Random Tie-Breaking
                # Calculate gain for each variable in the clause
                best_var = -1
                best_gain = -1000
                best_flip_val = None
                candidates = []
                
                for v, p in clause:
                    # Calculate gain if we flip v
                    gain = 0
                    for c_idx in var_clauses[v]:
                        c = clause_data[c_idx]
                        current_sat = clause_sat[c_idx]
                        
                        # Find the polarity of v in clause c
                        lit_pol = None
                        for vv, pp in c:
                            if vv == v:
                                lit_pol = pp
                                break
                        
                        current_lit_sat = (assignment[v] == lit_pol)
                        
                        if current_sat == 0:
                            # Clause is unsatisfied. All literals are False.
                            # So current_lit_sat must be False.
                            # Flipping v makes literal True.
                            # Clause becomes satisfied (sat_count 1).
                            gain += 1
                        else:
                            # Clause is satisfied.
                            if current_lit_sat:
                                # Literal is True. Flipping makes it False.
                                # sat_count decreases by 1.
                                if current_sat == 1:
                                    # Becomes unsatisfied
                                    gain -= 1
                            else:
                                # Literal is False. Flipping makes it True.
                                # sat_count increases. Clause stays satisfied.
                                gain += 0
                    
                    # Add small noise to break ties and encourage exploration
                    noise = rng.gauss(0, 0.1)
                    total_gain = gain + noise
                    
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_var = v
                        best_flip_val = not assignment[v]
                        candidates = [(best_gain, best_var, best_flip_val)]
                    elif abs(total_gain - best_gain) < 1e-9:
                        candidates.append((total_gain, best_var, best_flip_val))
                
                if not candidates:
                    # Fallback (should not happen)
                    v, p = rng.choice(clause)
                    assignment[v] = not assignment[v]
                    # Update clause_sat
                    for c_idx in var_clauses[v]:
                        c = clause_data[c_idx]
                        old_sat = clause_sat[c_idx]
                        new_sat = 0
                        for vv, pp in c:
                            if assignment[vv] == pp:
                                new_sat += 1
                        clause_sat[c_idx] = new_sat
                        if old_sat == 0 and new_sat > 0:
                            current_unsat -= 1
                        elif old_sat > 0 and new_sat == 0:
                            current_unsat += 1
                else:
                    # Pick randomly from top candidates to add stochasticity
                    _, v_to_flip, new_val = candidates[0]
                    
                    # Apply flip
                    old_val = assignment[v_to_flip]
                    assignment[v_to_flip] = new_val
                    
                    # Update clause_sat and current_unsat
                    for c_idx in var_clauses[v_to_flip]:
                        c = clause_data[c_idx]
                        old_sat = clause_sat[c_idx]
                        
                        # Determine if the literal in this clause changed satisfaction
                        lit_pol = None
                        for vv, pp in c:
                            if vv == v_to_flip:
                                lit_pol = pp
                                break
                        
                        old_lit_sat = (old_val == lit_pol)
                        new_lit_sat = (new_val == lit_pol)
                        
                        if old_lit_sat and not new_lit_sat:
                            # Literal became False
                            new_sat = old_sat - 1
                        elif not old_lit_sat and new_lit_sat:
                            # Literal became True
                            new_sat = old_sat + 1
                        else:
                            new_sat = old_sat
                        
                        # Update global unsat count
                        if old_sat == 0 and new_sat > 0:
                            current_unsat -= 1
                        elif old_sat > 0 and new_sat == 0:
                            current_unsat += 1
                        
                        clause_sat[c_idx] = new_sat

            # End of steps for this restart
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            if current_unsat == 0:
                return assignment[:]

        return best_assignment

    def simulated_annealing_search(seed, max_steps, initial_temp=1.0, cooling_rate=0.999):
        """
        Simulated Annealing approach to escape local minima.
        Accepts worsening moves with probability exp(-delta_E / T).
        """
        rng = random.Random(seed)
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat = init_state(assignment)
        current_unsat = count_unsat(clause_sat)
        
        best_assignment = assignment[:]
        best_unsat_count = current_unsat
        
        temp = initial_temp
        
        for step in range(max_steps):
            if time_remaining() < 1.0:
                break
            
            if current_unsat == 0:
                return assignment[:]
            
            # Pick a random variable to flip (or from unsat clauses for bias)
            # Strategy: 50% chance pick from an unsat clause, 50% chance pick any variable
            if rng.random() < 0.5 and current_unsat > 0:
                # Pick a random unsat clause
                unsat_indices = [i for i, s in enumerate(clause_sat) if s == 0]
                if unsat_indices:
                    chosen_clause_idx = rng.choice(unsat_indices)
                    clause = clause_data[chosen_clause_idx]
                    v, p = rng.choice(clause)
                else:
                    v = rng.randint(0, n - 1)
            else:
                v = rng.randint(0, n - 1)
            
            # Flip variable v
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            
            # Calculate new unsat count (or delta)
            # For efficiency, we can just recalculate the affected clauses
            old_unsat = current_unsat
            
            # Update clause_sat for affected clauses
            new_unsat = old_unsat
            for c_idx in var_clauses[v]:
                c = clause_data[c_idx]
                old_sat = clause_sat[c_idx]
                
                lit_pol = None
                for vv, pp in c:
                    if vv == v:
                        lit_pol = pp
                        break
                
                old_lit_sat = (old_val == lit_pol)
                new_lit_sat = (new_val == lit_pol)
                
                if old_lit_sat and not new_lit_sat:
                    new_sat = old_sat - 1
                elif not old_lit_sat and new_lit_sat:
                    new_sat = old_sat + 1
                else:
                    new_sat = old_sat
                
                # Update global unsat count
                if old_sat == 0 and new_sat > 0:
                    new_unsat -= 1
                elif old_sat > 0 and new_sat == 0:
                    new_unsat += 1
                
                clause_sat[c_idx] = new_sat
            
            current_unsat = new_unsat
            
            # Accept or reject
            if current_unsat <= best_unsat_count:
                # Always accept if it's the best so far
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            elif current_unsat < old_unsat:
                # Always accept if it improves current state
                pass
            else:
                # Worsening move: accept with probability exp(-delta / T)
                delta = current_unsat - old_unsat
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        # Accept
                        pass
                    else:
                        # Reject: revert
                        assignment[v] = old_val
                        for c_idx in var_clauses[v]:
                            c = clause_data[c_idx]
                            lit_pol = None
                            for vv, pp in c:
                                if vv == v:
                                    lit_pol = pp
                                    break
                            old_lit_sat = (old_val == lit_pol)
                            new_lit_sat = (new_val == lit_pol)
                            # Revert clause_sat
                            if old_lit_sat and not new_lit_sat:
                                clause_sat[c_idx] += 1
                            elif not old_lit_sat and new_lit_sat:
                                clause_sat[c_idx] -= 1
                        current_unsat = old_unsat
            
            # Cool down
            temp *= cooling_rate
            
            if best_unsat_count == 0:
                return best_assignment[:]
        
        return best_assignment

    # Strategy Portfolio
    best_result = None
    best_score = float('inf')

    # Parameters tuned for n=50
    # High restart count, moderate steps
    seeds = [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680, 111, 222, 333, 444, 555]
    
    for i, seed in enumerate(seeds):
        if time_remaining() < 5.0:
            break
        
        # Adaptive steps: more steps for earlier seeds? Or keep constant?
        # Let's use a mix of short and long runs.
        if i % 3 == 0:
            steps = 500
            restarts = 20
        elif i % 3 == 1:
            steps = 1000
            restarts = 10
        else:
            steps = 200
            restarts = 50
            
        use_weighting = (i % 2 == 0)
        
        result = gsat_search(seed, restarts, steps, use_weighting)
        
        if result is not None:
            # Verify
            clause_sat = init_state(result)
            unsat_cnt = count_unsat(clause_sat)
            
            if unsat_cnt == 0:
                return result
            else:
                if unsat_cnt < best_score:
                    best_score = unsat_cnt
                    best_result = result

    # If we still have time, try Simulated Annealing
    if time_remaining() > 15.0:
        for seed in [77, 88, 99, 100, 101, 102]:
            if time_remaining() < 2.0:
                break
            
            # Run SA for a fixed number of steps
            max_steps = 5000
            result = simulated_annealing_search(seed, max_steps, initial_temp=1.0, cooling_rate=0.9995)
            
            if result is not None:
                clause_sat = init_state(result)
                unsat_cnt = count_unsat(clause_sat)
                if unsat_cnt == 0:
                    return result
                if unsat_cnt < best_score:
                    best_score = unsat_cnt
                    best_result = result

    # If we still have time, try a pure random walk with very many short restarts
    if time_remaining() > 10.0:
        for seed in [77, 88, 99, 100, 101, 102]:
            if time_remaining() < 2.0:
                break
            result = gsat_search(seed, 100, 100, use_weighting=False)
            if result is not None:
                clause_sat = init_state(result)
                unsat_cnt = count_unsat(clause_sat)
                if unsat_cnt == 0:
                    return result
                if unsat_cnt < best_score:
                    best_score = unsat_cnt
                    best_result = result

    if best_result is not None:
        return best_result
        
    return [True] * n
