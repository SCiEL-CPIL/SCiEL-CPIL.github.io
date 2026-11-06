import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 320.0 # Safety margin

    # Pre-processing
    # var_to_clauses[v] = list of (clause_index, polarity)
    # polarity is True if literal is positive (v), False if negative (-v)
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = [[] for _ in range(m)]
    
    for i, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            var_to_clauses[v].append((i, p))
            clause_vars[i].append(v)

    # State
    assignment = [False] * n
    sat_count = [0] * m
    unsat_clauses = []
    clause_weights = [1.0] * m

    # Helper to initialize state
    def init_state(assignment):
        nonlocal unsat_clauses
        unsat_clauses = []
        for i in range(m):
            count = 0
            for lit in clauses[i]:
                v = abs(lit) - 1
                p = lit > 0
                if assignment[v] == p:
                    count += 1
            sat_count[i] = count
            if count == 0:
                unsat_clauses.append(i)

    # Helper to flip variable and update state
    def flip(v):
        assignment[v] = not assignment[v]
        for ci, p in var_to_clauses[v]:
            old_count = sat_count[ci]
            if (not assignment[v]) == p:
                # v was satisfying, now not
                sat_count[ci] = old_count - 1
                if old_count == 1:
                    unsat_clauses.append(ci)
            else:
                # v was not satisfying, now is
                sat_count[ci] = old_count + 1
                if old_count == 0:
                    # Clause becomes sat - lazy removal
                    pass

    # Helper to remove a clause from unsat_clauses efficiently (lazy)
    def pick_unsat_clause():
        while True:
            if not unsat_clauses:
                return None
            # Weighted random selection
            total_weight = sum(clause_weights[ci] for ci in unsat_clauses)
            if total_weight <= 0:
                idx = random.randrange(len(unsat_clauses))
                ci = unsat_clauses[idx]
            else:
                r = random.random() * total_weight
                cumulative = 0
                idx = 0
                for j, ci in enumerate(unsat_clauses):
                    cumulative += clause_weights[ci]
                    if r <= cumulative:
                        idx = j
                        break
                ci = unsat_clauses[idx]
            
            if sat_count[ci] == 0:
                return ci
            else:
                # It's satisfied, remove it
                unsat_clauses[idx] = unsat_clauses[-1]
                unsat_clauses.pop()

    # Unit propagation
    def unit_propagate():
        nonlocal assignment
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if sat_count[i] == 0:
                    # Find which variable can satisfy this clause
                    for lit in clauses[i]:
                        v = abs(lit) - 1
                        p = lit > 0
                        if assignment[v] != p:
                            # Flipping v would satisfy this clause
                            # Check if this is the only unsatisfied literal
                            unsat_count = 0
                            for lit2 in clauses[i]:
                                v2 = abs(lit2) - 1
                                p2 = lit2 > 0
                                if assignment[v2] != p2:
                                    unsat_count += 1
                            if unsat_count == 1:
                                flip(v)
                                changed = True
                                break
                elif sat_count[i] == 1:
                    # Check if any variable is the only satisfying literal
                    for lit in clauses[i]:
                        v = abs(lit) - 1
                        p = lit > 0
                        if assignment[v] == p:
                            # This variable is currently satisfying
                            # Check if it's the only one
                            sat_count_check = 0
                            for lit2 in clauses[i]:
                                v2 = abs(lit2) - 1
                                p2 = lit2 > 0
                                if assignment[v2] == p2:
                                    sat_count_check += 1
                            if sat_count_check == 1:
                                # This variable is critical - don't flip it
                                pass

    # Calculate gain and breakage with weights
    def calculate_gain_breakage(v):
        gain = 0.0
        breakage = 0.0
        for uci, p in var_to_clauses[v]:
            w = clause_weights[uci]
            if sat_count[uci] == 0:
                # Clause is unsat. If flipping v satisfies it, gain++
                if (not assignment[v]) == p:
                    gain += w
            else:
                # Clause is sat. If flipping v breaks it, breakage++
                if sat_count[uci] == 1:
                    if assignment[v] == p:
                        breakage += w
        return gain, breakage

    rng = random.Random(42)
    best_unsat = m + 1
    best_assignment = None

    # Main Loop
    max_steps_per_restart = 100000
    noise_prob = 0.2
    weight_boost = 1.1
    weight_decay = 0.95
    max_weight = 100.0
    min_weight = 0.1

    while time.time() < deadline - 5.0:
        # Random Restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        init_state(assignment)
        
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment[:]
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

        steps = 0
        stuck_counter = 0
        local_best_unsat = current_unsat
        local_best_assignment = assignment[:]
        stagnation_counter = 0

        while steps < max_steps_per_restart:
            if time.time() > deadline - 5.0:
                break
            
            if not unsat_clauses:
                return assignment[:]

            # Pick a weighted random unsatisfied clause
            ci = pick_unsat_clause()
            if ci is None:
                return assignment[:]

            # Evaluate variables in this clause
            best_v = -1
            best_score = -10**9
            cvars = clause_vars[ci]
            
            scores = []
            for v in cvars:
                gain, breakage = calculate_gain_breakage(v)
                score = gain - breakage
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_v = v

            # Noise: with probability noise_prob, pick random var from clause
            if rng.random() < noise_prob:
                v = rng.choice(cvars)
            else:
                v = best_v

            flip(v)
            steps += 1
            
            # Periodic checks and weight updates
            if steps % 20 == 0:
                current_unsat = sum(1 for i in range(m) if sat_count[i] == 0)
                
                if current_unsat == 0:
                    return assignment[:]
                
                if current_unsat < local_best_unsat:
                    local_best_unsat = current_unsat
                    local_best_assignment = assignment[:]
                    stuck_counter = 0
                    stagnation_counter = 0
                else:
                    stuck_counter += 20
                    stagnation_counter += 20
                
                # Boost weights of unsatisfied clauses when stagnating
                if stagnation_counter > 100:
                    for i in range(m):
                        if sat_count[i] == 0:
                            clause_weights[i] = min(clause_weights[i] * weight_boost, max_weight)
                        else:
                            clause_weights[i] = max(clause_weights[i] * weight_decay, min_weight)
                
                if stuck_counter > 2000:
                    break # Restart

        # End of local search
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assignment[:]

        # Reset weights for next restart
        clause_weights = [1.0] * m

    if best_assignment:
        return best_assignment
    return assignment
