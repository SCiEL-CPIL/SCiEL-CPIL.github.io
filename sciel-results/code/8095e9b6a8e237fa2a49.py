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
    # Returns list of clauses affected
    def flip(v):
        assignment[v] = not assignment[v]
        affected = []
        for ci, p in var_to_clauses[v]:
            old_count = sat_count[ci]
            # Determine if v was satisfying before flip
            # If assignment[v] is now the flipped value:
            # If p is True, v satisfies if assignment[v] is True.
            # If p is False, v satisfies if assignment[v] is False.
            # So v satisfies if assignment[v] == p.
            
            # Before flip, assignment[v] was not assignment[v] (current).
            # So old_satisfying = (not assignment[v]) == p
            
            if (not assignment[v]) == p:
                # v was satisfying, now not
                sat_count[ci] = old_count - 1
                if old_count == 1:
                    # Clause becomes unsat
                    unsat_clauses.append(ci)
            else:
                # v was not satisfying, now is
                sat_count[ci] = old_count + 1
                if old_count == 0:
                    # Clause becomes sat
                    # Need to remove ci from unsat_clauses
                    # We'll handle removal lazily or via swap-pop
                    pass
            affected.append(ci)
        return affected

    # Helper to remove a clause from unsat_clauses efficiently
    # We maintain a list. To remove, we can just mark it and clean up when picking.
    # Or, we can keep a dictionary clause_index -> index_in_unsat_list
    # Let's use the lazy approach: keep a list, and when picking, verify.
    
    # Better: Maintain `unsat_clauses` as a list. 
    # When a clause becomes sat, we don't remove it immediately. 
    # When we pick a random clause from `unsat_clauses`, we check if it's actually unsat.
    # If it is sat, we swap it with the last element and pop.
    
    rng = random.Random(42)
    best_unsat = m + 1
    best_assignment = None

    # Main Loop
    max_steps_per_restart = 50000
    noise_prob = 0.25 # Probability to pick random variable from clause
    
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

        while steps < max_steps_per_restart:
            if time.time() > deadline - 5.0:
                break
            
            if not unsat_clauses:
                return assignment[:]

            # Pick a random unsatisfied clause
            # Lazy removal: pick random index, check if still unsat
            while True:
                idx = rng.randrange(len(unsat_clauses))
                ci = unsat_clauses[idx]
                if sat_count[ci] == 0:
                    break
                else:
                    # It's satisfied, remove it
                    unsat_clauses[idx] = unsat_clauses[-1]
                    unsat_clauses.pop()
                    if not unsat_clauses:
                        break
            if not unsat_clauses:
                return assignment[:]

            # Evaluate variables in this clause
            best_v = -1
            best_score = -10**9
            cvars = clause_vars[ci]
            
            scores = []
            for v in cvars:
                # Calculate Gain and Breakage
                gain = 0
                breakage = 0
                for uci, p in var_to_clauses[v]:
                    if sat_count[uci] == 0:
                        # Clause is unsat. If flipping v satisfies it, gain++
                        # v satisfies if new_assignment[v] == p
                        # new_assignment[v] = not assignment[v]
                        if (not assignment[v]) == p:
                            gain += 1
                    else:
                        # Clause is sat. If flipping v breaks it, breakage++
                        # It breaks if v was the ONLY satisfying literal.
                        # So sat_count[uci] must be 1.
                        if sat_count[uci] == 1:
                            # Check if v is currently satisfying
                            if assignment[v] == p:
                                breakage += 1
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
            
            # Update local best
            # We need to know current unsat count. 
            # Maintaining exact count is hard with lazy removal.
            # We can estimate or just check if 0.
            # For "stuck" detection, we can track if we found a new minimum.
            # Let's just run for a fixed number of steps or until time limit.
            # To detect improvement, we can periodically count unsat clauses.
            # Or, we can just rely on the fact that if we find 0, we return.
            # If we don't find 0, we restart.
            
            # Simple heuristic: if steps % 100 == 0, count unsat
            if steps % 50 == 0:
                current_unsat = sum(1 for i in range(m) if sat_count[i] == 0)
                if current_unsat == 0:
                    return assignment[:]
                if current_unsat < local_best_unsat:
                    local_best_unsat = current_unsat
                    local_best_assignment = assignment[:]
                    stuck_counter = 0
                else:
                    stuck_counter += 50
                    
                if stuck_counter > 1000:
                    break # Restart

        # End of local search
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assignment[:]

    if best_assignment:
        return best_assignment
    return assignment
