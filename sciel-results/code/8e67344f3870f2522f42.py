import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all configurations (sets of 3 disjoint pairs on 6 vertices)
    # Each configuration defines a constraint: sum of its 8 transversal triples <= 7
    config_transversal_indices = []
    
    # To speed up precomputation, we can iterate over combinations of 6 points
    # and the 15 perfect matchings on them.
    # C(16, 6) = 8008. 8008 * 15 = 120,120 configs. This is manageable.
    
    # We will store for each config the list of triple indices it involves.
    # We also need the reverse map: for each triple, which configs does it belong to?
    
    # First pass: collect all configs
    raw_configs = []
    for v6 in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = v6
        # The 15 perfect matchings on 6 labeled vertices
        # We can hardcode the 15 pairings for indices 0..5
        matchings = [
            ((0, 1), (2, 3), (4, 5)),
            ((0, 1), (2, 4), (3, 5)),
            ((0, 1), (2, 5), (3, 4)),
            ((0, 2), (1, 3), (4, 5)),
            ((0, 2), (1, 4), (3, 5)),
            ((0, 2), (1, 5), (3, 4)),
            ((0, 3), (1, 2), (4, 5)),
            ((0, 3), (1, 4), (2, 5)),
            ((0, 3), (1, 5), (2, 4)),
            ((0, 4), (1, 2), (3, 5)),
            ((0, 4), (1, 3), (2, 5)),
            ((0, 4), (1, 5), (2, 3)),
            ((0, 5), (1, 2), (3, 4)),
            ((0, 5), (1, 3), (2, 4)),
            ((0, 5), (1, 4), (2, 3)),
        ]
        
        for m in matchings:
            p1, p2, p3 = m
            a, b = v6[p1[0]], v6[p1[1]]
            c, d = v6[p2[0]], v6[p2[1]]
            e, f = v6[p3[0]], v6[p3[1]]
            
            # The 8 transversals are formed by picking one from each pair
            transversals = [
                (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                (b, c, e), (b, c, f), (b, d, e), (b, d, f)
            ]
            
            indices = []
            valid = True
            for t in transversals:
                t_sorted = tuple(sorted(t))
                idx = triple_to_idx.get(t_sorted)
                if idx is None:
                    valid = False
                    break
                indices.append(idx)
            
            if valid:
                raw_configs.append(indices)
                
    num_configs = len(raw_configs)
    
    # Build reverse mapping: triple_index -> list of config indices
    triple_to_configs = [[] for _ in range(num_triples)]
    for cfg_idx, indices in enumerate(raw_configs):
        for t_idx in indices:
            triple_to_configs[t_idx].append(cfg_idx)
            
    # 2. Simulated Annealing Setup
    # State: binary vector x of length num_triples
    # Objective: Maximize sum(x)
    # Constraints: For each config c, sum(x_i for i in c) <= 7
    
    # We track:
    # - x: current assignment (0 or 1)
    # - config_counts: for each config, how many of its triples are currently in x
    # - current_score: sum(x)
    # - current_violations: number of configs with count > 7 (or sum of excess?)
    #   For SA, it's often better to define an energy function.
    #   Energy = -current_score + penalty * sum(max(0, config_count - 7))
    #   We want to minimize Energy.
    
    # Let's use a penalty-based approach.
    # Energy E = -sum(x) + P * sum_c (max(0, count_c - 7))
    # If we find a state with E < 0 and no violations, it's a valid solution with score -E.
    # Actually, if valid, E = -score. So minimizing E maximizes score.
    # If invalid, E includes penalty.
    
    # We will track violations separately to ensure we return a VALID solution.
    # The SA will try to minimize (violations, -score) lexicographically or via a combined cost.
    # Let's use Cost = violations * BIG_NUMBER - score.
    # BIG_NUMBER should be larger than max possible score (560). Let's use 1000.
    
    BIG_PENALTY = 1000
    
    def calculate_cost(x, config_counts):
        score = sum(x)
        violations = 0
        for c in range(num_configs):
            if config_counts[c] > 7:
                violations += 1
        return violations * BIG_PENALTY - score, score, violations

    # Initialization: Random valid subset? Or just random and let SA fix it?
    # Starting from a random set of triples likely has many violations.
    # A better start: Greedy construction to get a valid baseline, then SA to improve.
    
    # Greedy Initialization
    def greedy_init():
        x = [0] * num_triples
        config_counts = [0] * num_configs
        # Random order
        indices = list(range(num_triples))
        random.shuffle(indices)
        for t_idx in indices:
            # Check if adding t_idx keeps all configs valid
            can_add = True
            for c_idx in triple_to_configs[t_idx]:
                if config_counts[c_idx] >= 7:
                    can_add = False
                    break
            if can_add:
                x[t_idx] = 1
                for c_idx in triple_to_configs[t_idx]:
                    config_counts[c_idx] += 1
        return x, config_counts

    x, config_counts = greedy_init()
    current_cost, current_score, current_violations = calculate_cost(x, config_counts)
    
    best_x = x[:]
    best_cost = current_cost
    best_score = current_score
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995 # Adjust to fit time
    # We need to estimate number of steps.
    # Let's run until time limit.
    
    # Precompute neighbors? No, random selection is fine.
    # Move: Flip a random triple i.
    # Delta Cost calculation:
    # If x[i] was 0 -> 1:
    #   Score increases by 1.
    #   For each config c in triple_to_configs[i]:
    #     count_c increases by 1.
    #     If count_c was 7 -> 8, violations increase by 1 (cost +BIG_PENALTY).
    #     If count_c was 8 -> 9, violations stays same (cost 0 change in penalty part, but wait, violation is boolean >7? 
    #     My cost function uses count of violating configs. So 7->8 adds 1 to violation count. 8->9 adds 0.
    #     So delta_violations = sum(1 for c in configs[i] if count_c == 7)
    #   delta_cost = -1 + delta_violations * BIG_PENALTY
    # If x[i] was 1 -> 0:
    #   Score decreases by 1.
    #   For each config c in triple_to_configs[i]:
    #     count_c decreases by 1.
    #     If count_c was 8 -> 7, violations decrease by 1 (cost -BIG_PENALTY).
    #     If count_c was 9 -> 8, violations stays same.
    #     So delta_violations = -sum(1 for c in configs[i] if count_c == 8)
    #   delta_cost = 1 + delta_violations * BIG_PENALTY
    
    # Let's refine the cost function to be smoother for SA.
    # Cost = sum(max(0, count_c - 7)) * BIG_PENALTY - sum(x)
    # This penalizes excess capacity, not just violation existence.
    # This allows SA to navigate through "slightly invalid" states more smoothly.
    
    def calc_smooth_cost(x, config_counts):
        penalty = 0
        for c in range(num_configs):
            if config_counts[c] > 7:
                penalty += config_counts[c] - 7
        score = sum(x)
        return penalty * BIG_PENALTY - score, score, penalty

    current_cost, current_score, current_penalty = calc_smooth_cost(x, config_counts)
    best_cost = current_cost
    best_score = current_score
    best_x = x[:]
    
    T = T_start
    
    while time.time() < deadline - 2:
        # Batch of steps? No, just loop.
        # Pick random triple to flip
        i = random.randint(0, num_triples - 1)
        
        if x[i] == 0:
            # Try to add
            delta_penalty = 0
            for c_idx in triple_to_configs[i]:
                if config_counts[c_idx] >= 7:
                    # Adding will increase penalty by 1 (from 7 to 8 is +1, 8 to 9 is +1)
                    delta_penalty += 1
            delta_score = 1
            delta_cost = delta_penalty * BIG_PENALTY - delta_score
            
            if delta_cost < 0 or (T > 0 and random.random() < np.exp(-delta_cost / T)):
                # Accept
                x[i] = 1
                for c_idx in triple_to_configs[i]:
                    config_counts[c_idx] += 1
                current_cost += delta_cost
                current_score += delta_score
                current_penalty += delta_penalty
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_score = current_score
                    best_x = x[:]
                    
        else:
            # Try to remove
            delta_penalty = 0
            for c_idx in triple_to_configs[i]:
                if config_counts[c_idx] > 7:
                    # Removing will decrease penalty by 1
                    delta_penalty -= 1
            delta_score = -1
            delta_cost = delta_penalty * BIG_PENALTY - delta_score
            
            if delta_cost < 0 or (T > 0 and random.random() < np.exp(-delta_cost / T)):
                # Accept
                x[i] = 0
                for c_idx in triple_to_configs[i]:
                    config_counts[c_idx] -= 1
                current_cost += delta_cost
                current_score += delta_score
                current_penalty += delta_penalty
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_score = current_score
                    best_x = x[:]
                    
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    # Extract best valid solution
    # We need to ensure the returned solution is VALID.
    # best_x might be invalid if best_cost still has penalty > 0.
    # We should keep track of the best VALID solution separately, or filter at the end.
    # Given the penalty is large, best_cost likely corresponds to a valid solution if one exists.
    # But to be safe, let's verify best_x.
    
    # Verify best_x
    valid_best = True
    final_config_counts = [0] * num_configs
    for t_idx in range(num_triples):
        if best_x[t_idx]:
            for c_idx in triple_to_configs[t_idx]:
                final_config_counts[c_idx] += 1
                
    for c in range(num_configs):
        if final_config_counts[c] > 7:
            valid_best = False
            break
            
    if not valid_best:
        # Fallback: use the last valid state we saw? Or just return greedy init?
        # Let's try to clean up best_x by removing violating triples greedily.
        x = best_x[:]
        config_counts = final_config_counts[:]
        # Remove triples from most violated configs
        changed = True
        while changed:
            changed = False
            # Find config with max violation
            max_viol = 0
            max_cfg = -1
            for c in range(num_configs):
                if config_counts[c] > 7 and config_counts[c] - 7 > max_viol:
                    max_viol = config_counts[c] - 7
                    max_cfg = c
            if max_cfg == -1:
                break
            # Remove a random triple from this config
            candidates = [t for t in raw_configs[max_cfg] if x[t]]
            if candidates:
                t_rm = random.choice(candidates)
                x[t_rm] = 0
                for c_idx in triple_to_configs[t_rm]:
                    config_counts[c_idx] -= 1
                changed = True
        
        # Verify again
        valid_final = True
        for c in range(num_configs):
            if config_counts[c] > 7:
                valid_final = False
                break
        
        if valid_final:
            best_x = x
            best_score = sum(x)
        else:
            # Absolute fallback: Greedy
            x, config_counts = greedy_init()
            best_x = x
            best_score = sum(x)

    return [all_triples[i] for i in range(num_triples) if best_x[i]]

