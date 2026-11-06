import time
import random
from itertools import combinations
import numpy as np
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (rem[0], rem[j])
                if j == 1:
                    pair3 = (rem[2], rem[3])
                elif j == 2:
                    pair3 = (rem[1], rem[3])
                else:
                    pair3 = (rem[1], rem[2])
                parts.append([pair1, pair2, pair3])
        return parts
    
    for six_subset in combinations(range(n), 6):
        for partition in get_pair_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)
    
    num_k222s = len(k222_triples)
    
    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]
    
    # Max-SAT local search: each triple is a boolean variable, each K(2,2,2) is a clause
    # A clause is violated if all 8 of its triples are present
    # Objective: maximize number of present triples subject to no violated clauses
    
    def maxsat_local_search(initial_present, max_steps=500000):
        """
        Local search for Max-SAT formulation.
        State: boolean array of which triples are present.
        Clause satisfaction: k_counts[k] = number of present triples in clause k.
        A clause is violated if k_counts[k] == 8.
        
        Moves:
        - Add: set a missing triple to true (if it doesn't create a violation)
        - Remove: set a present triple to false
        - Swap: remove one present, add one missing
        
        Objective: maximize sum(present) - penalty * sum(violations)
        But we want zero violations, so we use a hard constraint approach:
        Only allow moves that keep violations at 0, or temporarily allow 1 violation
        to escape local optima (simulated annealing style).
        """
        present = initial_present[:]
        k_counts = [0] * num_k222s
        for t in range(num_triples):
            if present[t]:
                for k_idx in triple_to_k222s[t]:
                    k_counts[k_idx] += 1
        
        # Count violations
        violations = sum(1 for k in range(num_k222s) if k_counts[k] == 8)
        
        best_present = present[:]
        best_size = sum(present)
        best_violations = violations
        
        temp = 0.5
        cooling = 0.9999
        accepted_moves = 0
        total_moves = 0
        
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            
            total_moves += 1
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_present = present[:]
            new_k_counts = k_counts[:]
            
            if move_type == 'add':
                missing = [i for i in range(num_triples) if not present[i]]
                if not missing:
                    continue
                t_in = random.choice(missing)
                # Check if adding t_in would create a violation
                would_violate = 0
                for k_idx in triple_to_k222s[t_in]:
                    if new_k_counts[k_idx] == 7:
                        would_violate += 1
                new_present[t_in] = True
                for k_idx in triple_to_k222s[t_in]:
                    new_k_counts[k_idx] += 1
                new_violations = sum(1 for k in range(num_k222s) if new_k_counts[k] == 8)
                delta_size = 1
                delta_viol = new_violations - violations
                
            elif move_type == 'remove':
                present_list = [i for i in range(num_triples) if present[i]]
                if not present_list:
                    continue
                t_out = random.choice(present_list)
                new_present[t_out] = False
                for k_idx in triple_to_k222s[t_out]:
                    new_k_counts[k_idx] -= 1
                new_violations = sum(1 for k in range(num_k222s) if new_k_counts[k] == 8)
                delta_size = -1
                delta_viol = new_violations - violations
                
            else:  # swap
                present_list = [i for i in range(num_triples) if present[i]]
                missing = [i for i in range(num_triples) if not present[i]]
                if not present_list or not missing:
                    continue
                t_out = random.choice(present_list)
                t_in = random.choice(missing)
                new_present[t_out] = False
                for k_idx in triple_to_k222s[t_out]:
                    new_k_counts[k_idx] -= 1
                new_present[t_in] = True
                for k_idx in triple_to_k222s[t_in]:
                    new_k_counts[k_idx] += 1
                new_violations = sum(1 for k in range(num_k222s) if new_k_counts[k] == 8)
                delta_size = 0
                delta_viol = new_violations - violations
            
            # Acceptance criterion: prefer no violations, then maximize size
            # Use soft penalty for violations
            penalty = 100
            old_score = sum(present) - penalty * violations
            new_score = sum(new_present) - penalty * new_violations
            delta_score = new_score - old_score
            
            if delta_score >= 0:
                accept = True
            else:
                # Simulated annealing: accept worse moves with probability exp(-delta/temp)
                if temp > 1e-6:
                    accept = random.random() < math.exp(delta_score / temp)
                else:
                    accept = False
            
            if accept:
                present = new_present
                k_counts = new_k_counts
                violations = new_violations
                accepted_moves += 1
                
                current_size = sum(present)
                if violations == 0 and current_size > best_size:
                    best_size = current_size
                    best_present = present[:]
                elif violations == best_violations and current_size > best_size:
                    best_size = current_size
                    best_present = present[:]
                    best_violations = violations
            
            temp *= cooling
        
        return best_present, best_size
    
    def greedy_init(seed):
        local_rng = random.Random(seed)
        present = [False] * num_triples
        k_counts = [0] * num_k222s
        
        # Sort by constraint count (fewer K(2,2,2)s first) with randomness
        base_order = sorted(range(num_triples), key=lambda t: triple_k222_count[t])
        for i in range(len(base_order)):
            if i > 0 and local_rng.random() < 0.1:
                j = max(0, min(len(base_order)-1, i + local_rng.randint(-5, 5)))
                base_order[i], base_order[j] = base_order[j], base_order[i]
        
        for t_id in base_order:
            if time.time() > DEADLINE:
                break
            # Check if adding this triple would create a violation
            can_add = True
            for k_idx in triple_to_k222s[t_id]:
                if k_counts[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                present[t_id] = True
                for k_idx in triple_to_k222s[t_id]:
                    k_counts[k_idx] += 1
        
        return present
    
    def polish(present, max_iters=100000):
        """Try to add any missing triple that fits."""
        k_counts = [0] * num_k222s
        for t in range(num_triples):
            if present[t]:
                for k_idx in triple_to_k222s[t]:
                    k_counts[k_idx] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Get missing triples sorted by constraint count (ascending)
            missing = [(triple_k222_count[t], t) for t in range(num_triples) if not present[t]]
            missing.sort()
            for _, t_id in missing:
                can_add = True
                for k_idx in triple_to_k222s[t_id]:
                    if k_counts[k_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    present[t_id] = True
                    for k_idx in triple_to_k222s[t_id]:
                        k_counts[k_idx] += 1
                    improved = True
            # Random additions
            for _ in range(200):
                if time.time() > DEADLINE:
                    break
                t_id = random.randint(0, num_triples - 1)
                if not present[t_id]:
                    can_add = True
                    for k_idx in triple_to_k222s[t_id]:
                        if k_counts[k_idx] == 7:
                            can_add = False
                            break
                    if can_add:
                        present[t_id] = True
                        for k_idx in triple_to_k222s[t_id]:
                            k_counts[k_idx] += 1
                        improved = True
        
        return present
    
    best_size = 0
    best_present = None
    rng = random.Random(42)
    
    # Main loop: multiple restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919
        
        if restart_count <= 5:
            # Greedy init
            present = greedy_init(seed)
        else:
            # Start from best and perturb
            present = best_present[:] if best_present else [False] * num_triples
            n_perturb = min(20, sum(present) // 10 + 3)
            present_list = [i for i in range(num_triples) if present[i]]
            for _ in range(n_perturb):
                if present_list:
                    t = random.choice(present_list)
                    present[t] = False
                    present_list = [i for i in range(num_triples) if present[i]]
        
        # Polish
        present = polish(present)
        current_size = sum(present)
        
        if current_size > best_size:
            best_size = current_size
            best_present = present[:]
        
        # Max-SAT local search
        if time.time() < DEADLINE - 5:
            sa_present, sa_size = maxsat_local_search(present, max_steps=100000)
            if sa_size > best_size:
                best_size = sa_size
                best_present = sa_present[:]
        
        # Polish after SA
        if time.time() < DEADLINE - 2:
            final_present = polish(best_present)
            if sum(final_present) > best_size:
                best_size = sum(final_present)
                best_present = final_present[:]
    
    # Final high-precision polish pass
    if best_present and time.time() < DEADLINE - 1:
        final_state = polish(best_present, max_iters=200000)
        if sum(final_state) > best_size:
            best_size = sum(final_state)
            best_present = final_state[:]
    
    return [list(triples[t_id]) for t_id in range(num_triples) if best_present and best_present[t_id]]
