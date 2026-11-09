import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    points = list(range(16))
    all_triples = list(itertools.combinations(range(16), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Pre-compute 7-sets and their indices
    seven_sets = list(itertools.combinations(range(16), 7))
    num_7sets = len(seven_sets)
    
    # Map each triple to the list of 7-set indices it covers
    # triple (a,b,c) is contained in 7-sets that include a,b,c and 4 others from the remaining 13.
    triple_to_7set_indices = {}
    for t in all_triples:
        a, b, c = t
        # Remaining points
        rem = [p for p in range(16) if p not in (a,b,c)]
        indices = []
        for combo in itertools.combinations(rem, 4):
            s = tuple(sorted((a,b,c) + combo))
            # We need the index of this 7-set in seven_sets
            # This is slow if done naively.
            pass
            
    # Better pre-computation:
    # Iterate through all 7-sets, and for each, add it to the lists of its 3-subsets.
    # There are 11440 7-sets. Each has 35 3-subsets.
    # Total ops: 11440 * 35 = 400,400.
    triple_to_7set_indices = {t: [] for t in all_triples}
    for idx, s in enumerate(seven_sets):
        for sub in itertools.combinations(s, 3):
            triple_to_7set_indices[sub].append(idx)
            
    # Greedy initialization
    def greedy_init():
        triples = []
        used_pairs = set()
        all_t = all_triples[:]
        random.shuffle(all_t)
        for t in all_t:
            pairs = set(itertools.combinations(t, 2))
            if not (pairs & used_pairs):
                triples.append(t)
                used_pairs.update(pairs)
        return triples, used_pairs
        
    best_triples = []
    best_uncovered = num_7sets
    
    # Run SA
    # We will run multiple restarts if time permits
    
    def run_sa(init_triples, init_used_pairs, time_limit):
        triples = list(init_triples)
        used_pairs = set(init_used_pairs)
        
        # Initialize counts
        count_7sets = [0] * num_7sets
        uncovered = num_7sets
        
        for t in triples:
            for idx in triple_to_7set_indices[t]:
                if count_7sets[idx] == 0:
                    uncovered -= 1
                count_7sets[idx] += 1
                
        current_uncovered = uncovered
        T = 10.0
        cooling = 0.995
        best_local_uncovered = current_uncovered
        best_local_triples = list(triples)
        best_local_used = set(used_pairs)
        
        while time.time() < time_limit:
            if T < 0.01:
                break
            
            # Choose move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                if len(triples) >= 40: continue # Max packing bound approx
                # Pick random triple not in triples
                # We can just pick random from all_triples and check
                t = random.choice(all_triples)
                if t in triples: continue
                pairs = set(itertools.combinations(t, 2))
                if pairs & used_pairs: continue
                
                # Apply
                triples.append(t)
                used_pairs.update(pairs)
                for idx in triple_to_7set_indices[t]:
                    if count_7sets[idx] == 0:
                        uncovered -= 1
                    count_7sets[idx] += 1
                
                delta = uncovered - current_uncovered
                if delta <= 0 or random.random() < pow(2.718, -delta / T):
                    current_uncovered = uncovered
                    if current_uncovered < best_local_uncovered:
                        best_local_uncovered = current_uncovered
                        best_local_triples = list(triples)
                        best_local_used = set(used_pairs)
                else:
                    # Undo
                    triples.pop()
                    for p in pairs:
                        used_pairs.discard(p)
                    for idx in triple_to_7set_indices[t]:
                        count_7sets[idx] -= 1
                        if count_7sets[idx] == 0:
                            uncovered += 1
                            
            elif move_type == 'remove':
                if not triples: continue
                t = random.choice(triples)
                pairs = set(itertools.combinations(t, 2))
                
                # Apply
                triples.remove(t)
                for p in pairs:
                    used_pairs.discard(p)
                for idx in triple_to_7set_indices[t]:
                    count_7sets[idx] -= 1
                    if count_7sets[idx] == 0:
                        uncovered += 1
                        
                delta = uncovered - current_uncovered
                if delta <= 0 or random.random() < pow(2.718, -delta / T):
                    current_uncovered = uncovered
                    if current_uncovered < best_local_uncovered:
                        best_local_uncovered = current_uncovered
                        best_local_triples = list(triples)
                        best_local_used = set(used_pairs)
                else:
                    # Undo
                    triples.append(t)
                    used_pairs.update(pairs)
                    for idx in triple_to_7set_indices[t]:
                        if count_7sets[idx] == 0:
                            uncovered -= 1
                        count_7sets[idx] += 1

            else: # swap
                if not triples: continue
                t_remove = random.choice(triples)
                t_add = random.choice(all_triples)
                if t_add in triples: continue
                
                p_rem = set(itertools.combinations(t_remove, 2))
                p_add = set(itertools.combinations(t_add, 2))
                
                # Check linearity for add
                # The pairs of t_add must not be in (used_pairs - p_rem)
                # i.e. p_add & (used_pairs - p_rem) must be empty
                # Equivalent to: p_add & used_pairs is subset of p_rem
                # And p_add & p_rem must be empty? No, t_add and t_remove are distinct.
                # If they share a pair, that pair is in p_add and p_rem.
                # If they share a pair, we can't swap directly because the pair would be duplicated?
                # No, if we remove t_remove, the pair is free. Then we add t_add.
                # If t_add contains a pair from t_remove, it's fine because that pair is freed.
                # But if t_add contains a pair from ANOTHER triple, it's invalid.
                
                conflict = p_add & used_pairs
                # The conflict is valid only if all conflicting pairs are in p_rem
                if not conflict.issubset(p_rem):
                    continue
                # Also, t_add must not be the same as t_remove
                # And t_add must not create a conflict with itself? No.
                
                # Apply
                triples.remove(t_remove)
                for p in p_rem:
                    used_pairs.discard(p)
                triples.append(t_add)
                used_pairs.update(p_add)
                
                # Update counts
                for idx in triple_to_7set_indices[t_remove]:
                    count_7sets[idx] -= 1
                    if count_7sets[idx] == 0:
                        uncovered += 1
                for idx in triple_to_7set_indices[t_add]:
                    if count_7sets[idx] == 0:
                        uncovered -= 1
                    count_7sets[idx] += 1
                        
                delta = uncovered - current_uncovered
                if delta <= 0 or random.random() < pow(2.718, -delta / T):
                    current_uncovered = uncovered
                    if current_uncovered < best_local_uncovered:
                        best_local_uncovered = current_uncovered
                        best_local_triples = list(triples)
                        best_local_used = set(used_pairs)
                else:
                    # Undo
                    triples.remove(t_add)
                    for p in p_add:
                        used_pairs.discard(p)
                    triples.append(t_remove)
                    used_pairs.update(p_rem)
                    
                    for idx in triple_to_7set_indices[t_add]:
                        count_7sets[idx] -= 1
                        if count_7sets[idx] == 0:
                            uncovered += 1
                    for idx in triple_to_7set_indices[t_remove]:
                        if count_7sets[idx] == 0:
                            uncovered -= 1
                        count_7sets[idx] += 1
                            
            T *= cooling
            
        return best_local_triples, best_local_uncovered

    # Restart loop
    remaining_time = deadline - time.time()
    num_restarts = 5
    time_per_restart = remaining_time / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        init_t, init_p = greedy_init()
        res_t, res_u = run_sa(init_t, init_p, time.time() + time_per_restart)
        if res_u < best_uncovered:
            best_uncovered = res_u
            best_triples = res_t
            
    # Verify and return
    # Convert tuples to lists if necessary? The prompt says "list of triples [a, b, c]"
    # The current program returns lists.
    return [list(t) for t in best_triples]
