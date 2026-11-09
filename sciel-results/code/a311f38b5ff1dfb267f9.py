import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute configurations
    # A configuration is defined by a set of 6 points and a perfect matching of those 6 points.
    # The 8 transversal triples are the sets of 3 points chosen one from each pair.
    
    # We need to map each configuration to a list of 8 triple indices.
    configs = []
    # triple_to_configs: for each triple index, list of config indices containing it
    triple_to_configs = [[] for _ in range(num_triples)]
    
    config_id = 0
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            config_triple_indices = [triple_index[t] for t in transversal]
            configs.append(config_triple_indices)
            
            for t_idx in config_triple_indices:
                triple_to_configs[t_idx].append(config_id)
            
            config_id += 1
            
    num_configs = len(configs)
    
    # Greedy Set Cover to find initial R (set of removed triples)
    # We want to cover all configs with minimum number of triples.
    # R is the set of triples we REMOVE.
    # T is the set of triples we KEEP. T = All - R.
    # We maximize |T| by minimizing |R|.
    
    R = set()
    count = [0] * num_configs
    uncovered = set(range(num_configs))
    
    # Greedy phase
    while uncovered:
        best_t = -1
        best_gain = 0
        # Iterate through all triples not in R
        # Optimization: We can just iterate all triples and check if in R
        for t in range(num_triples):
            if t in R:
                continue
            gain = 0
            for c in triple_to_configs[t]:
                if c in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_t = t
        
        if best_t == -1 or best_gain == 0:
            # Should not happen if problem is solvable, but break to avoid infinite loop
            break
            
        R.add(best_t)
        for c in triple_to_configs[best_t]:
            count[c] += 1
            if count[c] == 1:
                uncovered.discard(c)
                
    # Local Search: Try to remove triples from R
    # A triple t in R can be removed if no config c in triple_to_configs[t] has count[c] == 1
    # (i.e., t is the only triple in R covering c)
    
    def try_remove_triples(R, count, uncovered):
        changed = True
        while changed:
            changed = False
            # Shuffle R to avoid bias
            R_list = list(R)
            random.shuffle(R_list)
            for t in R_list:
                removable = True
                for c in triple_to_configs[t]:
                    if count[c] == 1:
                        removable = False
                        break
                if removable:
                    R.remove(t)
                    for c in triple_to_configs[t]:
                        count[c] -= 1
                        if count[c] == 0:
                            uncovered.add(c)
                    changed = True
                    if time.time() > deadline:
                        return R, count, uncovered
        
    R, count, uncovered = try_remove_triples(R, count, uncovered)
    
    # Simulated Annealing to escape local optima
    # Cost = |R| + lambda * |uncovered|
    # We start with a valid solution (uncovered is empty)
    # We allow moves that create violations if they reduce |R| significantly, 
    # hoping to find a valid solution with smaller |R| later.
    
    lambda_val = 10.0 # Weight for violations
    T = 10.0
    cooling = 0.99
    
    best_R = set(R)
    best_cost = len(R) # uncovered is empty
    
    current_R = set(R)
    current_count = list(count)
    current_uncovered = set(uncovered)
    current_cost = len(current_R) + lambda_val * len(current_uncovered)
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        if move_type < 0.4:
            # Remove a triple from R
            if not current_R:
                continue
            t = random.choice(list(current_R))
            # Apply move
            current_R.remove(t)
            for c in triple_to_configs[t]:
                current_count[c] -= 1
                if current_count[c] == 0:
                    current_uncovered.add(c)
            new_cost = len(current_R) + lambda_val * len(current_uncovered)
            
            # Acceptance
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                if len(current_uncovered) == 0 and len(current_R) < len(best_R):
                    best_R = set(current_R)
                    best_cost = len(best_R)
            else:
                # Revert
                current_R.add(t)
                for c in triple_to_configs[t]:
                    current_count[c] += 1
                    if current_count[c] == 1:
                        current_uncovered.discard(c)
                        
        elif move_type < 0.8:
            # Add a triple to R
            # Pick a triple not in R
            # To be efficient, pick from triples that cover uncovered configs
            if not current_uncovered:
                # Randomly add a triple? No, this increases cost. 
                # Maybe swap is better.
                pass
            else:
                # Find a triple that covers at least one uncovered config
                candidates = []
                for c in list(current_uncovered):
                    for t in configs[c]:
                        if t not in current_R:
                            candidates.append(t)
                            break # Just one candidate per uncovered config is enough? 
                                 # No, we want a good candidate.
                if candidates:
                    # Pick the one covering most uncovered configs
                    best_t = -1
                    best_gain = 0
                    for t in set(candidates): # Unique candidates
                        gain = 0
                        for c in triple_to_configs[t]:
                            if c in current_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_t = t
                            
                    if best_t != -1:
                        current_R.add(best_t)
                        for c in triple_to_configs[best_t]:
                            current_count[c] += 1
                            if current_count[c] == 1:
                                current_uncovered.discard(c)
                        new_cost = len(current_R) + lambda_val * len(current_uncovered)
                        
                        delta_cost = new_cost - current_cost
                        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                            current_cost = new_cost
                            if len(current_uncovered) == 0 and len(current_R) < len(best_R):
                                best_R = set(current_R)
                                best_cost = len(best_R)
                        else:
                            # Revert
                            current_R.remove(best_t)
                            for c in triple_to_configs[best_t]:
                                current_count[c] -= 1
                                if current_count[c] == 0:
                                    current_uncovered.add(c)
                            
        else:
            # Swap
            if current_R:
                t_out = random.choice(list(current_R))
                # Find t_in
                # Ideally t_in covers configs that t_out covers, or uncovered configs
                # Let's just pick a random triple not in R
                t_in = random.choice([t for t in range(num_triples) if t not in current_R])
                
                # Apply swap
                current_R.remove(t_out)
                current_R.add(t_in)
                
                for c in triple_to_configs[t_out]:
                    current_count[c] -= 1
                    if current_count[c] == 0:
                        current_uncovered.add(c)
                        
                for c in triple_to_configs[t_in]:
                    current_count[c] += 1
                    if current_count[c] == 1:
                        current_uncovered.discard(c)
                        
                new_cost = len(current_R) + lambda_val * len(current_uncovered)
                
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    current_cost = new_cost
                    if len(current_uncovered) == 0 and len(current_R) < len(best_R):
                        best_R = set(current_R)
                        best_cost = len(best_R)
                else:
                    # Revert
                    current_R.remove(t_in)
                    current_R.add(t_out)
                    
                    for c in triple_to_configs[t_in]:
                        current_count[c] -= 1
                        if current_count[c] == 0:
                            current_uncovered.add(c)
                            
                    for c in triple_to_configs[t_out]:
                        current_count[c] += 1
                        if current_count[c] == 1:
                            current_uncovered.discard(c)
                            
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # Final cleanup: try to remove triples from best_R
    # We need to reconstruct count and uncovered for best_R
    # Or just run try_remove_triples on best_R
    # But try_remove_triples modifies the sets passed. 
    # Let's create a helper to check validity and remove.
    
    # Re-initialize count and uncovered for best_R
    final_R = set(best_R)
    final_count = [0] * num_configs
    final_uncovered = set(range(num_configs))
    for t in final_R:
        for c in triple_to_configs[t]:
            final_count[c] += 1
            if final_count[c] == 1:
                final_uncovered.discard(c)
                
    # Run local search on final_R
    final_R, final_count, final_uncovered = try_remove_triples(final_R, final_count, final_uncovered)
    
    # Construct the list of kept triples
    # Kept triples are those NOT in final_R
    kept_triples = [all_triples[t] for t in range(num_triples) if t not in final_R]
    
    return [list(t) for t in sorted(kept_triples)]

