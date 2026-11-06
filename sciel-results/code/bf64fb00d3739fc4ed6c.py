import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_mask = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute pairs for each triple to check compatibility quickly
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset([(min(a,b), max(a,b)), (min(a,c), max(a,c)), (min(b,c), max(b,c))])
        
    def calc_alpha(triple_set):
        if not triple_set:
            return n
        
        masks = [triple_to_mask[t] for t in triple_set]
        
        # Check sizes from n down to 1
        for size in range(n, 0, -1):
            # Optimization: If we have found an independent set of size k, 
            # we don't need to check smaller sizes? No, we want the MAX.
            # So we check n, n-1, ... until we find one.
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_indep = True
                for tm in masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return size
        return 0

    best_alpha = n
    best_triples = []
    
    # Greedy Randomized Construction
    # We want to build a dense linear triple system.
    # Strategy: Randomly shuffle triples, pick if compatible.
    # This tends to produce systems with ~20-30 triples.
    
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        current_triples = []
        used_pairs = set()
        
        for t in all_triples:
            p_set = triple_pairs[t]
            if used_pairs.isdisjoint(p_set):
                current_triples.append(t)
                used_pairs.update(p_set)
        
        if not current_triples:
            continue
            
        alpha = calc_alpha(current_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples.copy()
            
            # If we hit a very low alpha, we can stop or refine
            if best_alpha <= 6:
                # Don't break immediately, try to refine this one
                pass

    # Local Search on best found
    if best_triples:
        current_set = set(best_triples)
        current_alpha = best_alpha
        current_triples_list = list(current_set)
        
        # Precompute compatibility for local search?
        # We can just check on the fly.
        
        no_improve_count = 0
        max_no_improve = 500
        
        while no_improve_count < max_no_improve and time.time() < deadline:
            # Pick a random triple to remove
            if not current_set:
                break
            t_out = random.choice(list(current_set))
            new_set = current_set - {t_out}
            
            # Find a compatible triple to add
            # We can iterate through all_triples and pick a random compatible one
            # Or just try a few random ones
            
            # Optimization: Precompute list of all triples
            # Filter compatible ones
            # This is O(N_all * N_current) which is 560 * 30 = 16800 ops.
            # We can do this once per iteration.
            
            candidates = []
            for t in all_triples:
                if t in new_set:
                    continue
                p_set = triple_pairs[t]
                if used_pairs_isdisjoint(p_set, new_set, triple_pairs):
                    candidates.append(t)
            
            if not candidates:
                no_improve_count += 1
                continue
                
            t_in = random.choice(candidates)
            candidate_set = new_set | {t_in}
            
            alpha_new = calc_alpha(candidate_set)
            
            if alpha_new < current_alpha:
                current_set = candidate_set
                current_alpha = alpha_new
                best_alpha = current_alpha
                best_triples = list(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1

    return best_triples

def used_pairs_isdisjoint(p_set, triple_set, triple_pairs):
    # Check if p_set is disjoint from all pairs in triple_set
    for t in triple_set:
        if not p_set.isdisjoint(triple_pairs[t]):
            return False
    return True
