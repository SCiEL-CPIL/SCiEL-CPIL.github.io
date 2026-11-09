import random
import itertools
import time

def _popcount(x):
    return bin(x).count('1')

def _independence_number_fast(triple_masks, n=16):
    """
    Calculate independence number using bitmask representation.
    triple_masks: list of integers, each representing a triple as a bitmask.
    n: number of points.
    """
    best = 0
    # Precompute popcounts for all masks? 
    # 2^16 = 65536. 
    # We can just iterate.
    for mask in range(1 << n):
        bits = _popcount(mask)
        if bits <= best:
            continue
        
        # Check if mask is independent
        # Independent means it does NOT contain any triple.
        # i.e., for all t in triple_masks, (mask & t) != t
        is_independent = True
        for t in triple_masks:
            if (mask & t) == t:
                is_independent = False
                break
        
        if is_independent:
            best = bits
            
    return best

def _generate_initial_packing(n=16, seed=0):
    """Generate a maximal linear packing using greedy."""
    rng = random.Random(seed)
    used_pairs = set()
    triples = []
    
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        if not any(tuple(sorted(p)) in used_pairs for p in pairs):
            for p in pairs:
                used_pairs.add(tuple(sorted(p)))
            triples.append(t)
            
    return triples

def construct():
    n = 16
    best_triples = None
    best_indep = float('inf')
    
    # Start with a few random packings
    for seed in range(10):
        triples = _generate_initial_packing(n, seed)
        # Convert to masks for fast checking
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        indep = _independence_number_fast(triple_masks, n)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
            
        # If we already found a very good solution, we might stop early
        # but let's run a local search on the best one found so far
        if best_indep <= 5:
             break

    if best_triples is None:
        return []

    # Simulated Annealing / Local Search
    # We operate on the best_triples found
    current_triples = list(best_triples)
    current_indep = best_indep
    
    # Precompute all possible triples and their pairs
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_pairs = {t: [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples}
    triple_to_mask = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_to_mask[t] = m

    # Initial state for SA
    used_pairs = set()
    for t in current_triples:
        for p in triple_to_pairs[t]:
            used_pairs.add(tuple(sorted(p)))
            
    current_masks = [triple_to_mask[t] for t in current_triples]
    
    T_start = 1.0
    T_end = 0.01
    iterations = 5000
    step = (T_start - T_end) / iterations
    
    for i in range(iterations):
        T = T_start - i * step
        if T <= T_end:
            T = T_end
            
        # Pick a random triple to remove
        idx_out = random.randint(0, len(current_triples) - 1)
        t_out = current_triples[idx_out]
        
        # Remove t_out from used_pairs
        for p in triple_to_pairs[t_out]:
            used_pairs.remove(tuple(sorted(p)))
        
        # Find a candidate triple to add
        # We can pick a random triple from all_triples that is not in current_triples
        # To be efficient, we can just pick a random index from all_triples
        # and check validity.
        # Since |T| is small, checking all 560 triples is fast.
        
        best_candidate = None
        # Randomize the order of candidates to avoid bias
        candidates = list(all_triples)
        random.shuffle(candidates)
        
        for t_in in candidates:
            if t_in in current_triples:
                continue
            
            pairs_in = triple_to_pairs[t_in]
            valid = True
            for p in pairs_in:
                if tuple(sorted(p)) in used_pairs:
                    valid = False
                    break
            
            if valid:
                best_candidate = t_in
                break
        
        if best_candidate is None:
            # Restore t_out
            for p in triple_to_pairs[t_out]:
                used_pairs.add(tuple(sorted(p)))
            continue
        
        # Perform swap
        t_in = best_candidate
        for p in triple_to_pairs[t_in]:
            used_pairs.add(tuple(sorted(p)))
            
        current_triples[idx_out] = t_in
        
        # Update masks
        current_masks[idx_out] = triple_to_mask[t_in]
        
        # Calculate new independence number
        new_indep = _independence_number_fast(current_masks, n)
        
        # Acceptance criterion
        if new_indep < current_indep:
            current_indep = new_indep
            if new_indep < best_indep:
                best_indep = new_indep
                best_triples = list(current_triples)
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_indep - current_indep
            if delta > 0:
                if random.random() < (2.71828 ** (-delta / T)):
                    current_indep = new_indep
                else:
                    # Reject: swap back
                    # Remove t_in
                    for p in triple_to_pairs[t_in]:
                        used_pairs.remove(tuple(sorted(p)))
                    # Add t_out
                    for p in triple_to_pairs[t_out]:
                        used_pairs.add(tuple(sorted(p)))
                    current_triples[idx_out] = t_out
                    current_masks[idx_out] = triple_to_mask[t_out]
            else:
                # new_indep == current_indep, accept
                current_indep = new_indep

    return best_triples
