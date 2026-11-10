import itertools
import random
import time
import math

def construct():
    n = 16
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute bitmasks for each triple
    triple_masks = [0] * num_triples
    # Precompute pairs for linearity check
    triple_pairs = []
    for i, t in enumerate(all_triples):
        a, b, c = t
        triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
        triple_pairs.append([(a, b), (a, c), (b, c)])
        
    # Precompute pair keys for fast lookup
    # Map pair (min, max) to a unique integer for hashing
    def pair_key(p):
        return p[0] * 16 + p[1]

    # Fast independence number calculation
    # Iterates all 2^16 subsets.
    # A subset S is independent if it does not contain any active triple.
    # i.e., for all active triple masks T, (S & T) != T.
    def get_alpha(active_indices):
        if not active_indices:
            return n
            
        active_masks = [triple_masks[i] for i in active_indices]
        max_ind = 0
        
        # Iterate through all subsets
        # Optimization: If we find a set of size k, we don't need to check smaller ones?
        # No, we want the MAX. So we can just check all and take max.
        # Or iterate from high size down?
        # Iterating all 2^16 is ~65k iterations.
        # Each iteration checks M triples.
        
        # To speed up, we can precompute a "covered" array?
        # No, let's just use the bitmask check.
        
        # We can prune: if the number of bits in S is <= max_ind, skip?
        # bit_count() is fast in Python 3.8+?
        
        for s in range(1 << n):
            # Count bits
            # If we want to be faster, we can group by size?
            # For now, simple loop.
            if s == 0:
                continue
                
            # Check if s is independent
            is_ind = True
            for m in active_masks:
                if (s & m) == m:
                    is_ind = False
                    break
            
            if is_ind:
                # Count bits
                # bin(s).count("1") is slow.
                # Use a lookup table or bit trick?
                # Python 3.10+ has int.bit_count()
                size = s.bit_count()
                if size > max_ind:
                    max_ind = size
        
        return max_ind

    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(num_triples))
        random.shuffle(indices)
        active = []
        used_pairs = set()
        
        for idx in indices:
            # Check if this triple conflicts with used_pairs
            conflict = False
            for p in triple_pairs[idx]:
                key = pair_key(p)
                if key in used_pairs:
                    conflict = True
                    break
            if not conflict:
                active.append(idx)
                for p in triple_pairs[idx]:
                    key = pair_key(p)
                    used_pairs.add(key)
        return active, used_pairs

    # Local Search with Simulated Annealing
    def local_search(initial_active, initial_used_pairs, max_time=100):
        active = initial_active[:]
        used_pairs = initial_used_pairs.copy()
        
        # Calculate initial alpha
        current_alpha = get_alpha(active)
        best_alpha = current_alpha
        best_active = active[:]
        
        start_time = time.time()
        T = 5.0 # Initial temperature
        cooling_rate = 0.995
        
        # Precompute for each triple, which pairs it uses
        # To efficiently add/remove, we need to know which pairs are free.
        
        while time.time() - start_time < max_time:
            # 1. Remove a random triple
            if not active:
                break
            rem_idx = random.choice(active)
            active.remove(rem_idx)
            # Free its pairs
            for p in triple_pairs[rem_idx]:
                key = pair_key(p)
                used_pairs.discard(key)
            
            # 2. Add a random valid triple
            # To be efficient, we can pick a random triple and check validity.
            # If invalid, try another.
            add_idx = -1
            for _ in range(20): # Max 20 attempts
                cand_idx = random.randint(0, num_triples - 1)
                if cand_idx in active:
                    continue
                
                valid = True
                for p in triple_pairs[cand_idx]:
                    key = pair_key(p)
                    if key in used_pairs:
                        valid = False
                        break
                
                if valid:
                    add_idx = cand_idx
                    break
            
            if add_idx == -1:
                # Could not find a valid triple to add, revert
                active.append(rem_idx)
                for p in triple_pairs[rem_idx]:
                    key = pair_key(p)
                    used_pairs.add(key)
                continue
            
            active.append(add_idx)
            for p in triple_pairs[add_idx]:
                key = pair_key(p)
                used_pairs.add(key)
            
            # 3. Evaluate new alpha
            new_alpha = get_alpha(active)
            
            # 4. Acceptance criterion
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_active = active[:]
                    # Optional: restart cooling if we found a new best?
            else:
                # Simulated Annealing acceptance
                delta = new_alpha - current_alpha
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        active.remove(add_idx)
                        for p in triple_pairs[add_idx]:
                            key = pair_key(p)
                            used_pairs.discard(key)
                        active.append(rem_idx)
                        for p in triple_pairs[rem_idx]:
                            key = pair_key(p)
                            used_pairs.add(key)
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
        
        return best_active, best_alpha

    # Main Logic
    best_alpha = n
    best_active = []
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 110:
        seed += 1
        active, used_pairs = greedy_construct(seed)
        
        # Run local search for a short duration
        # We can run multiple local searches with different seeds/initial states
        # Or just one long one?
        # Let's run local search for a fraction of the time.
        # Since get_alpha is somewhat expensive, we shouldn't run it too frequently in tight loops if possible, 
        # but the bitmask check is okay.
        
        # Let's run local search until time runs out or we find alpha <= 4
        # But we need to allow for multiple restarts if possible.
        
        # Strategy: Run local search for 10 seconds, then restart with a new greedy seed.
        search_time = 10
        if time.time() - start_time + search_time > 110:
            search_time = 110 - (time.time() - start_time)
            if search_time <= 1:
                break
            
        best_active_ls, best_alpha_ls = local_search(active, used_pairs, max_time=search_time)
        
        if best_alpha_ls < best_alpha:
            best_alpha = best_alpha_ls
            best_active = best_active_ls
            
        if best_alpha <= 4:
            break

    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_active]
    
    return result
