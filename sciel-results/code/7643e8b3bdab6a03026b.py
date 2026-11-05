import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n = 16
    # Precompute all possible triples as frozensets or tuples
    all_triples = [tuple(t) for t in itertools.combinations(range(n), 3)]
    
    # Helper to check if a triple is valid to add to current set
    # We maintain a set of pairs that are already used
    # If a triple contains a pair that is already used, it's invalid.
    
    def get_initial_packing():
        used_pairs = set()
        current_triples = []
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            p1, p2 = t[0], t[1]
            p1, p3 = t[0], t[2]
            p2, p3 = t[1], t[2]
            
            # Check if any pair is already used
            # Normalize pairs for set storage
            pair1 = (min(p1, p2), max(p1, p2))
            pair2 = (min(p1, p3), max(p1, p3))
            pair3 = (min(p2, p3), max(p2, p3))
            
            if pair1 not in used_pairs and pair2 not in used_pairs and pair3 not in used_pairs:
                current_triples.append(t)
                used_pairs.add(pair1)
                used_pairs.add(pair2)
                used_pairs.add(pair3)
                
        return current_triples

    def calc_alpha(triples):
        # Branch and bound for Max Independent Set
        # A set S is independent if it contains no triple from 'triples'
        # We want the max size of such S.
        
        # Convert triples to bitmasks for faster checking?
        # Or just use sets.
        # Given small n, sets are fine.
        
        triples_set = [set(t) for t in triples]
        
        # We can prune if current_size + remaining_points <= best
        # We iterate through points.
        
        best = 0
        
        # Sort triples to help pruning? Maybe not necessary for n=16.
        
        def dfs(idx, current_set, size):
            nonlocal best
            if idx == n:
                if size > best:
                    best = size
                return
            
            # Option 1: Exclude point idx
            dfs(idx + 1, current_set, size)
            
            # Option 2: Include point idx
            # Check if adding idx to current_set creates a triple
            # A triple is formed if there exists t in triples_set such that t is subset of current_set U {idx}
            # This is expensive to check naively.
            
            # Better: Precompute for each point, which triples it belongs to.
            # If we add idx, we must ensure no triple containing idx is fully included in current_set U {idx}.
            # i.e., for all t containing idx, (t \ {idx}) is NOT a subset of current_set.
            
            # Let's optimize this specific check.
            pass

        # Simpler approach for n=16:
        # Iterate all 2^16 subsets? 65536 subsets.
        # For each subset, check if it's independent.
        # To check if subset S is independent:
        # For each triple t, if t is subset of S, S is not independent.
        
        # We can speed this up by precomputing a "bad" mask.
        # A subset S is bad if it contains any triple.
        # We can compute the union of all supersets of each triple.
        # There are only ~20 triples.
        # Each triple has 2^(16-3) = 8192 supersets.
        # Total bad subsets <= 20 * 8192 = 163840.
        # We can just use a boolean array of size 2^16.
        
        is_bad = [False] * (1 << n)
        
        for t in triples:
            mask_t = 0
            for p in t:
                mask_t |= (1 << p)
            
            # Mark all supersets of mask_t as bad
            # Supersets of mask_t are masks M such that (M & mask_t) == mask_t
            # We can iterate through all masks and check, or generate supersets.
            # Generating supersets:
            # Let free_bits = ~mask_t & ((1 << n) - 1)
            # Iterate through all submasks of free_bits
            free = ((1 << n) - 1) ^ mask_t
            sub = free
            while True:
                is_bad[mask_t | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & free
                
        # Now find the max popcount of any mask where is_bad is False
        max_size = 0
        for mask in range(1 << n):
            if not is_bad[mask]:
                pop = bin(mask).count('1')
                if pop > max_size:
                    max_size = pop
                    if max_size == n: # Can't get better
                        break
        return max_size

    # Initial solution
    current_triples = get_initial_packing()
    current_alpha = calc_alpha(current_triples)
    best_triples = current_triples[:]
    best_alpha = current_alpha
    
    # Local Search
    # We will try to swap triples or add/remove.
    # Since we start with a maximal packing, we can't just add.
    # We must remove one to add another.
    
    while time.time() < deadline:
        # Pick a random triple to remove
        if not current_triples:
            break
        idx_remove = random.randint(0, len(current_triples) - 1)
        t_remove = current_triples[idx_remove]
        
        # Remove it
        current_triples.pop(idx_remove)
        
        # Find valid triples to add
        # Recalculate used pairs? Or just check against current_triples.
        # For speed, let's just check against current_triples directly.
        # Number of triples is small (~20).
        
        valid_candidates = []
        for t in all_triples:
            if t == t_remove:
                continue
            # Check linearity
            is_valid = True
            for t2 in current_triples:
                common = len(set(t) & set(t2))
                if common > 1:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(t)
                
        if not valid_candidates:
            # Re-add the removed one if no candidate found
            current_triples.append(t_remove)
            continue
            
        # Pick a random candidate
        t_add = random.choice(valid_candidates)
        current_triples.append(t_add)
        
        # Evaluate
        new_alpha = calc_alpha(current_triples)
        
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_triples = current_triples[:]
            
        # Acceptance criterion
        # If new_alpha < current_alpha, accept.
        # If new_alpha == current_alpha, accept with prob 0.5 (to explore).
        # If new_alpha > current_alpha, accept with small prob (simulated annealing) or reject.
        # Given the small search space, simple hill climbing with random restarts or swaps might be enough.
        
        # Let's just accept if it's better or equal, to keep the system dense.
        # Actually, keeping the system dense is good for minimizing alpha.
        # So we prefer to keep the number of triples high.
        
        # If we rejected, we should revert?
        # For simplicity, let's just keep the new state if it's not worse, 
        # and occasionally accept worse states to escape local minima.
        
        if new_alpha <= current_alpha or random.random() < 0.1:
            current_alpha = new_alpha
        else:
            # Revert
            current_triples.pop()
            current_triples.append(t_remove)
            # Note: current_alpha remains the same as before the move

    return [list(t) for t in best_triples]
