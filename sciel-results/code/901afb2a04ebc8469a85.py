import time
import random
import itertools

def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    Uses random greedy search to find packings with small independence number.
    """
    n = 16
    points = list(range(n))
    
    # Pre-generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    
    def is_valid_addition(triple, existing_triples):
        t_set = set(triple)
        for e in existing_triples:
            e_set = set(e)
            if len(t_set.intersection(e_set)) > 1:
                return False
        return True

    def compute_independence_number(triples):
        if not triples:
            return n
        
        # Represent triples as bitmasks
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # is_independent[mask] will be True if no triple is subset of mask
        # We can compute this efficiently.
        # A mask is independent if it does NOT contain any triple_mask.
        # Let's initialize all as True, then mark as False those containing a triple.
        # Actually, it's easier to check for each mask if it contains a triple.
        # But 2^16 is small.
        
        # Optimization: 
        # If a mask M contains a triple T, it is not independent.
        # We want to find the max popcount M such that no T is subset of M.
        
        # Let's just iterate k from n down to 1.
        # For each k, check if there is a subset of size k that is independent.
        
        # To check if a specific subset S (mask) is independent:
        # Check if any triple_mask is subset of S.
        # triple_mask & S == triple_mask
        
        # We can precompute for each mask if it is "bad" (contains a triple).
        # bad[mask] = True if any triple is subset of mask.
        
        # We can fill bad[] by iterating through triples and their supersets?
        # Or just check on the fly for candidates.
        
        # Given n=16, we can just iterate all masks and check.
        # 65536 masks * (len(triples) checks).
        # If len(triples) ~ 30, 65536 * 30 ~ 2M ops. Fast.
        
        max_alpha = 0
        for k in range(n, 0, -1):
            found = False
            # Iterate all masks with popcount k
            # Generating combinations is faster than iterating all masks and checking popcount
            for combo in itertools.combinations(points, k):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                
                is_ind = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_ind = False
                        break
                if is_ind:
                    max_alpha = k
                    found = True
                    break # No need to check other subsets of size k if we found one? 
                    # Wait, we want to know IF there exists one. 
                    # If we found one, alpha >= k. 
                    # Since we iterate k downwards, the first k we find is the max.
                    break
            
            if found:
                break
                
        return max_alpha

    best_alpha = n
    best_triples = []
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Seed random
    random.seed(42)
    
    # Run random restarts
    while time.time() < deadline:
        # Greedy construction
        triples = []
        # Shuffle triples to randomize the greedy choice
        shuffled_triples = all_triples.copy()
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            if is_valid_addition(t, triples):
                triples.append(t)
                
        # Calculate independence number
        alpha = compute_independence_number(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            # If we hit a very low alpha, we might stop early?
            # Theoretical min is likely 4 or 5.
            if best_alpha <= 5:
                break

    return best_triples

# EVOLVE-BLOCK-END
