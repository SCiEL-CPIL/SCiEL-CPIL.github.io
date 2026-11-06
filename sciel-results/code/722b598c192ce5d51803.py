import random
import time

def construct():
    n = 16
    # Pre-compute all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    # Function to calculate independence number
    def calc_alpha(triples):
        # triples is a list of tuples
        # Check all subsets
        # Optimization: If we have the set of triples, we can check if a mask contains a triple.
        # A mask contains a triple (a,b,c) if mask has bits a,b,c set.
        # This is (mask >> a) & 1 ... etc.
        # Or precompute for each triple a bitmask.
        triple_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples ]
        
        max_size = 0
        # Iterate through all masks
        # We can prune: if popcount(mask) <= max_size, skip
        # But simple iteration is fine for 16 bits.
        for mask in range(1 << n):
            # Check if mask is independent
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            if is_indep:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        return max_size

    # Initial random linear hypergraph
    # Greedy construction
    used_pairs = set()
    current_triples = []
    
    # Shuffle triples to get randomness
    random.seed(42) # For reproducibility in dev, but maybe random is better? 
    # Let's try multiple restarts
    
    best_alpha = n
    best_config = []
    
    # Time limit check
    start_time = time.time()
    time_limit = 100 # 100 seconds to be safe
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random linear hypergraph
        # Start with empty
        cur_triples = []
        used_pairs_set = set()
        
        # Shuffle all triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            # Check if any pair is already used
            conflict = False
            for p in pairs:
                if p in used_pairs_set:
                    conflict = True
                    break
            if not conflict:
                # Add triple
                cur_triples.append(t)
                used_pairs_set.update(pairs)
                
        # Now we have a maximal linear hypergraph (random greedy)
        # Evaluate alpha
        alpha = calc_alpha(cur_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_config = cur_triples[:]
            
        # Local Search
        # Try to add triples
        # We need to find triples that can be added without conflict
        # This is complex to do efficiently in local search.
        # Instead, let's do a "swap" or "re-add" strategy?
        # Or just do more random restarts.
        
        # Let's try a simple hill climb on the set of triples
        # Represent state as set of triples
        # Move: Add a random valid triple, or Remove a random triple
        
        # To check if a triple can be added, we need to check its 3 pairs against all existing triples.
        # This is O(N * 3).
        
        # Let's implement a faster check for "can add"
        # We maintain used_pairs_set.
        # If we remove a triple, we remove its pairs from used_pairs_set.
        # If we add a triple, we check if its pairs are in used_pairs_set.
        
        cur_set = set(cur_triples)
        cur_used_pairs = set()
        for t in cur_set:
            a,b,c = t
            cur_used_pairs.add((a,b))
            cur_used_pairs.add((a,c))
            cur_used_pairs.add((b,c))
            
        cur_alpha = alpha
        
        # Hill Climbing
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Try to add a random triple
            # Pick a random triple from all_triples
            # Check if it can be added
            # If yes, add it, recalc alpha (expensive), if better keep, else revert
            
            # To make it faster, we can just do random restarts which are cheap to generate.
            # The greedy generation is O(N^3). 16^3 = 4096. Very fast.
            # So 50 restarts is nothing. Let's do 500 restarts.
            
        # Actually, let's just do more restarts.
        
    # Convert best_config to list of lists
    return [list(t) for t in best_config]
