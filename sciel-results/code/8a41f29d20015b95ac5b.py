import time
import random
from itertools import combinations

# Precompute all triples and their pair signatures
ALL_TRIPLES = []
TRIPLE_PAIRS = {}
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            t = (a, b, c)
            pairs = frozenset([(a,b), (a,c), (b,c)])
            ALL_TRIPLES.append(t)
            TRIPLE_PAIRS[t] = pairs

def calculate_alpha(triples):
    """
    Calculate the independence number of the hypergraph defined by triples.
    Returns the size of the largest subset of vertices containing no triple.
    """
    if not triples:
        return 16
        
    # Convert triples to bitmasks for fast subset checking
    triple_masks = []
    for a, b, c in triples:
        mask = (1 << a) | (1 << b) | (1 << c)
        triple_masks.append(mask)
    
    # We check for independent sets of size k from 16 down to 1
    # An independent set S is one where no triple is a subset of S.
    # i.e., for all t in triple_masks, (t & S) != t
    
    # To speed up, we can precompute the list of subsets for each size?
    # Or just iterate all masks?
    # 2^16 = 65536 masks.
    # For each mask, check against ~40 triples.
    # Total ~2.6M ops.
    
    # Optimization: Iterate masks in descending order of bit count?
    # Python's range(65535, -1, -1) doesn't order by bit count.
    # We can generate combinations.
    
    # Let's use the combination approach for k from 16 down to 1.
    # This stops early if we find an independent set of large size.
    
    # Precompute all subsets of size k?
    # Or just check all 2^16 subsets once and track max size?
    # Checking all 2^16 is simpler and fast enough.
    
    max_alpha = 0
    # Iterate through all possible subsets
    # We can prune: if a subset S is not independent, any superset is not independent.
    # But checking all is simple.
    
    # Let's try the combination method which allows early termination for large k.
    for k in range(16, 0, -1):
        found = False
        for subset in combinations(range(16), k):
            # Create mask for subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if any triple is contained in this subset
            is_independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                max_alpha = k
                found = True
                break
        
        if found:
            break
            
    return max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_alpha = 16
    best_triples = []
    
    # Number of restarts
    # Each restart is very fast.
    # Let's do as many as possible within time.
    
    random.seed(42) # Fixed seed for reproducibility in testing, but random in production? 
    # The prompt implies we should find the best. Randomness helps exploration.
    # We will use random restarts.
    
    for i in range(10000):
        if time.time() > deadline:
            break
            
        # Greedy construction
        used_pairs = set()
        current_triples = []
        
        # Shuffle all triples
        shuffled_triples = ALL_TRIPLES[:]
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            pairs = TRIPLE_PAIRS[t]
            # Check if any pair is already used
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                current_triples.append(t)
                used_pairs.update(pairs)
                
        # Calculate alpha
        alpha = calculate_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples
            
            # If we hit a very low alpha, we can stop or continue to see if lower is possible
            # Theoretical minimum for n=16 is likely 5 or 6.
            if best_alpha <= 5:
                break

    # If no triples were found (should not happen), return seed
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
