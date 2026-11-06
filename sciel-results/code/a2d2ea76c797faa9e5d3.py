import time
import random
from itertools import combinations

# Precompute all triples and their pair signatures
ALL_TRIPLES = []
TRIPLE_PAIRS = {}
TRIPLE_MASKS = {}
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            t = (a, b, c)
            pairs = frozenset([(a,b), (a,c), (b,c)])
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append(t)
            TRIPLE_PAIRS[t] = pairs
            TRIPLE_MASKS[t] = mask

# Precompute supersets for each triple mask to speed up alpha calculation
# Actually, we can just iterate masks and check if (mask & t_mask) == t_mask
# But the "mark all supersets" approach is faster for repeated calls if we can reset efficiently.
# However, resetting a 65536 array is fast.
# Let's try the direct check first, it might be fast enough with bit_count.
# Or we can precompute a list of masks for each possible triple mask? No, there are 560 triple masks.
# Let's use the "mark supersets" method inside calculate_alpha.

def calculate_alpha(triples):
    if not triples:
        return 16
    
    # is_ind[i] will be True if subset i is independent
    # We use a bytearray for speed and memory
    is_ind = bytearray([1]) * (1 << 16)
    
    # For each triple, mark all its supersets as non-independent
    for t in triples:
        m = TRIPLE_MASKS[t]
        # Iterate over all supersets of m
        # A superset S of m can be formed by adding any subset of the remaining bits to m
        # remaining_bits = ((1 << 16) - 1) ^ m
        # We can iterate submasks of remaining_bits?
        # No, we want all S such that S & m == m.
        # This is equivalent to S = m | sub, where sub is a submask of ~m.
        # ~m in 16 bits is ((1<<16)-1) ^ m.
        free_bits = ((1 << 16) - 1) ^ m
        
        # Iterate all submasks of free_bits
        sub = free_bits
        while True:
            s = m | sub
            is_ind[s] = 0
            if sub == 0:
                break
            sub = (sub - 1) & free_bits
            
    # Find the largest independent set
    max_alpha = 0
    # We can iterate all masks and check popcount
    # To speed up, we can iterate masks in descending order of popcount?
    # Or just check all. 65536 checks is fast.
    for mask in range(1 << 16):
        if is_ind[mask]:
            # Use bit_count for speed
            c = mask.bit_count()
            if c > max_alpha:
                max_alpha = c
                if max_alpha == 16:
                    break # Can't be better
            
    return max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_alpha = 16
    best_triples = []
    
    random.seed(42)
    
    # Greedy construction function
    def greedy_construct():
        used_pairs = set()
        current_triples = []
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
        return current_triples

    # Initial greedy solution
    current_triples = greedy_construct()
    current_alpha = calculate_alpha(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = current_triples[:]
        
    # Local Search
    # We maintain the set of used pairs to quickly check validity of additions
    used_pairs = set()
    for t in current_triples:
        used_pairs.update(TRIPLE_PAIRS[t])
        
    # Precompute valid triples that can be added given used_pairs?
    # This is dynamic.
    
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Try a random move
        if not current_triples:
            # If empty, just do greedy
            current_triples = greedy_construct()
            current_alpha = calculate_alpha(current_triples)
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
            continue
            
        # Pick a random triple to remove
        idx_out = random.randint(0, len(current_triples) - 1)
        t_out = current_triples[idx_out]
        pairs_out = TRIPLE_PAIRS[t_out]
        
        # Remove it
        current_triples.pop(idx_out)
        used_pairs.difference_update(pairs_out)
        
        # Try to add a random triple
        # We need a triple T_in such that P(T_in) is disjoint from used_pairs
        # We can pick a random triple and check.
        # To make it more likely to find a good one, we might pick from a pre-filtered list?
        # For now, just pick random and check.
        
        t_in = None
        # Try a few random candidates
        for _ in range(10):
            candidate = random.choice(ALL_TRIPLES)
            if candidate == t_out:
                continue
            pairs_in = TRIPLE_PAIRS[candidate]
            valid = True
            for p in pairs_in:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                t_in = candidate
                break
                
        if t_in is not None:
            # Perform the swap
            current_triples.append(t_in)
            used_pairs.update(TRIPLE_PAIRS[t_in])
            
            new_alpha = calculate_alpha(current_triples)
            
            # Acceptance criteria
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha <= 5:
                        break
            elif new_alpha == current_alpha:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.remove(t_in)
                    used_pairs.difference_update(TRIPLE_PAIRS[t_in])
                    current_triples.insert(idx_out, t_out)
                    used_pairs.update(pairs_out)
            else:
                # Worse, accept with probability based on temperature
                prob = pow(2.71828, -(new_alpha - current_alpha) / temperature)
                if random.random() < prob:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.remove(t_in)
                    used_pairs.difference_update(TRIPLE_PAIRS[t_in])
                    current_triples.insert(idx_out, t_out)
                    used_pairs.update(pairs_out)
        else:
            # Could not find a valid replacement, revert
            current_triples.insert(idx_out, t_out)
            used_pairs.update(pairs_out)
            
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
