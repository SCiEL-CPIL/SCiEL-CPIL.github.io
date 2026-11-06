import time
import itertools
import random
import math

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute pair masks for each triple
# A triple (a, b, c) has pairs (a,b), (a,c), (b,c)
# We represent a pair (u, v) with u < v as a single integer u * 16 + v or similar.
# Or better, use a bitmask of size 16*16 = 256 bits? No, just use a set of frozensets or a bitmask of 128 pairs?
# There are 16*15/2 = 120 pairs. We can use a 128-bit integer or a Python int as a bitmask.
# Pair index for (u, v) with u < v: u*15 - (u*(u+1)//2) + v - u - 1?
# Simpler: map (u, v) to u*16+v and v*16+u? No, just u*16+v for u<v.
# Let's use a dictionary or just a set of pairs for simplicity in the search, 
# but for speed in local search, a bitmask is better.
# Let's use a Python integer as a bitmask for the 120 pairs.
# Index for pair (u, v) where u < v:
# We can just use a tuple (min(u,v), max(u,v)) in a set.
# For 560 triples, checking disjointness of sets of 3 pairs is fast.

TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    a, b, c = t
    TRIPLE_PAIRS.append(frozenset([(a, b), (a, c), (b, c)]))

TRIPLE_MASKS = [sum(1 << x for x in t) for t in ALL_TRIPLES]
UNIVERSE_MASK = (1 << N) - 1

# Precompute popcount for all 16-bit masks
POP = [0] * (1 << N)
for i in range(1, 1 << N):
    POP[i] = POP[i >> 1] + (i & 1)

def calc_alpha_from_masks(masks):
    """
    Computes the independence number.
    """
    if not masks:
        return N
        
    # independent[i] is True if subset i is independent
    # We start with all True, and mark False for any set containing a triple.
    independent = bytearray([1]) * (1 << N)
    
    for m in masks:
        # Iterate all supersets of m
        # Supersets of m are m | z where z is a subset of (~m & UNIVERSE_MASK)
        y = (~m) & UNIVERSE_MASK
        z = y
        while True:
            independent[m | z] = 0
            if z == 0:
                break
            z = (z - 1) & y
        
    max_alpha = 0
    for i in range(1 << N):
        if independent[i]:
            c = POP[i]
            if c > max_alpha:
                max_alpha = c
    return max_alpha

def greedy_construct(seed_val):
    """
    Constructs a random maximal linear triple system.
    """
    random.seed(seed_val)
    indices = list(range(NUM_TRIPLES))
    random.shuffle(indices)
    
    triples = []
    used_pairs = set()
    
    for idx in indices:
        pairs = TRIPLE_PAIRS[idx]
        if pairs.isdisjoint(used_pairs):
            triples.append(idx)
            used_pairs |= pairs
            
    return triples

def local_search(initial_triples, seed_val, time_limit):
    """
    Improves the triple system using Simulated Annealing.
    """
    random.seed(seed_val)
    start_time = time.time()
    
    current_triples = list(initial_triples)
    current_alpha = calc_alpha_from_masks([TRIPLE_MASKS[i] for i in current_triples])
    
    best_triples = list(current_triples)
    best_alpha = current_alpha
    
    # Precompute set of current triples for fast lookup?
    # current_set = set(current_triples)
    
    # Temperature parameters
    T = 0.5
    cooling = 0.95
    min_T = 0.01
    
    # We will try to swap triples.
    # To make it faster, we can maintain a list of "available" triples?
    # No, just pick random indices.
    
    # Precompute pair sets for all triples for fast disjoint check?
    # We have TRIPLE_PAIRS.
    
    # To speed up disjoint check, we can maintain a global pair count or mask.
    # But since we are swapping, we just need to check if new triple is disjoint from (current - old).
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick a random triple to remove
        if not current_triples:
            break
        out_idx_pos = random.randint(0, len(current_triples) - 1)
        out_idx = current_triples[out_idx_pos]
        
        # Pick a random triple to add
        in_idx = random.randint(0, NUM_TRIPLES - 1)
        if in_idx == out_idx:
            continue
            
        # Check if in_idx is already in current_triples
        # Linear search is O(K) where K is number of triples. K is at most 560/3 ~ 180.
        # This is acceptable.
        if in_idx in current_triples:
            continue
            
        # Check if in_idx is linear with (current_triples - {out_idx})
        # We need to check if pairs of in_idx are disjoint from pairs of all other triples in current_triples.
        in_pairs = TRIPLE_PAIRS[in_idx]
        out_pairs = TRIPLE_PAIRS[out_idx]
        
        # Optimization: If in_pairs is not disjoint from out_pairs, it doesn't matter 
        # because out_pairs is being removed. But if in_pairs intersects with ANY other pair, it's invalid.
        
        valid = True
        for pos, t_idx in enumerate(current_triples):
            if t_idx == out_idx:
                continue
            if not in_pairs.isdisjoint(TRIPLE_PAIRS[t_idx]):
                valid = False
                break
                
        if not valid:
            continue
            
        # Perform swap
        new_triples = list(current_triples)
        new_triples[out_idx_pos] = in_idx
        
        new_alpha = calc_alpha_from_masks([TRIPLE_MASKS[i] for i in new_triples])
        
        delta = new_alpha - current_alpha
        
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
            current_triples = new_triples
            current_alpha = new_alpha
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = list(current_triples)
                
        T *= cooling
        
    return best_triples, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_alpha = N + 1
    best_triples = []
    
    num_restarts = 0
    
    # Phase 1: Greedy + Local Search
    while time.time() < deadline:
        seed = num_restarts
        greedy_triples = greedy_construct(seed)
        
        # Allocate time for local search
        # If we have plenty of time, give more to local search.
        # Let's say 1.0s per local search iteration initially, decreasing?
        # Or just fixed 0.5s?
        # Let's try 0.5s for local search.
        ls_time = 0.5
        if time.time() + ls_time > deadline:
            ls_time = max(0.1, deadline - time.time())
            
        ls_triples, ls_alpha = local_search(greedy_triples, seed, ls_time)
        
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples
            
        if best_alpha <= 5:
            break
            
        num_restarts += 1
        
    # Convert indices to triples
    return [ALL_TRIPLES[i] for i in best_triples]

# Run a quick self-test if executed directly?
# No, just define the function.

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Triples: {len(res)}")
    if res:
        masks = [sum(1 << x for x in t) for t in res]
        alpha = calc_alpha_from_masks(masks)
        print(f"Alpha: {alpha}")
