import time
import random
from itertools import combinations

# Precompute all triples
ALL_TRIPLES = list(combinations(range(16), 3))
TRIPLE_COUNT = len(ALL_TRIPLES)

# Precompute masks for triples
TRIPLE_MASKS = [0] * TRIPLE_COUNT
for i, t in enumerate(ALL_TRIPLES):
    mask = 0
    for x in t:
        mask |= (1 << x)
    TRIPLE_MASKS[i] = mask

# We need to track independence numbers for k = 4, 5, 6, 7, 8
# We will use an array of size 2^16 for counts for each k
# count_hits[k][mask] = number of active triples contained in subset 'mask'
# We only care about masks with exactly k bits set.
# But we can just use the full array.

# Precompute list of subset masks for each k?
# Or just iterate?
# For a triple T (mask mT), we need to iterate over all masks mS such that mS & mT == mT and popcount(mS) == k.
# This is equivalent to choosing k-3 bits from the remaining 13 positions.

# To make updates fast, we precompute for each triple index i, and for each k,
# the list of subset masks that contain triple i.
# This might be memory heavy?
# 560 triples * 5 k-values * 1287 subsets (max) ~ 3.6 million integers.
# 3.6M * 8 bytes ~ 28MB. Acceptable.

# Let's precompute:
# SUPERSETS[i][k] = list of masks of subsets of size k containing triple i
SUPERSETS = [[[] for _ in range(9)] for _ in range(TRIPLE_COUNT)]

# To generate these, we can iterate over all subsets of size k, and for each triple contained in it, add the subset mask to the triple's list.
# This is more efficient to build.
# Total subsets of size 8: 12870.
# Each subset contains 56 triples.
# 12870 * 56 ~ 720k operations.

# Initialize count_hits
# We will store count_hits as a list of lists of ints, indexed by [k][mask]
# k ranges from 4 to 8.
# Let's use 0-indexed k in the list, so index k-4.
K_MIN = 4
K_MAX = 8
NUM_K = K_MAX - K_MIN + 1

count_hits = [[0] * (1 << 16) for _ in range(NUM_K)]
num_independent = [0] * NUM_K

# Initialize num_independent
# For each k, num_independent[k] = number of subsets of size k
# which is C(16, k)
from math import comb
for i in range(NUM_K):
    k = K_MIN + i
    num_independent[i] = comb(16, k)

# Build SUPERSETS
# Iterate over all subsets of size k
# For each subset, find all triples contained in it
# Add subset mask to SUPERSETS[triple_idx][k]

# To find triples in a subset, we can iterate over all triples and check?
# 12870 subsets * 560 triples = 7M checks. Fast enough.

# Alternatively, iterate over triples, and generate supersets.
# For each triple i:
#   For each k:
#     Generate all combinations of k-3 points from the 13 points not in triple i.
#     Form mask.
#     Append to SUPERSETS[i][k].

# This is 560 * 5 * 1287 ~ 3.6M operations.
# Let's do this.

# Points not in triple i
for i, t in enumerate(ALL_TRIPLES):
    t_set = set(t)
    remaining_points = [p for p in range(16) if p not in t_set]
    t_mask = TRIPLE_MASKS[i]
    
    for k in range(K_MIN, K_MAX + 1):
        # Choose k-3 points from remaining_points
        # remaining_points has size 13
        need = k - 3
        if need < 0:
            continue
        if need > len(remaining_points):
            continue
            
        for comb_pts in combinations(remaining_points, need):
            # Construct mask
            mask = t_mask
            for p in comb_pts:
                mask |= (1 << p)
            SUPERSETS[i][k - K_MIN].append(mask)

def add_triple(i):
    """Add triple i to the system. Updates counters."""
    for k_idx in range(NUM_K):
        k = K_MIN + k_idx
        for mask in SUPERSETS[i][k_idx]:
            if count_hits[k_idx][mask] == 0:
                num_independent[k_idx] -= 1
            count_hits[k_idx][mask] += 1

def remove_triple(i):
    """Remove triple i from the system. Updates counters."""
    for k_idx in range(NUM_K):
        k = K_MIN + k_idx
        for mask in SUPERSETS[i][k_idx]:
            count_hits[k_idx][mask] -= 1
            if count_hits[k_idx][mask] == 0:
                num_independent[k_idx] += 1

def get_alpha():
    """Returns the current independence number."""
    # Check from largest k down
    for k_idx in range(NUM_K - 1, -1, -1):
        if num_independent[k_idx] > 0:
            return K_MIN + k_idx
    return K_MIN - 1 # Should not happen, as single points are independent

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_alpha = 16
    best_triples = []
    
    # Greedy initialization
    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(TRIPLE_COUNT))
        random.shuffle(indices)
        triples = []
        used_pairs = set()
        
        for idx in indices:
            a, b, c = ALL_TRIPLES[idx]
            if ((a, b) not in used_pairs and 
                (a, c) not in used_pairs and 
                (b, c) not in used_pairs):
                triples.append(idx)
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
        return triples

    # Run a few greedy restarts
    for s in range(5):
        if time.time() > deadline:
            break
        triples = greedy_construct(s)
        # Reset counters
        for k_idx in range(NUM_K):
            count_hits[k_idx] = [0] * (1 << 16)
            num_independent[k_idx] = comb(16, K_MIN + k_idx)
        
        for idx in triples:
            add_triple(idx)
            
        alpha = get_alpha()
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples.copy()

    if not best_triples:
        best_triples = greedy_construct(0)
        # Reset and add
        for k_idx in range(NUM_K):
            count_hits[k_idx] = [0] * (1 << 16)
            num_independent[k_idx] = comb(16, K_MIN + k_idx)
        for idx in best_triples:
            add_triple(idx)
        best_alpha = get_alpha()

    current_triples = best_triples.copy()
    current_alpha = best_alpha
    
    # Local Search
    # We maintain the set of active triple indices as a set for O(1) lookup
    active_set = set(current_triples)
    
    # Precompute valid candidates?
    # We can just pick random triples and check validity against active_set.
    # To check validity, we need to check pairs.
    # We can maintain a set of used pairs.
    used_pairs = set()
    for idx in current_triples:
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.add((a, b))
        used_pairs.add((a, c))
        used_pairs.add((b, c))

    def is_valid(idx):
        if idx in active_set:
            return False
        a, b, c = ALL_TRIPLES[idx]
        return ((a, b) not in used_pairs and 
                (a, c) not in used_pairs and 
                (b, c) not in used_pairs)

    def add_to_state(idx):
        active_set.add(idx)
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.add((a, b))
        used_pairs.add((a, c))
        used_pairs.add((b, c))
        add_triple(idx)

    def remove_from_state(idx):
        active_set.remove(idx)
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.remove((a, b))
        used_pairs.remove((a, c))
        used_pairs.remove((b, c))
        remove_triple(idx)

    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 0.01
    
    while time.time() < deadline:
        if temperature < min_temperature:
            break
            
        # Generate a move
        move_type = random.random()
        new_alpha = current_alpha
        
        if move_type < 0.5:
            # Try to add a triple
            # Pick random candidates until valid
            for _ in range(10):
                cand_idx = random.randint(0, TRIPLE_COUNT - 1)
                if is_valid(cand_idx):
                    # Tentatively add
                    add_to_state(cand_idx)
                    new_alpha = get_alpha()
                    # Check if we should accept
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = list(active_set)
                        # Keep the move
                        break
                    elif new_alpha == current_alpha:
                        if random.random() < 0.5:
                            current_alpha = new_alpha
                            break
                        else:
                            remove_from_state(cand_idx)
                    else:
                        # Accept with prob exp(-delta/T)
                        delta = new_alpha - current_alpha
                        if delta == 1:
                            prob = 0.1
                        else:
                            prob = 0.0
                        if random.random() < prob:
                            current_alpha = new_alpha
                            break
                        else:
                            remove_from_state(cand_idx)
                    break # Only try one candidate per iteration for speed? Or keep trying?
                    # If we rejected, we removed it. We can try another candidate.
                    # But to keep it simple, let's just do one attempt per loop iteration.
                    # Actually, if we reject, we should try a different candidate?
                    # Let's just break and try again in next loop.
                    pass 
            # If we didn't break, we either added and accepted, or added and rejected (and removed).
            # If we added and accepted, state is updated.
            # If we added and rejected, state is reverted.
            # If no valid candidate found, state unchanged.

        elif move_type < 0.8:
            # Try to remove a triple
            if len(active_set) > 0:
                cand_idx = random.choice(list(active_set))
                remove_from_state(cand_idx)
                new_alpha = get_alpha()
                
                # Removing a triple usually INCREASES alpha or keeps it same.
                # We only accept if alpha decreases (unlikely) or stays same (to explore) with low prob?
                # Actually, removing a triple might allow adding better ones later.
                # But direct acceptance criterion:
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = list(active_set)
                elif new_alpha == current_alpha:
                    if random.random() < 0.1:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        add_to_state(cand_idx)
                else:
                    # Revert
                    add_to_state(cand_idx)

        else:
            # Swap: Remove one, add one
            if len(active_set) > 0:
                rem_idx = random.choice(list(active_set))
                remove_from_state(rem_idx)
                
                # Try to add a valid triple
                for _ in range(10):
                    add_idx = random.randint(0, TRIPLE_COUNT - 1)
                    if is_valid(add_idx):
                        add_to_state(add_idx)
                        new_alpha = get_alpha()
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = list(active_set)
                            break
                        elif new_alpha == current_alpha:
                            if random.random() < 0.5:
                                current_alpha = new_alpha
                                break
                            else:
                                remove_from_state(add_idx)
                                add_to_state(rem_idx) # Revert
                        else:
                            delta = new_alpha - current_alpha
                            prob = 0.1 if delta == 1 else 0.0
                            if random.random() < prob:
                                current_alpha = new_alpha
                                break
                            else:
                                remove_from_state(add_idx)
                                add_to_state(rem_idx) # Revert
                        break
                else:
                    # No valid add found, revert remove
                    add_to_state(rem_idx)

        temperature *= cooling_rate

    return [list(ALL_TRIPLES[i]) for i in best_triples]
