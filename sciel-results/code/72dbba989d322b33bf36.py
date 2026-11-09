import time
import random
import math
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append([i, j, k])
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute subsets by size for independence check
# We only need to check subsets of size >= 1
# But we check from large to small to find max alpha quickly
SUBSETS_BY_SIZE = {k: [] for k in range(1, 17)}
for mask in range(1, 1 << 16):
    # Count bits
    # Python 3.10+ has int.bit_count()
    # Assuming modern Python
    bc = bin(mask).count('1')
    if bc <= 16:
        SUBSETS_BY_SIZE[bc].append(mask)

def is_independent(mask, H_list):
    """
    Check if the set of vertices represented by mask is independent
    given the list of triple indices H_list.
    """
    for idx in H_list:
        if (ALL_TRIPLE_MASKS[idx] & mask) == ALL_TRIPLE_MASKS[idx]:
            return False
    return True

def compute_alpha(H_indices):
    """
    Compute the independence number of the hypergraph defined by H_indices.
    H_indices is a set of indices into ALL_TRIPLES.
    """
    if not H_indices:
        return 16
    
    H_list = list(H_indices)
    
    # Check from largest possible size down
    for k in range(16, 0, -1):
        for mask in SUBSETS_BY_SIZE[k]:
            if is_independent(mask, H_list):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_alpha = 16
    best_system = []
    
    # Parameters for Simulated Annealing
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    max_iterations = 5000
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # 1. Random Greedy Packing to start
        H_indices = set()
        used_pairs_set = set()
        
        # Shuffle indices for randomization
        indices = list(range(NUM_TRIPLES))
        random.shuffle(indices)
        
        for idx in indices:
            i, j, k = ALL_TRIPLES[idx]
            norm_pairs = [tuple(sorted(p)) for p in ((i, j), (i, k), (j, k))]
            if all(p not in used_pairs_set for p in norm_pairs):
                H_indices.add(idx)
                for p in norm_pairs:
                    used_pairs_set.add(p)
        
        current_alpha = compute_alpha(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [ALL_TRIPLES[i] for i in H_indices]
            
        # 2. Simulated Annealing
        temp = initial_temp
        
        # Precompute list of valid moves? No, dynamic.
        # We will try to add, remove, or swap.
        
        while temp > min_temp and time.time() < deadline:
            # Choose a move type
            move_type = random.random()
            
            # Get a random candidate index
            # For add/swap, we need a valid triple not in H
            # For remove, we need a triple in H
            
            if move_type < 0.3:
                # Try to add a triple
                # Pick random idx not in H
                # Optimization: just pick random idx from ALL, check if in H
                idx = random.randint(0, NUM_TRIPLES - 1)
                if idx not in H_indices:
                    i, j, k = ALL_TRIPLES[idx]
                    norm_pairs = [tuple(sorted(p)) for p in ((i, j), (i, k), (j, k))]
                    if all(p not in used_pairs_set for p in norm_pairs):
                        # Valid move
                        H_indices.add(idx)
                        for p in norm_pairs:
                            used_pairs_set.add(p)
                        
                        new_alpha = compute_alpha(H_indices)
                        delta = new_alpha - current_alpha
                        
                        # Accept if better or with probability exp(-delta/temp)
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [ALL_TRIPLES[i] for i in H_indices]
                        else:
                            # Revert
                            H_indices.remove(idx)
                            for p in norm_pairs:
                                used_pairs_set.discard(p)
                        
            elif move_type < 0.5:
                # Try to remove a triple
                if H_indices:
                    rem_idx = random.choice(list(H_indices))
                    H_indices.remove(rem_idx)
                    ri, rj, rk = ALL_TRIPLES[rem_idx]
                    rem_norm_pairs = [tuple(sorted(p)) for p in ((ri, rj), (ri, rk), (rj, rk))]
                    for p in rem_norm_pairs:
                        used_pairs_set.discard(p)
                    
                    new_alpha = compute_alpha(H_indices)
                    delta = new_alpha - current_alpha
                    
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [ALL_TRIPLES[i] for i in H_indices]
                    else:
                        # Revert
                        H_indices.add(rem_idx)
                        for p in rem_norm_pairs:
                            used_pairs_set.add(p)
                        
            else:
                # Try to swap: remove one, add another
                if H_indices:
                    rem_idx = random.choice(list(H_indices))
                    # Find a valid add_idx
                    # We can just pick random and check validity
                    # To increase chance of success, we might retry a few times
                    found_add = False
                    for _ in range(10):
                        add_idx = random.randint(0, NUM_TRIPLES - 1)
                        if add_idx == rem_idx:
                            continue
                        ai, aj, ak = ALL_TRIPLES[add_idx]
                        # Check if add_idx is valid given rem_idx is removed
                        # Pairs of rem_idx are freed
                        ri, rj, rk = ALL_TRIPLES[rem_idx]
                        rem_norm_pairs = [tuple(sorted(p)) for p in ((ri, rj), (ri, rk), (rj, rk))]
                        
                        # Temporarily assume rem_idx is removed
                        # Check if add_idx conflicts with remaining H
                        # This is slightly complex. Easier:
                        # 1. Remove rem_idx from H and used_pairs
                        # 2. Check if add_idx is valid
                        # 3. If yes, add it.
                        
                        # Let's do the full swap logic carefully
                        # We already identified rem_idx.
                        break # We will do the swap logic below in a cleaner way
                    
                    # Cleaner Swap Logic
                    H_indices.remove(rem_idx)
                    ri, rj, rk = ALL_TRIPLES[rem_idx]
                    rem_norm_pairs = [tuple(sorted(p)) for p in ((ri, rj), (ri, rk), (rj, rk))]
                    for p in rem_norm_pairs:
                        used_pairs_set.discard(p)
                    
                    # Now try to add a random triple
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx not in H_indices:
                        ai, aj, ak = ALL_TRIPLES[add_idx]
                        add_norm_pairs = [tuple(sorted(p)) for p in ((ai, aj), (ai, ak), (aj, ak))]
                        if all(p not in used_pairs_set for p in add_norm_pairs):
                            H_indices.add(add_idx)
                            for p in add_norm_pairs:
                                used_pairs_set.add(p)
                            
                            new_alpha = compute_alpha(H_indices)
                            delta = new_alpha - current_alpha
                            
                            if delta <= 0 or random.random() < math.exp(-delta / temp):
                                current_alpha = new_alpha
                                if current_alpha < best_alpha:
                                    best_alpha = current_alpha
                                    best_system = [ALL_TRIPLES[i] for i in H_indices]
                            else:
                                # Revert add, keep rem removed? No, revert whole swap
                                H_indices.remove(add_idx)
                                for p in add_norm_pairs:
                                    used_pairs_set.discard(p)
                                H_indices.add(rem_idx)
                                for p in rem_norm_pairs:
                                    used_pairs_set.add(p)
                        else:
                            # Invalid add, revert removal
                            H_indices.add(rem_idx)
                            for p in rem_norm_pairs:
                                used_pairs_set.add(p)
                    else:
                        # Invalid add (already in H), revert removal
                        H_indices.add(rem_idx)
                        for p in rem_norm_pairs:
                            used_pairs_set.add(p)
            
            temp *= cooling_rate
            
            # Check time
            if time.time() > deadline:
                break
        
        # End SA for this restart

    return best_system

def _fallback_system():
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

def construct_safe():
    try:
        res = construct()
        if not res:
            return _fallback_system()
        # Validate linearity just in case
        used = set()
        for t in res:
            i,j,k = t
            pairs = [tuple(sorted((i,j))), tuple(sorted((i,k))), tuple(sorted((j,k)))]
            for p in pairs:
                if p in used:
                    return _fallback_system()
                used.add(p)
        return res
    except:
        return _fallback_system()

construct = construct_safe
