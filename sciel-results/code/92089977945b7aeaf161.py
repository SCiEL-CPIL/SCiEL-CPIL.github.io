"""Rewritten program for the suite task; maximizes comparable pairs in a 40-subset family on 12 vertices."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

N = 12
M = 40
TIME_LIMIT = 50.0  # Safety margin before 60s kill

def _precompute():
    """Precompute all subsets and comparability matrix."""
    # All 2^12 = 4096 subsets as bitmasks
    all_masks = list(range(1 << N))
    
    # Precompute: for each mask, list of masks that are strict subsets of it
    # And for each mask, list of masks that strictly contain it
    # Actually, we need the comparability relation: A is strict subset of B
    # We'll build a matrix or at least a way to quickly check
    
    # For efficiency, let's precompute for each mask the list of all other masks 
    # that are comparable with it (either strict subset or strict superset)
    # But that's 4096*4096 which is ~16M, might be too much memory/time
    
    # Better approach: precompute subset relations using bit manipulation
    # For a given mask m, its subsets are masks s where (s & m) == s
    # Its supersets are masks s where (s | m) == s
    
    # Let's precompute a sorted list of masks by popcount for the greedy approach
    masks_by_size = [[] for _ in range(N + 1)]
    for m in all_masks:
        pc = bin(m).count('1')
        masks_by_size[pc].append(m)
    
    return all_masks, masks_by_size

def _count_comparable(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        mi = family[i]
        for j in range(i + 1, n):
            mj = family[j]
            # Check if mi is strict subset of mj or mj is strict subset of mi
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _greedy_construct():
    """Greedy construction: start with empty set, then add sets that maximize new comparable pairs."""
    # Strategy: build a chain-like structure
    # Start with the empty set and full set, then add intermediate sets
    
    # Better greedy: iteratively add the set that gives the most new comparable pairs
    all_masks = list(range(1 << N))
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << N)
    for m in range(1, 1 << N):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Start with empty set
    family = [0]
    in_family = [False] * (1 << N)
    in_family[0] = True
    
    # Precompute for each mask, which masks it is comparable with
    # comparable[i][j] = True if mask i and mask j are comparable (one strict subset of other)
    # This is 4096x4096 = 16.7M booleans, ~16MB, should be OK
    # But computing it naively is 4096^2 * 12 ops = ~200M, might be slow
    # Let's use a smarter approach
    
    # For each mask, precompute its subsets and supersets
    # Actually, let's just compute the gain on the fly but efficiently
    
    # Greedy: at each step, pick the mask that maximizes the number of new comparable pairs
    # with the current family
    
    # Precompute: for each mask m, list of masks that are strict subsets of m
    # and list of masks that are strict supersets of m
    # This can be done by iterating over submasks
    
    # For efficiency, let's precompute subset lists
    subsets_of = [[] for _ in range(1 << N)]
    supersets_of = [[] for _ in range(1 << N)]
    
    # For each mask, find all its proper subsets
    # Using the trick: iterate over all submasks of m
    for m in range(1 << N):
        # Find all proper subsets of m
        sub = (m - 1) & m if m > 0 else 0
        while sub:
            subsets_of[m].append(sub)
            sub = (sub - 1) & m
        # Also include 0 if m != 0
        if m != 0:
            subsets_of[m].append(0)
        # Find all proper supersets of m
        # Supersets of m are masks s where (s | m) == s and s != m
        # Equivalently, s = m | t where t is a non-zero subset of ~m (within N bits)
        complement = ((1 << N) - 1) ^ m
        if complement:
            t = complement
            while t:
                s = m | t
                supersets_of[m].append(s)
                t = (t - 1) & complement
    
    # Now greedy construction
    # For each candidate mask, compute how many sets in the current family it's comparable with
    # We want to maximize the sum over all pairs
    
    # To make this efficient, maintain for each mask the count of comparable sets in family
    comparable_count = [0] * (1 << N)
    
    # Initialize: empty set is in family
    # For each mask m, if m is a strict superset of 0 (i.e., m != 0), then m is comparable with 0
    for m in range(1, 1 << N):
        comparable_count[m] = 1  # comparable with empty set
    
    for _ in range(M - 1):
        # Find the mask with max comparable_count that is not in family
        best_mask = -1
        best_count = -1
        for m in range(1 << N):
            if not in_family[m] and comparable_count[m] > best_count:
                best_count = comparable_count[m]
                best_mask = m
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        in_family[best_mask] = True
        
        # Update comparable_count for all other masks
        # For each mask m not in family, if m is comparable with best_mask, increment its count
        # m is comparable with best_mask if m is a strict subset or strict superset of best_mask
        for s in subsets_of[best_mask]:
            if not in_family[s]:
                comparable_count[s] += 1
        for s in supersets_of[best_mask]:
            if not in_family[s]:
                comparable_count[s] += 1
    
    return family

def _local_search(family, time_limit):
    """Local search: try to improve by swapping sets."""
    start_time = time.time()
    
    # Convert to set for fast lookup
    family_set = set(family)
    
    # Precompute comparable_count for each mask relative to current family
    # comparable_count[m] = number of sets in family that are comparable with m
    comparable_count = [0] * (1 << N)
    for m in range(1 << N):
        for f in family:
            if (m & f) == m and m != f:
                comparable_count[m] += 1
            elif (m & f) == f and m != f:
                comparable_count[m] += 1
    
    # Precompute subsets and supersets
    subsets_of = [[] for _ in range(1 << N)]
    supersets_of = [[] for _ in range(1 << N)]
    for m in range(1 << N):
        sub = (m - 1) & m if m > 0 else 0
        while sub:
            subsets_of[m].append(sub)
            sub = (sub - 1) & m
        if m != 0:
            subsets_of[m].append(0)
        complement = ((1 << N) - 1) ^ m
        if complement:
            t = complement
            while t:
                s = m | t
                supersets_of[m].append(s)
                t = (t - 1) & complement
    
    current_score = _count_comparable(family)
    best_family = list(family)
    best_score = current_score
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swapping each set in family with a set not in family
        for i, fi in enumerate(family):
            # Compute the gain from removing fi
            # Removing fi loses its comparable pairs with other family members
            loss = 0
            for j, fj in enumerate(family):
                if i == j:
                    continue
                if (fi & fj) == fi and fi != fj:
                    loss += 1
                elif (fi & fj) == fj and fi != fj:
                    loss += 1
            
            # Try adding each mask not in family
            best_gain = -1
            best_new_mask = -1
            for m in range(1 << N):
                if m in family_set:
                    continue
                # Gain from adding m: comparable_count[m] minus any that overlap with fi removal
                # Actually, we need to compute: how many sets in (family - {fi}) are comparable with m
                # That's comparable_count[m] minus (1 if fi is comparable with m else 0)
                is_comp_with_fi = 0
                if (m & fi) == m and m != fi:
                    is_comp_with_fi = 1
                elif (m & fi) == fi and m != fi:
                    is_comp_with_fi = 1
                
                gain = comparable_count[m] - is_comp_with_fi - loss
                if gain > best_gain:
                    best_gain = gain
                    best_new_mask = m
            
            if best_gain > 0 and best_new_mask != -1:
                # Perform the swap
                family[i] = best_new_mask
                family_set.discard(fi)
                family_set.add(best_new_mask)
                
                # Update comparable_count
                # Remove contributions from fi
                for s in subsets_of[fi]:
                    if s in family_set:
                        comparable_count[s] -= 1
                for s in supersets_of[fi]:
                    if s in family_set:
                        comparable_count[s] -= 1
                # Add contributions from best_new_mask
                for s in subsets_of[best_new_mask]:
                    if s in family_set:
                        comparable_count[s] += 1
                for s in supersets_of[best_new_mask]:
                    if s in family_set:
                        comparable_count[s] += 1
                
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break
    
    return best_family, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    family = _greedy_construct()
    
    # Ensure we have exactly M sets
    if len(family) < M:
        # Fill with random masks
        all_masks = list(range(1 << N))
        family_set = set(family)
        random.seed(42)
        while len(family) < M:
            m = random.choice(all_masks)
            if m not in family_set:
                family.append(m)
                family_set.add(m)
    
    # Phase 2: Local search
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0  # 5s safety margin
    if remaining_time > 2.0:
        family, score = _local_search(family, remaining_time)
    
    # Convert to lists of vertices
    result = []
    for m in family:
        vertices = [b for b in range(N) if (m >> b) & 1]
        result.append(vertices)
    
    return result
# EVOLVE-BLOCK-END
