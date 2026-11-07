"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1
    
    # Precompute comparability: comp[a][b] = 1 if a and b are comparable
    # Two masks a, b are comparable if (a & b) == a or (a & b) == b
    # i.e., a subset of b or b subset of a
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def calc_score(family):
        """Calculate number of comparable unordered pairs in family."""
        score = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    score += 1
        return score
    
    # Strategy: Use a layered approach.
    # Take subsets from a range of sizes to maximize comparabilities.
    # A chain (one set per size) gives m*(m-1)/2 comparable pairs if all are nested.
    # But we can do better by taking multiple sets per layer.
    
    # Initial solution: build a "thickened chain"
    # Take k subsets from each of several consecutive sizes
    # such that total = m
    
    # Let's try different distributions
    best_family = None
    best_score = -1
    
    # Try various size distributions
    # For n=12, sizes 0..12
    # We want to pick 40 subsets from these sizes
    
    # Approach 1: Pure chain - 40 nested sets
    # Max chain length is 13 (sizes 0 to 12), so we can't make a pure chain of 40.
    # Instead, take multiple sets per size.
    
    # Let's use a greedy approach:
    # Start with a good seed, then try to improve by swapping.
    
    # Initial: take subsets from sizes that have many elements
    # C(12, 6) = 924, C(12, 5) = 792, C(12, 7) = 792, etc.
    
    # Let's try: take 40 subsets from a window of sizes
    # If we take from sizes s to s+t, the number of comparable pairs depends on structure
    
    # Simpler: use random initialization + local search
    
    random.seed(42)
    
    # Generate candidate pools from each size
    by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        by_size[bit_counts[mask]].append(mask)
    
    # Initial solution: pick 40 random subsets, biased toward middle sizes
    def random_initial():
        family = []
        # Weight toward middle sizes
        weights = [math.comb(n, s) for s in range(n + 1)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        used = set()
        while len(family) < m:
            r = random.random()
            cum = 0
            size = 0
            for s in range(n + 1):
                cum += probs[s]
                if r < cum:
                    size = s
                    break
            if not by_size[size]:
                continue
            mask = random.choice(by_size[size])
            if mask not in used:
                used.add(mask)
                family.append(mask)
        return family
    
    family = random_initial()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score
    
    # Local search: try swapping one element for another
    # Precompute for speed
    
    # For faster local search, precompute comparability with current family members
    # This is expensive, so we'll be selective
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try random swaps
        idx = random.randint(0, m - 1)
        old_mask = family[idx]
        
        # Pick a new mask from a random size
        new_size = random.randint(0, n)
        if not by_size[new_size]:
            continue
        new_mask = random.choice(by_size[new_size])
        if new_mask == old_mask:
            continue
        # Ensure distinctness
        if new_mask in family:
            continue
        
        # Calculate delta score
        # Remove old_mask: lose comparabilities with other m-1 members
        # Add new_mask: gain comparabilities with other m-1 members
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            other = family[j]
            # Was old_mask comparable with other?
            if is_comparable(old_mask, other):
                delta -= 1
            # Is new_mask comparable with other?
            if is_comparable(new_mask, other):
                delta += 1
        
        if delta > 0:
            family[idx] = new_mask
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1
    
    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
