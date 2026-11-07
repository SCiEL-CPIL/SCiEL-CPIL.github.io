"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute bit counts
    bit_counts = [0] * N
    for i in range(1, N):
        bit_counts[i] = bit_counts[i >> 1] + (i & 1)
    
    # Precompute subset relation table: subset[a][b] = 1 if a is strict subset of b
    # Using a 2D list of bytes for memory efficiency and speed
    # Actually, let's use a flat array or just check (a & b) == a and a != b
    # For speed in the inner loop, let's precompute a set of supersets for each mask
    # But that might be too much memory. Let's use the bitwise check which is O(1).
    
    def count_comparable(family):
        """Count unordered pairs (i, j) with i < j where family[i] < family[j] or family[j] < family[i]"""
        cnt = 0
        for i in range(m):
            mi = family[i]
            for j in range(i + 1, m):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    cnt += 1
                elif (mj & mi) == mj and mj != mi:
                    cnt += 1
        return cnt
    
    # Initial family: create a strong starting point using chains
    # We want 40 distinct subsets of 12 vertices.
    # A good strategy is to create several chains of subsets.
    # For example, for each subset of a small core, extend it.
    # Or simply: take all subsets of size 0, 1, ..., k, and some of size k+1.
    
    # Let's generate a random initial family and then hill climb.
    # But a structured initial family is better.
    
    # Strategy: Use a base set of "small" sets and "large" sets.
    # To maximize comparability, we want many sets to be subsets of others.
    # This is like finding a large antichain's complement... no, we want the OPPOSITE.
    # We want the family to be as "chain-like" as possible.
    
    # Let's try to construct a family that is a union of chains.
    # For n=12, the maximum size of a chain is 13 (from empty set to full set).
    # We need 40 sets.
    
    # Let's use a greedy construction:
    # 1. Start with empty set.
    # 2. Add sets that are supersets of existing sets, preferring those that allow more extensions.
    
    # Actually, a simpler heuristic:
    # Randomly generate 40 distinct subsets, then hill climb.
    # But let's try a smarter initialization.
    
    # Let's use the following:
    # For each i in 0..11, let S_i be the set {0, 1, ..., i}. These form a chain of length 12.
    # We need 28 more.
    # Let's add sets that are subsets of the full set but not in the chain.
    
    # Better: Let's use a random restart hill climbing with a good initial guess.
    
    def random_family():
        """Generate a random family of 40 distinct subsets."""
        family = set()
        while len(family) < m:
            mask = random.randint(0, N - 1)
            family.add(mask)
        return list(family)
    
    def local_search(family, max_iter=10000):
        """Hill climbing: try to replace a set with another to increase score."""
        family = list(family)
        current_score = count_comparable(family)
        family_set = set(family)
        
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try swapping each element with a random other mask
            for i in range(m):
                old_mask = family[i]
                # Try a few random new masks
                for _ in range(5):
                    new_mask = random.randint(0, N - 1)
                    if new_mask in family_set:
                        continue
                    
                    # Compute delta score
                    # Remove old_mask, add new_mask
                    # This is O(m) per trial, so 5*m*m per iteration = 5*1600 = 8000 ops
                    # 10000 iters * 8000 = 80M ops. Might be too slow in Python.
                    
                    # Let's optimize: precompute the contribution of each mask to the total score.
                    # But that's complex. Let's just do it naively but with a tighter loop.
                    
                    # Actually, let's just do a full score calculation for the candidate family.
                    # That's O(m^2) = 1600 per trial. 5*40*1600 = 320,000 per iteration.
                    # 10000 iters = 3.2B ops. Too slow.
                    
                    # We need a faster delta calculation.
                    break  # Skip this for now, use a simpler approach
                
                # Simpler: just replace and rescore, but limit trials
                # Let's try replacing one element at a time with a random mask
                new_mask = random.randint(0, N - 1)
                if new_mask in family_set:
                    continue
                
                # Temporarily swap
                family[i] = new_mask
                family_set.remove(old_mask)
                family_set.add(new_mask)
                
                new_score = count_comparable(family)
                
                if new_score > current_score:
                    current_score = new_score
                    old_mask = new_mask
                    improved = True
                else:
                    # Revert
                    family[i] = old_mask
                    family_set.remove(new_mask)
                    family_set.add(old_mask)
        
        return family, current_score
    
    # Let's use a faster scoring method with precomputed subset relations
    # Precompute: for each mask, the list of masks in the family that are its supersets/subsets
    # This is dynamic, so we can't precompute statically.
    
    # Alternative: Use bitsets for the family? No, family changes.
    
    # Let's try a different approach: Simulated Annealing or just more iterations of simple hill climbing.
    
    # Given the time constraint, let's use a simpler but effective method:
    # 1. Generate a structured initial family
    # 2. Do hill climbing with efficient delta computation
    
    # Efficient delta computation:
    # When we remove mask A and add mask B:
    # Delta = (contributions of B with rest) - (contributions of A with rest)
    # Where contribution of X with Y is 1 if X < Y or Y < X
    
    def compute_contributions(family):
        """For each mask in family, compute how many other masks it is comparable with."""
        m = len(family)
        contribs = [0] * m
        for i in range(m):
            mi = family[i]
            for j in range(i + 1, m):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    contribs[i] += 1
                    contribs[j] += 1
                elif (mj & mi) == mj and mj != mi:
                    contribs[i] += 1
                    contribs[j] += 1
        return contribs
    
    def fast_swap(family, contribs, i, new_mask):
        """Compute new score after replacing family[i] with new_mask."""
        old_mask = family[i]
        # Remove contributions of old_mask
        delta = -contribs[i]
        # Add contributions of new_mask with all other masks
        for j in range(len(family)):
            if j == i:
                continue
            mj = family[j]
            if (new_mask & mj) == new_mask and new_mask != mj:
                delta += 1
            elif (mj & new_mask) == mj and mj != new_mask:
                delta += 1
        return delta
    
    # Let's implement a proper local search with this.
    
    best_family = None
    best_score = 0
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        
        # Generate initial family
        if trial == 0:
            # Structured: chains
            family = []
            # Chain 1: {0}, {0,1}, {0,1,2}, ..., {0..11}
            for i in range(n + 1):
                mask = (1 << (i + 1)) - 1
                family.append(mask)
            # Need 40 - 13 = 27 more
            # Add random sets that are likely to be comparable
            existing = set(family)
            while len(family) < m:
                mask = random.randint(0, N - 1)
                if mask not in existing:
                    family.append(mask)
                    existing.add(mask)
        else:
            family = random_family()
        
        contribs = compute_contributions(family)
        current_score = sum(contribs) // 2  # Each pair counted twice
        family_set = set(family)
        
        # Local search
        no_improve = 0
        max_no_improve = 500
        while no_improve < max_no_improve:
            if time.time() > deadline - 5:
                break
            
            # Pick a random index to swap
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Try a few candidate new masks
            best_delta = 0
            best_new_mask = None
            
            for _ in range(20):
                new_mask = random.randint(0, N - 1)
                if new_mask in family_set:
                    continue
                
                delta = fast_swap(family, contribs, i, new_mask)
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask is not None and best_delta > 0:
                # Apply the swap
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                family[i] = best_new_mask
                
                # Update contribs
                # This is tricky. We need to recompute contribs for all affected masks.
                # For simplicity, let's just recompute all contribs.
                contribs = compute_contributions(family)
                current_score += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        if current_score > best_score:
            best_score = current_score
            best_family = family[:]
    
    # Convert masks to lists of integers
    if best_family is None:
        # Fallback: just return the first 40 masks
        best_family = list(range(m))
    
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
