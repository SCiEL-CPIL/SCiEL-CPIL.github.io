import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask >> 1] + (mask & 1)
    
    # Precompute subset relationship: for each mask, list of masks that are strict supersets
    # and list of masks that are strict subsets
    # Actually, let's precompute a comparison matrix or use bit operations on the fly
    
    def is_comparable(a, b):
        """Check if a is strict subset of b or b is strict subset of a."""
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def count_comparable(family_masks):
        """Count comparable pairs in a family of masks."""
        count = 0
        k = len(family_masks)
        for i in range(k):
            for j in range(i + 1, k):
                if is_comparable(family_masks[i], family_masks[j]):
                    count += 1
        return count
    
    def greedy_construct():
        """Greedy construction: iteratively add the mask that maximizes comparable pairs."""
        family = []
        remaining = list(range(total_masks))
        
        # Start with empty set (mask 0) and full set (mask total_masks-1)
        # Actually, let's be smarter: start with a chain-like structure
        # Use greedy: at each step, pick the mask that adds the most comparable pairs
        
        # Precompute for each mask, which other masks it's comparable with
        # This is expensive: 4096 * 4096 = ~16M operations, which is fine
        
        # Actually, let's just do greedy without precomputation for simplicity
        # but with bit operations
        
        for _ in range(m):
            if not remaining:
                break
            best_mask = -1
            best_gain = -1
            for mask in remaining:
                # Count how many existing family members are comparable with this mask
                gain = 0
                for fam_mask in family:
                    if is_comparable(mask, fam_mask):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                break
            family.append(best_mask)
            remaining.remove(best_mask)
        
        return family
    
    def local_search(family, max_iterations=10000):
        """Local search: try swapping one mask in the family with one not in the family."""
        family = list(family)
        current_score = count_comparable(family)
        family_set = set(family)
        candidates = [mask for mask in range(total_masks) if mask not in family_set]
        
        improved = True
        while improved:
            improved = False
            # Shuffle candidates for diversity
            random.shuffle(candidates)
            for i in range(len(family)):
                if time.time() > deadline - 2.0:
                    return family
                for j in range(len(candidates)):
                    if time.time() > deadline - 2.0:
                        return family
                    new_family = family.copy()
                    new_family[i] = candidates[j]
                    new_score = count_comparable(new_family)
                    if new_score > current_score:
                        current_score = new_score
                        family = new_family
                        family_set = set(family)
                        improved = True
                        break
                if improved:
                    break
        return family
    
    def swap_local_search(family, max_iterations=5000):
        """More targeted local search: try swapping specific pairs."""
        family = list(family)
        current_score = count_comparable(family)
        family_set = set(family)
        
        # Precompute contribution of each element to the score
        # For each element i, score_i = number of other elements comparable with i
        def compute_contributions(fam):
            contribs = [0] * len(fam)
            for i in range(len(fam)):
                for j in range(len(fam)):
                    if i != j and is_comparable(fam[i], fam[j]):
                        contribs[i] += 1
            return contribs
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            contribs = compute_contributions(family)
            # Try removing the element with lowest contribution and adding a candidate
            min_idx = min(range(len(family)), key=lambda i: contribs[i])
            removed = family[min_idx]
            candidates = [mask for mask in range(total_masks) if mask not in family_set]
            random.shuffle(candidates)
            best_candidate = None
            best_gain = 0
            for cand in candidates:
                if time.time() > deadline - 2.0:
                    break
                # Compute gain: comparable pairs gained with cand minus lost with removed
                gain = 0
                for i, fam_mask in enumerate(family):
                    if i == min_idx:
                        continue
                    if is_comparable(cand, fam_mask):
                        gain += 1
                    if is_comparable(removed, fam_mask):
                        gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            if best_candidate is not None and best_gain > 0:
                family[min_idx] = best_candidate
                current_score += best_gain
                family_set = set(family)
                improved = True
        return family
    
    # Run greedy construction with multiple random restarts
    best_family = []
    best_score = -1
    
    # Try a few different greedy strategies
    # Strategy 1: Standard greedy
    family = greedy_construct()
    score = count_comparable(family)
    if score > best_score:
        best_score = score
        best_family = family[:]
    
    # Strategy 2: Greedy starting from chains
    # Build a long chain first, then fill in
    chain = []
    # Create a chain: 0, 1, 3, 7, 15, ... (adding one bit at a time)
    mask = 0
    for b in range(n):
        mask |= (1 << b)
        chain.append(mask)
    # Now extend with other masks greedily
    remaining = [mask for mask in range(total_masks) if mask not in chain]
    family2 = chain[:]
    for _ in range(m - len(chain)):
        if not remaining:
            break
        best_mask = -1
        best_gain = -1
        for mask in remaining:
            gain = 0
            for fam_mask in family2:
                if is_comparable(mask, fam_mask):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask == -1:
            break
        family2.append(best_mask)
        remaining.remove(best_mask)
    score2 = count_comparable(family2)
    if score2 > best_score:
        best_score = score2
        best_family = family2[:]
    
    # Local search on best
    if time.time() < deadline - 10.0:
        improved_family = local_search(best_family)
        improved_score = count_comparable(improved_family)
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_family
    
    # Swap-based local search
    if time.time() < deadline - 5.0:
        improved_family = swap_local_search(best_family)
        improved_score = count_comparable(improved_family)
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_family
    
    # Final check: ensure we have exactly m distinct masks
    if len(best_family) != m:
        # Pad if necessary
        existing = set(best_family)
        for mask in range(total_masks):
            if len(best_family) >= m:
                break
            if mask not in existing:
                best_family.append(mask)
                existing.add(mask)
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_family[:m]:
        vertices = [b for b in range(n) if (mask >> b) & 1]
        result.append(vertices)
    
    # Ensure all types are correct
    for i in range(len(result)):
        result[i] = list(result[i])
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    
    return result
