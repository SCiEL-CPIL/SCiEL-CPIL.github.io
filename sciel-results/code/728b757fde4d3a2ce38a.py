import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85.0
    
    # Precompute all masks and their list representations
    all_masks = list(range(1 << n))
    mask_to_list = {mask: [i for i in range(n) if (mask >> i) & 1] for mask in all_masks}
    
    # Precompute comparability: comp[i][j] = True if mask i and mask j are comparable
    # Two masks are comparable if one is a subset of the other
    # a is subset of b iff (a & b) == a
    # So masks a, b are comparable iff (a & b) == a or (a & b) == b
    # i.e., a | b == a or a | b == b
    # i.e., (a | b) == a or (a | b) == b
    
    # For efficiency, precompute a 4096x4096 boolean matrix or use bit tricks
    # Actually, let's precompute for each mask, which other masks it's comparable to
    # But 4096x4096 = 16M entries, might be too much memory/time
    # Let's just compute on the fly with bit operations - they're fast
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def calculate_score(masks):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score
    
    def score_delta(swap_idx, new_mask, masks):
        """Calculate change in score if we replace masks[swap_idx] with new_mask."""
        old_mask = masks[swap_idx]
        if old_mask == new_mask:
            return 0
        delta = 0
        for j in range(m):
            if j == swap_idx:
                continue
            other = masks[j]
            old_comp = is_comparable(old_mask, other)
            new_comp = is_comparable(new_mask, other)
            if old_comp and not new_comp:
                delta -= 1
            elif new_comp and not old_comp:
                delta += 1
        return delta
    
    # Initialize with a good starting point
    # Strategy: start with a chain-like structure
    # One approach: use all subsets of a fixed set of size k, plus some extras
    # Better: start with random and hill climb
    
    random.seed(42)
    
    # Start with a greedy construction
    # Build a family by starting with empty set and adding sets that maximize comparability
    masks = [0]  # Start with empty set
    used = {0}
    
    # Add sets greedily
    for _ in range(m - 1):
        best_mask = None
        best_score = -1
        # Try all unused masks
        for candidate in all_masks:
            if candidate in used:
                continue
            # Score: number of existing masks comparable to candidate
            s = 0
            for existing in masks:
                if is_comparable(candidate, existing):
                    s += 1
            if s > best_score:
                best_score = s
                best_mask = candidate
        if best_mask is None:
            break
        masks.append(best_mask)
        used.add(best_mask)
    
    # Pad if needed
    while len(masks) < m:
        for candidate in all_masks:
            if candidate not in used:
                masks.append(candidate)
                used.add(candidate)
                break
    
    current_score = calculate_score(masks)
    best_masks = masks[:]
    best_score = current_score
    
    # Hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for swap_idx in range(m):
            if time.time() > deadline:
                break
            best_delta = 0
            best_new_mask = None
            # Try all possible replacements
            for new_mask in all_masks:
                if new_mask == masks[swap_idx]:
                    continue
                d = score_delta(swap_idx, new_mask, masks)
                if d > best_delta:
                    best_delta = d
                    best_new_mask = new_mask
            if best_delta > 0 and best_new_mask is not None:
                masks[swap_idx] = best_new_mask
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_masks = masks[:]
    
    # Try some random restarts if time permits
    if time.time() < deadline - 5.0:
        for restart in range(10):
            if time.time() > deadline - 2.0:
                break
            random.seed(restart + 100)
            # Random initialization
            rand_masks = random.sample(all_masks, m)
            rand_score = calculate_score(rand_masks)
            if rand_score > best_score:
                best_masks = rand_masks[:]
                best_score = rand_score
                current_score = rand_score
                masks = rand_masks[:]
                # Hill climb from this start
                improved2 = True
                while improved2 and time.time() < deadline - 1.0:
                    improved2 = False
                    for swap_idx in range(m):
                        if time.time() > deadline - 1.0:
                            break
                        best_delta = 0
                        best_new_mask = None
                        for new_mask in all_masks:
                            if new_mask == masks[swap_idx]:
                                continue
                            d = score_delta(swap_idx, new_mask, masks)
                            if d > best_delta:
                                best_delta = d
                                best_new_mask = new_mask
                        if best_delta > 0 and best_new_mask is not None:
                            masks[swap_idx] = best_new_mask
                            current_score += best_delta
                            improved2 = True
                            if current_score > best_score:
                                best_score = current_score
                                best_masks = masks[:]
    
    # Return as list of lists of ints
    result = []
    for mask in best_masks:
        result.append([i for i in range(n) if (mask >> i) & 1])
    
    # Ensure types are correct
    final_result = []
    for lst in result:
        final_result.append([int(x) for x in lst])
    
    return final_result
