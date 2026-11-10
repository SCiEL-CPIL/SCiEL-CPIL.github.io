import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_sidon_collisions(subset):
    """Count the number of pair-sum collisions in a subset.
    For a Sidon set, all unordered pair sums (including diagonals) must be unique.
    We count how many sums are shared by multiple pairs.
    """
    sums = {}
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Count total number of "excess" pairs (collisions)
    collisions = 0
    for s, count in sums.items():
        if count > 1:
            collisions += count - 1
    return collisions

def _is_sidon(subset):
    """Check if a subset has all unique unordered pair sums (including diagonals)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(ambient):
    """Find the maximum size of a Sidon subset of ambient.
    Uses exhaustive search with pruning.
    """
    n = len(ambient)
    best = 0
    
    # Try all subsets of decreasing size until we find one that is Sidon
    # Start from largest possible
    for size in range(n, best, -1):
        found = False
        for subset in combinations(range(n), size):
            sub = [ambient[i] for i in subset]
            if _is_sidon(sub):
                best = size
                found = True
                break
        if found:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Greedy collision-density search:
    # Build the ambient set one element at a time.
    # At each step, choose the next integer from [0, 128] that maximizes
    # the number of pair-sum collisions with the already-chosen elements.
    
    # Start with a baseline: arithmetic progression {0, 1, ..., 11}
    # This gives max Sidon subset size of 4.
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    
    # Try greedy construction from different starting points
    best_candidate = baseline
    best_score = baseline_score
    
    # Strategy: Greedily build set by adding elements that create most collisions
    # We'll try multiple greedy runs with different initial seeds
    
    for seed_start in range(0, 129, 8):
        if time.time() > deadline:
            break
            
        # Start with a small seed set
        chosen = [seed_start]
        used = set(chosen)
        
        for step in range(1, 12):
            if time.time() > deadline:
                break
                
            best_next = None
            best_collision_count = -1
            
            # Try each candidate in [0, 128]
            for cand in range(0, 129):
                if cand in used:
                    continue
                    
                # Calculate how many new pair-sum collisions this creates
                # with the existing chosen elements
                new_collisions = 0
                existing_sums = set()
                for i in range(len(chosen)):
                    for j in range(i, len(chosen)):
                        existing_sums.add(chosen[i] + chosen[j])
                
                # Count how many sums involving cand collide with existing sums
                # or with each other
                new_sums = {}
                for i in range(len(chosen)):
                    s = cand + chosen[i]
                    if s in existing_sums:
                        new_collisions += 1
                    if s in new_sums:
                        new_collisions += 1
                    else:
                        new_sums[s] = True
                # Diagonal: cand + cand
                s_diag = 2 * cand
                if s_diag in existing_sums:
                    new_collisions += 1
                elif s_diag in new_sums:
                    new_collisions += 1
                    
                if new_collisions > best_collision_count:
                    best_collision_count = new_collisions
                    best_next = cand
                    
            if best_next is not None:
                chosen.append(best_next)
                used.add(best_next)
            else:
                # No valid candidate found, break
                break
        
        if len(chosen) == 12:
            score = _max_sidon_subset_size(chosen)
            if score < best_score:
                best_score = score
                best_candidate = chosen[:]
                
        if best_score <= 3:
            # Can't do better than 3 (minimum possible for 12 elements)
            break
    
    # Verify final candidate
    if len(best_candidate) != 12:
        return baseline
    if len(set(best_candidate)) != 12:
        return baseline
    if any(x < 0 or x > 128 for x in best_candidate):
        return baseline
        
    return best_candidate
# EVOLVE-BLOCK-END
