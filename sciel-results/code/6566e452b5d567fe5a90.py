"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _check_validity(nums):
    """
    Check divisibility-incomparable subset sum constraint using numpy bitset.
    Returns True if valid, False otherwise.
    """
    n = len(nums)
    if n == 0:
        return True
    
    max_sum = sum(nums)
    if max_sum > 32768:  # Too large for efficient bitset
        return False
    
    # Generate all subset sums using bitset approach
    # sums_mask[i] = 1 if sum i is achievable
    sums_mask = np.zeros(max_sum + 1, dtype=np.uint8)
    sums_mask[0] = 1
    
    for x in nums:
        if x > max_sum:
            continue
        # Shift and OR
        shifted = np.zeros(max_sum + 1, dtype=np.uint8)
        shifted[x:] = sums_mask[:max_sum - x + 1]
        sums_mask = sums_mask | shifted
    
    # Get all achievable sums
    achievable = np.where(sums_mask > 0)[0]
    if len(achievable) != (1 << n) - 1:
        return False  # Duplicate subset sums
    
    # Check divisibility: for each achievable sum s, no k*s (k>=2) should be achievable
    sums_set = set(achievable.tolist())
    for s in achievable:
        if s == 0:
            continue
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = [31, 32, 34]
    best_size = len(best)
    
    # Binary representation approach:
    # Assign each element a unique "signature" power of 2
    # Element i gets signature 2^i, so subset sum for mask m is sum of 2^i for i in mask
    # This ensures all subset sums are distinct
    # To maximize set size, use smallest elements that still avoid divisibility
    
    # Strategy: For a set of size n, element i (0-indexed) should be >= 2^i
    # to ensure unique subset sums
    # But we also need to avoid divisibility between subset sums
    
    # Let's try constructing sets of increasing size using the binary signature approach
    # Element i has base value 2^i, plus a small offset to avoid divisibility
    
    for target_size in range(4, 13):
        if time.time() > deadline - 5:
            break
        
        # Try multiple offset patterns
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(trial * 104729 + target_size * 7919)
            
            # Build candidate set with binary signatures
            # Element i: base = 2^i, offset = random small value
            candidates = []
            for i in range(target_size):
                base = 1 << i
                # Add offset to break divisibility patterns
                offset = rng.randint(1, min(10, 256 - base)) if base < 246 else 0
                val = base + offset
                if val > 256:
                    val = base
                candidates.append(val)
            
            # Ensure all distinct and in range
            if len(set(candidates)) != len(candidates):
                continue
            if any(v < 1 or v > 256 for v in candidates):
                continue
            
            # Check validity
            if _check_validity(candidates):
                if target_size > best_size:
                    best = candidates[:]
                    best_size = target_size
                    break  # Found valid set of this size, try larger
        
        # Also try greedy extension from best
        if time.time() > deadline - 5:
            break
        
        extended = best[:]
        for num in range(256, 0, -1):
            if num in extended:
                continue
            trial_set = extended + [num]
            if len(trial_set) <= best_size + 2:
                if _check_validity(trial_set):
                    extended = trial_set
                    best_size = len(extended)
        
        if len(extended) > best_size:
            best = extended[:]
            best_size = len(extended)
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 3:
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            new_best = best[:]
            improved = False
            for i in range(len(new_best)):
                for _ in range(20):
                    replacement = random.randint(1, 256)
                    if replacement in new_best:
                        continue
                    trial = new_best[:]
                    trial[i] = replacement
                    if _check_validity(trial):
                        new_best = trial
                        improved = True
                        break
                if improved:
                    break
            if len(new_best) > len(best):
                best = new_best[:]
                best_size = len(best)
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
