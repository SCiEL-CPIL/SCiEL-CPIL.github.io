"""Improved program: Explicit Cantor set (base-3 0/1 digits) construction for 3-AP free set, followed by repair for full constraint."""
import random

def _element_is_mean(a, S):
    """Return True if a is the exact arithmetic mean of some subset of S\\{a} of size >= 2.
    Uses a per-count bitset DP (Python big ints) over subset sums."""
    others = [x for x in S if x != a]
    n = len(others)
    if n < 2:
        return False
    max_sum = sum(others)
    mask = (1 << (max_sum + 1)) - 1
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0 set: empty subset sum 0
    for e in others:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                dp[c + 1] |= (dp[c] << e) & mask
    for c in range(2, n + 1):
        target = c * a
        if target <= max_sum and ((dp[c] >> target) & 1):
            return True
    return False

def _is_fully_valid(S):
    for a in S:
        if _element_is_mean(a, S):
            return False
    return True

def _repair(S):
    """Remove any element that is the mean of a subset of the others, until valid."""
    S = set(S)
    guard = len(S) + 2
    while guard > 0:
        guard -= 1
        removed = False
        for a in list(S):
            if _element_is_mean(a, S):
                S.remove(a)
                removed = True
                break
        if not removed:
            break
    return S

def _generate_cantor_set(n_max):
    """Generate all numbers <= n_max with base-3 digits in {0, 1}."""
    nums = []
    # 3^0 to 3^10 covers up to 59049, which is > 512.
    # Max power: 3^6 = 729. So digits 0..5.
    max_power = 0
    while (3 ** max_power) <= n_max:
        max_power += 1
    
    # Iterate through all combinations of 0/1 for digits 0..max_power-1
    for mask in range(1 << max_power):
        val = 0
        for i in range(max_power):
            if mask & (1 << i):
                val += 3 ** i
        if val <= n_max and val > 0:
            nums.append(val)
    return sorted(nums)

def construct():
    n_max = 512
    target_size = 48
    
    # 1. Generate the Cantor set (base-3 0/1 digits)
    # This set is 3-AP free.
    cantor_nums = _generate_cantor_set(n_max)
    
    # The Cantor set for n=512 has 64 elements (2^6, since 3^6=729>512, max digit index 5).
    # All are <= 364.
    # We need 48.
    
    # Strategy: Try to pick 48 from the Cantor set that are valid for the full constraint.
    # Since the Cantor set is 3-AP free, violations for k>=3 are rare.
    # We can just take the first 48 and repair if necessary.
    # Or, to be safe, we can iterate through candidates and build a valid set.
    
    # Let's start with the smallest 48, as they are "denser" at the low end?
    # Actually, larger numbers have fewer "partners" in the set to form means?
    # Let's just take a subset of size 48 and repair.
    
    # To maximize success, we might want to pick a subset that is "spread out".
    # But the Cantor set is already well-spread.
    
    # Let's try the first 48.
    candidate_set = set(cantor_nums[:target_size])
    
    # Verify and repair
    valid_set = _repair(candidate_set)
    
    # If repair reduced size significantly, try another subset?
    # With 64 candidates and needing 48, even if we lose 10, we are at 38.
    # We can try to add back elements from the remaining 16.
    
    if len(valid_set) < target_size:
        remaining = [x for x in cantor_nums[target_size:] if x not in valid_set]
        # Sort remaining by some heuristic? Maybe just add them one by one if valid.
        for x in remaining:
            if len(valid_set) >= target_size:
                break
            new_set = valid_set | {x}
            if _is_fully_valid(new_set):
                valid_set = new_set
    
    # Fallback: If we still don't have 48, try random subsets of the Cantor set
    if len(valid_set) < target_size:
        rng = random.Random(42)
        for _ in range(100):
            # Pick 50 random from Cantor set
            subset = set(rng.sample(cantor_nums, min(50, len(cantor_nums))))
            repaired = _repair(subset)
            if len(repaired) > len(valid_set):
                valid_set = repaired
            if len(valid_set) >= target_size:
                break

    # Final check
    if not _is_fully_valid(valid_set):
        valid_set = _repair(valid_set)
        
    # If still not 48, fallback to powers of 2 (known valid, size 9)
    # But we expect 48.
    if len(valid_set) < target_size:
        # Last resort: just take the first 48 of the Cantor set and hope repair works?
        # Or return what we have.
        pass

    result = sorted(x for x in valid_set if 1 <= x <= n_max)[:target_size]
    return result
