"""Program using a MODEL-AND-OPTIMIZE approach to maximize 3-term APs without 4-term APs in [1,256]."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _has_4ap_mask(mask, n=256):
    """Check if bitmask contains a 4-term AP using bit operations."""
    # For each difference d, check if there exists x such that x, x+d, x+2d, x+3d all in set
    # Represent set as bitmask where bit i means (i+1) is in set (0-indexed)
    # A 4-AP with difference d: positions p, p+d, p+2d, p+3d
    # For a given d, we can check: mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
    # If any bit is set in the result, there's a 4-AP
    for d in range(1, (n - 1) // 3 + 1):
        if mask & (mask >> d) & (mask >> (2 * d)) & (mask >> (3 * d)):
            return True
    return False


def _count_3aps_mask(mask, n=256):
    """Count 3-term APs in a bitmask using bit operations.
    A 3-AP with middle at position j and difference d: positions j-d, j, j+d
    For each d, count bits set in mask & (mask >> d) & (mask << d) shifted appropriately.
    Actually: for middle element at position i (0-indexed, value i+1), 
    we need (i-d) and (i+d) to both be set.
    So for each d: count = popcount(mask & (mask >> d) & (mask << d)) but we need to be careful.
    
    Better approach: for each d, the number of 3-APs with difference d is
    popcount(mask & (mask >> d) & (mask << d)) -- no wait.
    
    If we have elements at positions p, p+d, p+2d (0-indexed), then:
    - bit p is set in mask
    - bit p+d is set in mask  
    - bit p+2d is set in mask
    
    For a fixed d, a 3-AP starting at position p means bits p, p+d, p+2d are set.
    This is equivalent to: for position p, mask has bit p, bit p+d, bit p+2d.
    
    Alternative: for each d, consider the sequence. A 3-AP with difference d has middle at position m
    where m-d and m+d are also in the set. So count = popcount(mask & (mask >> d) & (mask << d))
    but we need to mask out bits that would go out of range.
    
    Actually simpler: for each d from 1 to (n-1)//2:
        shifted_right = mask >> d
        shifted_left = mask << d
        common = mask & shifted_right & shifted_left
        count += popcount(common)
    
    Wait, that's not right either. Let me think again.
    
    If element at position i is the middle of a 3-AP with difference d, then:
    - position i-d must be in set: bit (i-d) set in mask
    - position i must be in set: bit i set in mask  
    - position i+d must be in set: bit (i+d) set in mask
    
    So for each i, we check if bits (i-d), i, (i+d) are all set.
    For fixed d, this is: for each i from d to n-1-d, check mask[i-d] & mask[i] & mask[i+d].
    
    Using bit operations: 
    mask_shifted_left_by_d = mask << d  (bit i in this is bit i-d in original... no)
    
    Let me use a different approach. For fixed d:
    - Create mask_a where bit i is set if bit (i+d) is set in original: mask_a = mask >> d
    - Create mask_b where bit i is set if bit (i-d) is set in original: mask_b = mask << d
    - Then bits set in mask & mask_a & mask_b correspond to middle elements of 3-APs with difference d.
    
    So count_d = popcount(mask & (mask >> d) & (mask << d))
    But we need to ensure i-d >= 0 and i+d < n, which is handled by the shifts naturally (bits outside range become 0).
    
    Total 3-APs = sum over d of popcount(mask & (mask >> d) & (mask << d))
    """
    total = 0
    for d in range(1, n // 2 + 1):
        common = mask & (mask >> d) & (mask << d)
        total += bin(common).count('1')
    return total


def _delta_3aps_add(mask, pos, n=256):
    """Compute how many 3-APs are added when adding position pos (0-indexed) to mask."""
    added = 0
    # pos can be the first, middle, or last element of a 3-AP
    # As middle: need pos-d and pos+d in set
    for d in range(1, n // 2 + 1):
        left = pos - d
        right = pos + d
        if left >= 0 and right < n:
            if (mask >> left) & 1 and (mask >> right) & 1:
                added += 1
        # As first: need pos+d and pos+2d in set
        if pos + 2 * d < n:
            if (mask >> (pos + d)) & 1 and (mask >> (pos + 2 * d)) & 1:
                added += 1
        # As last: need pos-2d and pos-d in set
        if pos - 2 * d >= 0:
            if (mask >> (pos - 2 * d)) & 1 and (mask >> (pos - d)) & 1:
                added += 1
    return added


def _delta_3aps_remove(mask, pos, n=256):
    """Compute how many 3-APs are lost when removing position pos (0-indexed) from mask."""
    removed = 0
    for d in range(1, n // 2 + 1):
        left = pos - d
        right = pos + d
        if left >= 0 and right < n:
            if (mask >> left) & 1 and (mask >> right) & 1:
                removed += 1
        if pos + 2 * d < n:
            if (mask >> (pos + d)) & 1 and (mask >> (pos + 2 * d)) & 1:
                removed += 1
        if pos - 2 * d >= 0:
            if (mask >> (pos - 2 * d)) & 1 and (mask >> (pos - d)) & 1:
                removed += 1
    return removed


def _creates_4ap_on_add(mask, pos, n=256):
    """Check if adding position pos creates a 4-AP."""
    new_mask = mask | (1 << pos)
    # Only need to check 4-APs that include position pos
    # pos can be at position 0, 1, 2, or 3 in the 4-AP
    for d in range(1, (n - 1) // 3 + 1):
        # pos as first (index 0): need pos+d, pos+2d, pos+3d
        if pos + 3 * d < n:
            if ((mask >> (pos + d)) & 1) and ((mask >> (pos + 2 * d)) & 1) and ((mask >> (pos + 3 * d)) & 1):
                return True
        # pos as second (index 1): need pos-d, pos+d, pos+2d
        if pos - d >= 0 and pos + 2 * d < n:
            if ((mask >> (pos - d)) & 1) and ((mask >> (pos + d)) & 1) and ((mask >> (pos + 2 * d)) & 1):
                return True
        # pos as third (index 2): need pos-2d, pos-d, pos+d
        if pos - 2 * d >= 0 and pos + d < n:
            if ((mask >> (pos - 2 * d)) & 1) and ((mask >> (pos - d)) & 1) and ((mask >> (pos + d)) & 1):
                return True
        # pos as fourth (index 3): need pos-3d, pos-2d, pos-d
        if pos - 3 * d >= 0:
            if ((mask >> (pos - 3 * d)) & 1) and ((mask >> (pos - 2 * d)) & 1) and ((mask >> (pos - d)) & 1):
                return True
    return False


def _greedy_construct_bitmask():
    """Greedy construction using bitmask operations."""
    mask = 0
    n = 256
    for x in range(1, n + 1):
        pos = x - 1
        if not _creates_4ap_on_add(mask, pos, n):
            mask |= (1 << pos)
    return mask


def _randomized_greedy_bitmask(seed_val=42, max_iter=30):
    """Try multiple random orderings with bitmask operations."""
    n = 256
    best_mask = _greedy_construct_bitmask()
    best_three = _count_3aps_mask(best_mask, n)
    
    rng = random.Random(seed_val)
    for it in range(max_iter):
        order = list(range(n))
        rng.shuffle(order)
        mask = 0
        for pos in order:
            if not _creates_4ap_on_add(mask, pos, n):
                mask |= (1 << pos)
        three = _count_3aps_mask(mask, n)
        if three > best_three:
            best_three = three
            best_mask = mask
    return best_mask, best_three


def _local_search_bitmask(start_mask, start_three, max_iters=5000, seed_val=0, deadline=None):
    """Local search: try adding, removing, or swapping elements to increase 3-AP count."""
    n = 256
    current_mask = start_mask
    current_three = start_three
    best_mask = current_mask
    best_three = current_three
    rng = random.Random(seed_val)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Randomly choose move type: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try adding an element
            pos = rng.randint(0, n - 1)
            if (current_mask >> pos) & 1:
                continue
            if _creates_4ap_on_add(current_mask, pos, n):
                continue
            delta = _delta_3aps_add(current_mask, pos, n)
            new_mask = current_mask | (1 << pos)
            new_three = current_three + delta
            if new_three > current_three:
                current_mask = new_mask
                current_three = new_three
                if current_three > best_three:
                    best_three = current_three
                    best_mask = current_mask
        elif move_type < 0.5:
            # Try removing an element
            # Pick a random set bit
            bits = [i for i in range(n) if (current_mask >> i) & 1]
            if not bits:
                continue
            pos = rng.choice(bits)
            delta = _delta_3aps_remove(current_mask, pos, n)
            new_mask = current_mask & ~(1 << pos)
            new_three = current_three - delta
            if new_three > current_three:
                current_mask = new_mask
                current_three = new_three
                if current_three > best_three:
                    best_three = current_three
                    best_mask = current_mask
        else:
            # Try swapping: remove one, add another
            bits = [i for i in range(n) if (current_mask >> i) & 1]
            if not bits:
                continue
            out_pos = rng.choice(bits)
            in_pos = rng.randint(0, n - 1)
            if (current_mask >> in_pos) & 1:
                continue
            
            # Check if removing out_pos and adding in_pos is 4-AP free
            temp_mask = current_mask & ~(1 << out_pos)
            if _creates_4ap_on_add(temp_mask, in_pos, n):
                continue
            
            delta_remove = _delta_3aps_remove(current_mask, out_pos)
            delta_add = _delta_3aps_add(temp_mask, in_pos, n)
            net_delta = delta_add - delta_remove
            if net_delta > 0:
                new_mask = temp_mask | (1 << in_pos)
                new_three = current_three + net_delta
                current_mask = new_mask
                current_three = new_three
                if current_three > best_three:
                    best_three = current_three
                    best_mask = current_mask
    
    return best_mask, best_three


def _mask_to_list(mask):
    """Convert bitmask to sorted list of 1-indexed values."""
    result = []
    for i in range(256):
        if (mask >> i) & 1:
            result.append(i + 1)
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 256
    
    # Start with greedy
    best_mask = _greedy_construct_bitmask()
    best_three = _count_3aps_mask(best_mask, n)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return _mask_to_list(best_mask)
    
    # Try randomized greedy with multiple seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222]
    for seed_val in seeds:
        if time.time() - start_time > deadline - 30:
            break
        candidate_mask, three = _randomized_greedy_bitmask(seed_val=seed_val, max_iter=20)
        if three > best_three:
            best_three = three
            best_mask = candidate_mask
    
    # Local search from best solution
    if time.time() - start_time < deadline - 5:
        remaining = deadline - 5 - (time.time() - start_time)
        # Estimate: each iteration is fast with bitmask ops
        max_ls_iters = min(10000, int(remaining * 500))
        if max_ls_iters > 100:
            improved_mask, improved_three = _local_search_bitmask(
                best_mask, best_three, max_iters=max_ls_iters, seed_val=0, deadline=deadline - 5
            )
            if improved_three > best_three:
                best_three = improved_three
                best_mask = improved_mask
    
    # Second round of local search with different seed if time allows
    if time.time() - start_time < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        max_ls_iters2 = min(5000, int(remaining * 300))
        if max_ls_iters2 > 100:
            improved_mask2, improved_three2 = _local_search_bitmask(
                best_mask, best_three, max_iters=max_ls_iters2, seed_val=12345, deadline=deadline - 10
            )
            if improved_three2 > best_three:
                best_three = improved_three2
                best_mask = improved_mask2
    
    # Final verification
    if _has_4ap_mask(best_mask, n):
        # Fallback to pure greedy which is guaranteed no 4AP
        best_mask = _greedy_construct_bitmask()
    
    return _mask_to_list(best_mask)

# EVOLVE-BLOCK-END
