"""Bitset-based search for divisibility-incomparable subset sums."""
import time
import random

# EVOLVE-BLOCK-START

def _precompute():
    """Precompute divisibility masks for O(1) bitwise checks."""
    # Max possible subset sum: 30 elements from 1..256
    # Max sum of 30 distinct from 1..256: 256+255+...+227 = 30*(256+227)/2 = 7245
    MAX_SUM = 7500
    
    # divisibility_mask[s] has bits set at positions 2s, 3s, 4s, ... up to MAX_SUM
    divisibility_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        m = 2 * s
        while m <= MAX_SUM:
            mask |= (1 << m)
            m += s
        divisibility_mask[s] = mask
    
    return MAX_SUM, divisibility_mask

def _check_divisibility(sums_mask, divisibility_mask):
    """Check if no two distinct sums in sums_mask divide each other."""
    mask = sums_mask
    while mask:
        low_bit = mask & (-mask)
        s = low_bit.bit_length() - 1
        if sums_mask & divisibility_mask[s]:
            return False
        mask ^= low_bit
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MAX_SUM, divisibility_mask = _precompute()
    
    def compute_sums_mask(elements):
        """Compute the bitmask of all nonempty subset sums.
        Returns (mask, is_valid_unique) where is_valid_unique is False if any duplicate sums exist."""
        n = len(elements)
        if n == 0:
            return 0, True
        mask = 1  # bit 0 set, representing empty sum
        for x in elements:
            shifted = mask << x
            if mask & shifted:
                return 0, False  # collision detected
            mask |= shifted
        mask &= ~1  # remove empty sum
        return mask, True
    
    def try_add(current_mask, x):
        """Try to add x to the current set of subset sums.
        Returns new_mask if valid, None otherwise."""
        if current_mask & (1 << x):
            return None  # x is already a subset sum
        
        shifted = current_mask << x
        
        # Check collision between shifted and current_mask
        if shifted & current_mask:
            return None
        
        new_mask = current_mask | shifted | (1 << x)
        
        # Check divisibility
        if not _check_divisibility(new_mask, divisibility_mask):
            return None
        
        return new_mask
    
    def greedy_construct(order):
        """Build a valid set by adding elements in the given order."""
        current = []
        current_mask = 0
        
        for x in order:
            result = try_add(current_mask, x)
            if result is not None:
                current.append(x)
                current_mask = result
                if len(current) >= 30:
                    break
        
        return current, current_mask
    
    # Phase 1: Greedy construction with multiple orderings
    candidates_all = list(range(1, 257))
    rng = random.Random(42)
    
    # Precompute divisor counts for ordering
    div_count = [0] * 257
    for i in range(1, 257):
        for j in range(1, i + 1):
            if i % j == 0:
                div_count[i] += 1
    
    orderings = []
    # Descending
    orderings.append(sorted(candidates_all, reverse=True))
    # Ascending
    orderings.append(sorted(candidates_all))
    # By divisor count (fewer first)
    orderings.append(sorted(candidates_all, key=lambda x: (div_count[x], x)))
    # By divisor count (more first)
    orderings.append(sorted(candidates_all, key=lambda x: (-div_count[x], x)))
    # Random orderings
    for _ in range(100):
        shuffled = candidates_all[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_set = []
    best_mask = 0
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        elements, mask = greedy_construct(order)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_mask = mask
    
    # Phase 2: Local search - try to add more elements
    if time.time() < deadline and best_size < 30:
        remaining = [x for x in candidates_all if x not in best_set]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            result = try_add(best_mask, x)
            if result is not None:
                best_set.append(x)
                best_mask = result
                best_size += 1
    
    # Phase 3: Swap-based local search to find better configurations
    if time.time() < deadline and best_size < 30:
        current_set = list(best_set)
        current_mask = best_mask
        current_size = best_size
        
        # Try swaps to free up room for larger elements
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Try swapping a small element with a larger one
            if current_size == 0:
                break
            
            i = rng.randrange(current_size)
            old_x = current_set[i]
            
            # Find a candidate to swap with
            candidates = [x for x in candidates_all if x not in current_set and x > old_x]
            if not candidates:
                continue
            
            new_x = rng.choice(candidates)
            
            # Remove old_x
            remaining = current_set[:i] + current_set[i+1:]
            remaining_mask, valid = compute_sums_mask(remaining)
            if not valid:
                continue
            
            # Add new_x
            result = try_add(remaining_mask, new_x)
            if result is None:
                continue
            
            new_set = remaining + [new_x]
            new_mask = result
            new_size = current_size
            
            # After swap, try to add more elements
            improved = False
            temp_mask = new_mask
            temp_set = list(new_set)
            
            for y in candidates_all:
                if y in temp_set:
                    continue
                res = try_add(temp_mask, y)
                if res is not None:
                    temp_set.append(y)
                    temp_mask = res
                    improved = True
            
            if improved and len(temp_set) > current_size:
                current_set = temp_set
                current_mask = temp_mask
                current_size = len(temp_set)
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    best_mask = current_mask
    
    # Final verification
    if best_set:
        final_mask, valid = compute_sums_mask(best_set)
        if not valid or not _check_divisibility(final_mask, divisibility_mask):
            # Fallback to a known valid small set
            best_set = [31, 32, 34]
    
    return list(best_set)

# EVOLVE-BLOCK-END
