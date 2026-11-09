"""Optimized construction of a 3-AP-free subset of [1,256] maximizing sum of reciprocals."""
import time
import random
from fractions import Fraction

# Deadline guard
_DEADLINE = 90.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _is_3ap_free_bitset(mask):
    """Check if bitmask represents a 3-AP-free set."""
    # Iterate through set bits
    # To be efficient, we can just check pairs? 
    # Or use the property: for all a, b in S, 2b-a not in S.
    # This is O(N^2) worst case if we iterate all pairs.
    # But N is small (256). 
    # Better: iterate a, b.
    # Actually, the add-check is O(N). Full check is O(N^2).
    # We use this for validation only.
    s = set()
    for i in range(1, 257):
        if (mask >> i) & 1:
            s.add(i)
    for a in s:
        for b in s:
            if a < b:
                c = 2 * b - a
                if c != a and c != b and c in s:
                    return False
    return True

def _can_add_bitset(mask, x):
    """Check if adding x to set represented by mask keeps it 3-AP-free."""
    # x is 1-indexed. Bit index is x.
    # Check if x is already in
    if (mask >> x) & 1:
        return False
        
    # x as right term: a, b, x. a = 2b - x.
    # x as left term: x, b, c. c = 2b - x. (Same condition: if 2b-x in S, then AP exists)
    # x as middle term: a, x, c. a = 2x - c, c = 2x - a. (Condition: if 2x-b in S)
    
    # Iterate through all b in S
    # We need to check if there exists b in S such that:
    # 1. 2*b - x is in S (and 2*b - x != x)
    # 2. 2*x - b is in S (and 2*x - b != x)
    
    # To iterate through set bits in mask:
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        b = lsb.bit_length() - 1
        m ^= lsb
        
        # Check 1: b is middle, x is right (or left)
        # AP: (2b-x, b, x) or (x, b, 2b-x)
        # If 2b-x is in S, we have an AP.
        # Note: 2b-x must be >= 1 and != x.
        val1 = 2 * b - x
        if val1 != x and val1 >= 1 and val1 <= 256:
            if (mask >> val1) & 1:
                return False
        
        # Check 2: x is middle, b is left (or right)
        # AP: (b, x, 2x-b) or (2x-b, x, b)
        # If 2x-b is in S, we have an AP.
        val2 = 2 * x - b
        if val2 != x and val2 >= 1 and val2 <= 256:
            if (mask >> val2) & 1:
                return False
                
    return True

def _remove_bit(mask, x):
    return mask & ~(1 << x)

def _add_bit(mask, x):
    return mask | (1 << x)

def _sum_recip_float(mask):
    """Calculate sum of reciprocals as float."""
    total = 0.0
    m = mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        total += 1.0 / x
        m ^= lsb
    return total

def _greedy_construct_bitset():
    """Greedy: add elements in order of largest reciprocal (smallest x) that don't create 3-AP."""
    mask = 0
    for x in range(1, 257):
        if _can_add_bitset(mask, x):
            mask = _add_bit(mask, x)
    return mask

def _local_search_bitset(mask, max_iter=50000):
    """Local search with float sums."""
    current_sum = _sum_recip_float(mask)
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        
        improved = False
        
        # Try adding
        # We want to add small x to increase sum.
        # Iterate x from 1 to 256
        for x in range(1, 257):
            if not ((mask >> x) & 1):
                if _can_add_bitset(mask, x):
                    new_mask = _add_bit(mask, x)
                    # Sum increase is 1/x
                    new_sum = current_sum + 1.0 / x
                    if new_sum > current_sum: # Always true if x>0
                        mask = new_mask
                        current_sum = new_sum
                        improved = True
                        break
        if improved:
            continue
            
        # Try swapping: remove y, add x
        # To improve sum, we want 1/x > 1/y, so x < y.
        # Iterate y (to remove) and x (to add) such that x < y.
        # Or just iterate all x not in, y in.
        # Optimization: Iterate x (small) first, then y (large).
        found_swap = False
        for x in range(1, 257):
            if (mask >> x) & 1:
                continue
            # x is candidate to add
            # We need to remove some y such that 1/x > 1/y => x < y
            # And the swap is valid.
            # We don't know y yet. 
            # Strategy: For each x, try to find a y to remove.
            # To be fast, we can just check if adding x is possible *after* removing some y?
            # That's complex. 
            # Simpler: Iterate y in S. Remove y. Check if we can add x.
            # To prioritize gain, iterate x (small) and y (large).
            pass
            
        # Alternative: 
        # Generate list of (x, y) pairs where x not in, y in, x < y.
        # Sort by (1/x - 1/y) descending? i.e., x small, y large.
        # This is O(N^2) generation. N=256. 65k pairs. Manageable.
        
        candidates = []
        in_set = []
        not_in_set = []
        for i in range(1, 257):
            if (mask >> i) & 1:
                in_set.append(i)
            else:
                not_in_set.append(i)
                
        # We want to maximize 1/x - 1/y.
        # This is best when x is small and y is large.
        # Just iterate x in not_in_set (sorted asc) and y in in_set (sorted desc).
        # And check validity.
        
        best_gain = 0
        best_x = -1
        best_y = -1
        
        # Limit search to top few x and y to keep it fast?
        # Or just full scan if fast enough.
        # _can_add_bitset is O(N). 
        # Full scan is 256*256*256 = 16M ops. Too slow in Python.
        
        # Heuristic: Only try swapping x if x is small (e.g., < 50)
        # And y is large (e.g., > 50).
        
        for x in not_in_set:
            if x > 100: break # Don't add large x, gain is small
            # Try removing y from in_set
            # Sort in_set descending to find large y first?
            # We just need *any* improvement.
            for y in in_set:
                if y <= x: continue # Must remove larger to gain
                # Check if mask without y allows x
                temp_mask = _remove_bit(mask, y)
                if _can_add_bitset(temp_mask, x):
                    gain = 1.0/x - 1.0/y
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
                        
        if best_x != -1:
            mask = _remove_bit(mask, best_y)
            mask = _add_bit(mask, best_x)
            current_sum += best_gain
            improved = True

        if not improved:
            break
            
    return mask, current_sum

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with greedy
    mask = _greedy_construct_bitset()
    
    # Local search
    mask, _ = _local_search_bitset(mask, max_iter=10000)
    
    # Simulated Annealing (simplified)
    if _elapsed() < _DEADLINE - 10:
        import math
        random.seed(42)
        current_sum = _sum_recip_float(mask)
        best_mask = mask
        best_sum = current_sum
        T = 0.01
        
        for _ in range(50000):
            if _elapsed() > _DEADLINE - 5:
                break
            
            # Random move
            r = random.random()
            if r < 0.3:
                # Add
                candidates = [x for x in range(1, 257) if not ((mask >> x) & 1)]
                if candidates:
                    x = random.choice(candidates)
                    if _can_add_bitset(mask, x):
                        new_mask = _add_bit(mask, x)
                        new_sum = current_sum + 1.0/x
                        if new_sum > current_sum or random.random() < math.exp((new_sum - current_sum)/T):
                            mask = new_mask
                            current_sum = new_sum
                            if current_sum > best_sum:
                                best_mask = mask
                                best_sum = current_sum
            elif r < 0.6:
                # Remove
                in_set = [x for x in range(1, 257) if (mask >> x) & 1]
                if len(in_set) > 1:
                    y = random.choice(in_set)
                    new_mask = _remove_bit(mask, y)
                    new_sum = current_sum - 1.0/y
                    # Removing always decreases sum, so only accept if it enables better swaps later?
                    # In SA, we might accept worse states.
                    if random.random() < math.exp((new_sum - current_sum)/T):
                        mask = new_mask
                        current_sum = new_sum
            else:
                # Swap
                in_set = [x for x in range(1, 257) if (mask >> x) & 1]
                not_in_set = [x for x in range(1, 257) if not ((mask >> x) & 1)]
                if in_set and not_in_set:
                    y = random.choice(in_set)
                    x = random.choice(not_in_set)
                    temp_mask = _remove_bit(mask, y)
                    if _can_add_bitset(temp_mask, x):
                        new_mask = _add_bit(temp_mask, x)
                        new_sum = current_sum - 1.0/y + 1.0/x
                        if new_sum > current_sum or random.random() < math.exp((new_sum - current_sum)/T):
                            mask = new_mask
                            current_sum = new_sum
                            if current_sum > best_sum:
                                best_mask = mask
                                best_sum = current_sum
                            
            T *= 0.999
            if T < 1e-6:
                break
                
        mask = best_mask
        
    # Final polish
    if _elapsed() < _DEADLINE - 5:
        mask, _ = _local_search_bitset(mask, max_iter=5000)
        
    # Convert mask to list
    result = []
    m = mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        result.append(x)
        m ^= lsb
        
    if not result:
        result = [1]
        
    return result
