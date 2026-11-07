import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute: for each sum s (1..7680), a bitset of all multiples of s up to 7680
    MAX_SUM = 7680  # 30 * 256
    # multiples_mask[s] is a Python int where bit k is set iff k is a multiple of s
    # We'll use bit index = value, so bit v is set for value v
    multiples_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        for k in range(s, MAX_SUM + 1, s):
            mask |= (1 << k)
        multiples_mask[s] = mask

    def check_divisibility(sums_bitset):
        """Check if any two values in sums_bitset have one dividing the other.
        sums_bitset: Python int where bit v is set iff sum value v is present.
        Returns True if valid (no divisibility conflicts)."""
        # For each set bit s, check if sums_bitset & multiples_mask[s] has any bit set
        # other than s itself. If so, there's a conflict.
        # But we need to check: if s is present and some multiple of s (other than s) is present.
        # Actually, we need to check pairs. For efficiency:
        # If s divides t and s != t, then t is in multiples of s.
        # Check: for each s present, (sums_bitset & multiples_mask[s]) should have at most the bit for s.
        # But this is O(number of set bits) per check. Let's try a different approach.
        
        # Alternative: iterate over set bits and check
        bits = sums_bitset
        while bits:
            lsb = bits & (-bits)
            s = lsb.bit_length() - 1
            # Check if any multiple of s (other than s) is present
            # multiples_mask[s] includes s itself, so subtract bit s
            others = multiples_mask[s] & ~(1 << s)
            if sums_bitset & others:
                return False
            bits ^= lsb
        return True

    def greedy_search(initial_elements, deadline):
        """Greedy search starting from initial elements."""
        elements = list(initial_elements)
        n = len(elements)
        
        # Build initial subset sums bitset
        sums_bitset = 0
        for i in range(1, 1 << n):
            total = 0
            for j in range(n):
                if i & (1 << j):
                    total += elements[j]
            sums_bitset |= (1 << total)
        
        current_size = n
        
        # Try to add elements
        for val in range(1, 257):
            if val in elements:
                continue
            if time.time() > deadline:
                break
            
            # New sums: old_sums + {val} + {old_sum + val for old_sum in old_sums}
            new_sums_bitset = sums_bitset | (1 << val) | (sums_bitset << val)
            
            # Check for sum collisions: if two different subsets give same sum
            # This happens if (sums_bitset << val) & sums_bitset has any bit set
            # (meaning old_sum + val == another_old_sum, i.e., val = another_old_sum - old_sum)
            if (sums_bitset << val) & sums_bitset:
                continue
            
            # Check divisibility
            if not check_divisibility(new_sums_bitset):
                continue
            
            # Valid! Add this element
            elements.append(val)
            sums_bitset = new_sums_bitset
            current_size += 1
            
            if time.time() > deadline:
                break
        
        return elements

    best_result = []
    
    # Strategy 1: Pure greedy from empty
    result = greedy_search([], deadline)
    if len(result) > len(best_result):
        best_result = result[:]
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Strategy 2: Try different starting points
    random.seed(42)
    
    # Try starting with small primes and other values
    starting_sets = []
    # Single elements
    for start in range(2, 50):
        starting_sets.append([start])
    
    # Pairs
    for a in range(2, 30):
        for b in range(a + 1, 50):
            starting_sets.append([a, b])
    
    # Shuffle and try
    random.shuffle(starting_sets)
    
    for i, initial in enumerate(starting_sets):
        if time.time() > deadline:
            break
        result = greedy_search(initial, deadline)
        if len(result) > len(best_result):
            best_result = result[:]
    
    if time.time() > deadline:
        return [int(x) for x in best_result]
    
    # Strategy 3: Backtracking with bitset
    def backtrack(current_elements, sums_bitset, start_val, deadline):
        """Backtracking search."""
        if time.time() > deadline:
            return current_elements
        
        best = current_elements[:]
        
        for val in range(start_val, 257):
            if time.time() > deadline:
                return best
            
            new_sums_bitset = sums_bitset | (1 << val) | (sums_bitset << val)
            
            if (sums_bitset << val) & sums_bitset:
                continue
            
            if not check_divisibility(new_sums_bitset):
                continue
            
            deeper = backtrack(current_elements + [val], new_sums_bitset, val + 1, deadline)
            if len(deeper) > len(best):
                best = deeper
        
        return best

    # Try backtracking from best result's prefix
    if best_result:
        for prefix_len in range(len(best_result), max(0, len(best_result) - 3), -1):
            prefix = best_result[:prefix_len]
            if not prefix:
                continue
            n = len(prefix)
            sb = 0
            for i in range(1, 1 << n):
                total = 0
                for j in range(n):
                    if i & (1 << j):
                        total += prefix[j]
                sb |= (1 << total)
            
            result = backtrack(prefix, sb, prefix[-1] + 1, deadline)
            if len(result) > len(best_result):
                best_result = result[:]
            if time.time() > deadline:
                break
    
    return [int(x) for x in best_result]
