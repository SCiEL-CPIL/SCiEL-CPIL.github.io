import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    n = 256
    
    # Precompute conflict pairs
    # conflict_pairs[a] = list of (b, c) such that a < b <= c and a | (b+c)
    conflict_pairs = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    pairs.append((b, c))
        conflict_pairs[a] = pairs

    # Precompute for fast bitset checks
    # We will represent S as an integer bitmask.
    # Bit i is 1 if i is in S.
    
    def is_valid_bits(mask):
        """
        Check if the set represented by mask is valid.
        """
        # Iterate through all a in mask
        # We need to iterate set bits
        m = mask
        while m:
            # Get lowest set bit
            a = (m & -m).bit_length() - 1
            m &= m - 1 # clear lowest set bit
            
            if a == 1:
                # If 1 is in S, at most one other element can be in S
                # count bits in mask
                if mask.bit_count() > 2:
                    return False
                # Actually if 1 in S, and size > 1, we need to check the other one
                # If size == 2, {1, x}. Is it valid?
                # a=1, b=x, c=x? No, distinct a,b,c?
                # "a < b <= c". If S={1, x}, we don't have 3 elements.
                # The condition applies to "every a<b<=c in the set".
                # This implies we pick 3 elements? Or any 3?
                # "For every a<b<=c in the set" usually means for all distinct a,b,c from the set.
                # If |S| < 3, the condition is vacuously true.
                # So {1, x} is valid.
                # {1, x, y} with x<y: 1 < x <= y. 1 | (x+y) -> True. Invalid.
                # So if 1 in S, |S| <= 2.
                if mask.bit_count() > 2:
                    return False
                continue

            # Check conflicts for a
            for b, c in conflict_pairs[a]:
                # Check if b and c are in mask
                if (mask >> b) & 1 and (mask >> c) & 1:
                    return False
        
        return True

    def get_conflicting_elements(mask, x):
        """
        Returns list of elements y in S such that y is involved in a conflict with x.
        """
        conflicts = set()
        
        if x == 1:
            # 1 conflicts with any pair (b,c) in S
            # If there are >= 2 elements in S, they conflict with 1
            # We need to identify which ones to remove.
            # Any element in S (except 1) is part of a conflict if there is another.
            # To make S U {1} valid, we must reduce S to size <= 1.
            # So we should remove all but one?
            # This is a "many-swap" case.
            # For now, return all elements in S.
            m = mask
            while m:
                y = (m & -m).bit_length() - 1
                m &= m - 1
                if y != x:
                    conflicts.add(y)
            return list(conflicts)

        # Check if x is 'a'
        for b, c in conflict_pairs[x]:
            if (mask >> b) & 1:
                conflicts.add(b)
            if (mask >> c) & 1:
                conflicts.add(c)
        
        # Check if x is 'b' or 'c'
        # We need to check all a < x such that a | (x+c) or a | (b+x)
        # It's easier to iterate a in S, a < x
        m = mask
        while m:
            a = (m & -m).bit_length() - 1
            m &= m - 1
            if a >= x:
                continue
            
            # Case 1: x is b. Need c in S, c >= x, a | (x+c)
            # Case 2: x is c. Need b in S, a < b <= x, a | (b+x)
            
            # Check Case 1: x is b
            # c must be in S, c >= x
            # a | (x+c) => x+c = k*a
            # Iterate c in S, c >= x
            # This is slow.
            # Alternative: Precompute for each x, the set of a that conflict with x as b?
            
            # Let's just check all pairs in S?
            # No, let's iterate potential c from S.
            # To speed up, we can precompute `conflict_as_b[x]` = set of (a, c) ?
            
            pass # Placeholder for optimization

        # Fallback: Brute force check for conflicts where x is b or c
        # This is O(|S|^2)
        S_list = []
        m = mask
        while m:
            y = (m & -m).bit_length() - 1
            m &= m - 1
            if y != x:
                S_list.append(y)
        
        for a in S_list:
            if a >= x:
                continue
            # Check if x is b
            for c in S_list:
                if c < x:
                    continue
                if a < x <= c and a | (x + c):
                    conflicts.add(a)
                    conflicts.add(c)
            # Check if x is c
            for b in S_list:
                if b > x:
                    continue
                if a < b <= x and a | (b + x):
                    conflicts.add(a)
                    conflicts.add(b)
                    
        return list(conflicts)

    # Step 1: Find best starting interval
    best_mask = 0
    best_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline - 20:
            break
        mask = 0
        for i in range(L, n + 1):
            mask |= (1 << i)
        
        if is_valid_bits(mask):
            # Greedy add smaller elements
            for x in range(1, L):
                if time.time() > deadline - 20:
                    break
                new_mask = mask | (1 << x)
                if is_valid_bits(new_mask):
                    mask = new_mask
            
            size = mask.bit_count()
            if size > best_size:
                best_size = size
                best_mask = mask
    
    if best_mask == 0:
        best_mask = 1 << n
        best_size = 1

    # Step 2: Local Search
    current_mask = best_mask
    current_size = best_size
    
    # Precompute list of all elements
    all_elements = list(range(1, n + 1))
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        random.shuffle(all_elements)
        
        for x in all_elements:
            if time.time() > deadline - 2:
                break
            
            if (current_mask >> x) & 1:
                continue
            
            # Try to add x
            new_mask = current_mask | (1 << x)
            if is_valid_bits(new_mask):
                current_mask = new_mask
                current_size += 1
                best_size = current_size
                best_mask = current_mask
                improved = True
            else:
                # Try 1-swap
                # Find conflicting elements
                conflicts = get_conflicting_elements(current_mask, x)
                if not conflicts:
                    # Should not happen if is_valid is correct
                    continue
                
                # Try removing each conflicting element
                swapped = False
                # Sort conflicts by "importance" (e.g., degree) or just random
                random.shuffle(conflicts)
                for y in conflicts:
                    if y == x:
                        continue
                    if not ((current_mask >> y) & 1):
                        continue
                    
                    swap_mask = current_mask | (1 << x)
                    swap_mask &= ~(1 << y)
                    
                    if is_valid_bits(swap_mask):
                        current_mask = swap_mask
                        # Size stays same
                        improved = True
                        swapped = True
                        break
                
                # If no 1-swap worked, try 2-swap (remove 2, add 1)
                # This reduces size, so we only do it if we are stuck and want to explore?
                # For maximization, we generally avoid size reduction.
                # But if we are stuck, we might need to.
                # Let's skip 2-swap for now to keep size high.
                
                if swapped:
                    # After a swap, we might be able to add more elements
                    # Restart the loop? Or just continue.
                    pass

    return list(i for i in range(1, n + 1) if (best_mask >> i) & 1)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
