import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # S_large is fixed: 65..128
    # We search for A subset of 1..64
    # Constraint: max pair sum in A < 65
    # Constraint: No distinct a,b,c in A with a+b, a+c, b+c in A
    
    def _is_valid_A(mask):
        # mask is 64-bit int, bit i represents i+1 in A
        # Check internal forbidden triples
        elems = []
        for i in range(64):
            if (mask >> i) & 1:
                elems.append(i + 1)
        
        n = len(elems)
        # Check max pair sum
        if n >= 2:
            if elems[-1] + elems[-2] >= 65:
                return False
        
        # Check triples
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                ab = a + b
                if ab > 64:
                    continue
                if not ((mask >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, n):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > 64 or bc > 64:
                        continue
                    if ((mask >> (ac - 1)) & 1) and ((mask >> (bc - 1)) & 1):
                        return False
        return True

    def _can_add_A(mask, x):
        # x is 1..64
        # Check if adding x to A (represented by mask) keeps it valid
        # 1. Max pair sum constraint
        # If x is new max, check x + current_max < 65
        # If x is new second max, check second_max + current_max < 65 (already true)
        # Actually, just check if adding x violates max pair sum < 65
        # We can just extract elems and check, but that's slow.
        # Better: 
        # Let max1, max2 be largest two in mask.
        # If x > max1: new max pair is x + max1. Need x + max1 < 65.
        # If max1 >= x > max2: new max pair is max1 + x. Need max1 + x < 65.
        # If x <= max2: max pair unchanged.
        
        # Find max1, max2 in mask
        max1 = 0
        max2 = 0
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            val = idx + 1
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
            m ^= lb
        
        if x > max1:
            if x + max1 >= 65:
                return False
        elif x > max2:
            if max1 + x >= 65:
                return False
        # else max pair unchanged, OK
        
        # 2. Internal triple constraint
        # Adding x might create a triple (a, b, x) or (a, x, b) etc.
        # We need to check if there exist distinct a, b in A such that:
        # a+b, a+x, b+x are all in A U {x}
        # Note: a+x and b+x must be in A U {x}.
        # If a+x == x, a=0 (impossible).
        # So a+x must be in A. Similarly b+x in A.
        # And a+b must be in A U {x}.
        # Case 1: a+b in A.
        #   Then we need a+x in A and b+x in A.
        # Case 2: a+b == x.
        #   Then we need a+x in A and b+x in A.
        
        # Iterate over a in A
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            a = idx + 1
            m ^= lb
            
            # Check pairs (a, b) in A
            # We need b != a
            mb = m # Wait, m is modified. Use a copy or re-iterate?
            # Better to just iterate all a,b in A
            pass
        
        # Simpler: just check all pairs a,b in A
        elems = [i+1 for i in range(64) if (mask >> i) & 1]
        for i in range(len(elems)):
            a = elems[i]
            for j in range(len(elems)):
                if i == j: continue
                b = elems[j]
                # Check triple a, b, x
                # Sums: a+b, a+x, b+x
                # All must be in A U {x}
                
                s1 = a + b
                s2 = a + x
                s3 = b + x
                
                # s2, s3 must be in A (since s2!=x, s3!=x for a,b>=1)
                if s2 > 64 or s3 > 64:
                    continue
                if not ((mask >> (s2 - 1)) & 1):
                    continue
                if not ((mask >> (s3 - 1)) & 1):
                    continue
                
                # s1 must be in A or s1 == x
                if s1 == x:
                    return False
                if s1 > 64:
                    continue
                if (mask >> (s1 - 1)) & 1:
                    return False
        
        return True

    best_A = 0
    best_size = 0
    rng = random.Random(42)

    # Greedy construction of A
    # Start with empty, add elements in some order
    # Order: small to large? Large to small?
    # To satisfy max pair sum < 65, we should be careful with large numbers.
    
    def greedy_A(order):
        mask = 0
        for x in order:
            if _can_add_A(mask, x):
                mask |= (1 << (x - 1))
        return mask

    # Try different orders
    base_order = list(range(1, 65))
    
    # 1. Ascending
    A1 = greedy_A(base_order)
    if _is_valid_A(A1):
        s1 = bin(A1).count('1')
        if s1 > best_size:
            best_size = s1
            best_A = A1
            
    # 2. Descending
    A2 = greedy_A(sorted(base_order, reverse=True))
    if _is_valid_A(A2):
        s2 = bin(A2).count('1')
        if s2 > best_size:
            best_size = s2
            best_A = A2
            
    # 3. Odds first
    odds = [x for x in base_order if x % 2 == 1]
    evens = [x for x in base_order if x % 2 == 0]
    A3 = greedy_A(odds + evens)
    if _is_valid_A(A3):
        s3 = bin(A3).count('1')
        if s3 > best_size:
            best_size = s3
            best_A = A3

    # Local search on best_A
    def local_search_A(mask):
        cur = mask
        cur_size = bin(cur).count('1')
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding
            for x in range(1, 65):
                if (cur >> (x-1)) & 1:
                    continue
                if _can_add_A(cur, x):
                    cur |= (1 << (x-1))
                    cur_size += 1
                    improved = True
                    break
            if improved:
                continue
                
            # Try swapping
            elems = [i+1 for i in range(64) if (cur >> i) & 1]
            for r in elems:
                without = cur & ~(1 << (r-1))
                for a in range(1, 65):
                    if (without >> (a-1)) & 1:
                        continue
                    if _can_add_A(without, a):
                        new_mask = without | (1 << (a-1))
                        # Heuristic: prefer larger a? Or just take it?
                        # Taking it might block future additions.
                        # But if we can't add, swapping is better if a > r?
                        # Let's just swap if it allows future growth or a > r
                        # For simplicity, just swap
                        cur = new_mask
                        cur_size = bin(cur).count('1')
                        improved = True
                        break
                if improved:
                    break
                    
        return cur, cur_size

    if time.time() < deadline - 10:
        best_A, best_size = local_search_A(best_A)
        
    # Final result
    result_A = [i+1 for i in range(64) if (best_A >> i) & 1]
    result_large = list(range(65, 129))
    result = result_A + result_large
    
    # Verify
    if not _is_valid_A(best_A):
        # Fallback to odds
        result_A = [x for x in range(1, 65) if x % 2 == 1]
        result = result_A + result_large
        
    return [int(x) for x in result]
