import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Precompute nothing complex, just use bitmasks for speed.
    # S is represented as an integer bitmask. Bit i is set if i is in S.
    # Indices 1..128 correspond to bits 1..128. Bit 0 is unused.
    
    def can_add(mask, x):
        """
        Check if adding x to the set represented by mask results in a valid set.
        Violation condition: 
        There exist distinct a, b, c in S_new such that a+b, a+c, b+c are in S_new.
        If x is new, the only new sums involving x are x+a, x+b.
        So a violation is created if there exist a, b in S_old such that:
        1. a, b, x are distinct.
        2. a+b in S_old (since a+b doesn't involve x, it must be in S_old).
        3. a+x in S_new. Since x is new, a+x must be in S_old (as a+x != x).
        4. b+x in S_new. Similarly, b+x must be in S_old.
        
        So we need to check if there exist a, b in S_old such that:
        - a != b
        - a != x, b != x
        - (mask has bit a+b)
        - (mask has bit a+x)
        - (mask has bit b+x)
        """
        # Find candidates a in S_old such that a+x is in S_old
        # a+x <= 128 => a <= 128-x
        # a >= 1
        # Also a != x
        
        # Iterate through set bits in mask to find a
        # To be faster, we can iterate a from 1 to 128-x
        # But checking bit in mask is O(1).
        
        # Let's collect valid 'a's
        # a must be in S_old, a+x must be in S_old.
        # Let N = { a in S_old | a+x in S_old }
        
        # We need a, b in N such that a+b in S_old and a!=b.
        
        # Build list of N
        N = []
        for a in range(1, 129 - x):
            if (mask >> a) & 1 and (mask >> (a + x)) & 1:
                if a != x:
                    N.append(a)
        
        # Now check pairs in N
        n_len = len(N)
        for i in range(n_len):
            a = N[i]
            for j in range(i + 1, n_len):
                b = N[j]
                s = a + b
                if s <= 128 and ((mask >> s) & 1):
                    return False
        
        return True

    def get_size(mask):
        return bin(mask).count('1')

    def greedy_construct(order):
        mask = 0
        for x in order:
            if time.time() > deadline:
                break
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def local_search(mask, max_iters=2000):
        S = set()
        # Convert mask to set for easier manipulation in swap logic?
        # Or just use mask and iterate bits.
        # Iterating bits in Python is slow.
        # Let's convert to set for the local search, as |S| is small (~50).
        
        # Extract set from mask
        S = set()
        m = mask
        i = 1
        while m:
            if m & 1:
                S.add(i)
            m >>= 1
            i += 1
        
        not_in_S = set(range(1, 129)) - S
        best_size = len(S)
        best_mask = mask
        
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            progress = False
            
            # Try to add
            # Shuffle not_in_S for variety? Or sorted?
            # Adding is strictly better if valid.
            for x in list(not_in_S):
                if time.time() > deadline:
                    break
                # Convert S to mask for can_add? 
                # can_add takes mask.
                # We can maintain mask in sync.
                pass # This approach is messy. Let's stick to mask only.
            
            # Let's rewrite local search to use mask directly.
            # To iterate elements in mask, we need a helper.
            def bits_in_mask(m):
                res = []
                i = 1
                while m:
                    if m & 1:
                        res.append(i)
                    m >>= 1
                    i += 1
                return res
            
            S_list = bits_in_mask(mask)
            not_in_S_list = [x for x in range(1, 129) if not (mask >> x) & 1]
            
            # Try Add
            if not_in_S_list:
                random.shuffle(not_in_S_list)
                for x in not_in_S_list:
                    if time.time() > deadline:
                        break
                    if can_add(mask, x):
                        mask |= (1 << x)
                        progress = True
                        break
            
            if progress:
                continue
                
            # Try Swap
            if S_list and not_in_S_list:
                # Limit swaps to avoid excessive time
                random.shuffle(S_list)
                random.shuffle(not_in_S_list)
                found = False
                for x in not_in_S_list[:20]:
                    if time.time() > deadline:
                        break
                    for y in S_list[:20]:
                        if time.time() > deadline:
                            break
                        # Try swap y -> x
                        # New mask = mask with y removed, x added
                        new_mask = mask & ~(1 << y)
                        if can_add(new_mask, x):
                            mask = new_mask | (1 << x)
                            progress = True
                            found = True
                            break
                    if found:
                        break
            
            if not progress:
                break
                
        return mask

    best_mask = 0
    best_size = 0
    
    def update_best(mask):
        nonlocal best_mask, best_size
        sz = get_size(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Phase 1: Greedy
    update_best(greedy_construct(list(range(1, 129))))
    update_best(greedy_construct(list(range(128, 0, -1))))
    
    # Phase 2: Random Greedy + Local Search
    for i in range(500):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        m = greedy_construct(candidates)
        m = local_search(m, max_iters=1000)
        update_best(m)
        
    # Phase 3: Local Search on Best
    if time.time() < deadline:
        m = local_search(best_mask, max_iters=5000)
        update_best(m)
        
    # Convert best_mask to list of ints
    result = []
    m = best_mask
    i = 1
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
        
    return result

