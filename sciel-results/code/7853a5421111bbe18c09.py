import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def check_disjoint(sums):
        """Check if all sums[k] are pairwise disjoint."""
        n = len(sums)
        for i in range(n):
            for j in range(i + 1, n):
                if sums[i] & sums[j]:
                    return False
        return True

    def can_add(sums, x):
        """Check if adding x keeps sums disjoint."""
        # Create new sums
        # sums is a list of ints (bitsets)
        # We need to update backwards
        new_sums = sums[:]
        for k in range(len(new_sums) - 1, -1, -1):
            if new_sums[k]:
                new_sums[k + 1] |= (new_sums[k] << x)
        
        # Check disjointness
        n = len(new_sums)
        for i in range(n):
            for j in range(i + 1, n):
                if new_sums[i] & new_sums[j]:
                    return False, None
        return True, new_sums

    def greedy_construct(order, deadline):
        elements = []
        # sums[k] is a bitset of achievable sums with cardinality k
        # Max sum for k elements is k * 256.
        # We start with sums[0] = 1 (sum 0 is achievable)
        sums = [1]  # index 0
        
        for val in order:
            if time.time() > deadline:
                break
            # We need to extend sums list if we add an element?
            # No, sums list length grows with number of elements.
            # But we don't know final size.
            # We can just use a list that we extend.
            
            # Check if we can add val
            # We need to simulate adding val to current sums
            # Current max cardinality is len(elements)
            # New max cardinality will be len(elements) + 1
            
            # Let's prepare the check
            # We need to check against existing sums AND new sums
            # It's easier to just compute new_sums and check all pairs
            
            # To save memory/time, we only check up to current len + 1
            # But the check function above checks all.
            
            # Optimization: 
            # The new sums are formed from old sums.
            # Only the "new" parts can conflict.
            # But checking all is safer and fast enough for small n.
            
            ok, new_sums = can_add(sums, val)
            if ok:
                elements.append(val)
                sums = new_sums
                # If sums list grew, we are good.
                # Note: can_add returns new_sums of length len(sums)+1?
                # No, my can_add assumes sums is fixed length?
                # Let's fix can_add to handle growth.
                
        return elements

    # Revised can_add to handle growth
    def can_add_grow(sums, x):
        n = len(sums)
        new_sums = [0] * (n + 1)
        # Copy old
        for i in range(n):
            new_sums[i] = sums[i]
        # Add x
        for k in range(n - 1, -1, -1):
            if sums[k]:
                new_sums[k + 1] |= (sums[k] << x)
        
        # Check disjointness for all pairs in new_sums
        for i in range(len(new_sums)):
            for j in range(i + 1, len(new_sums)):
                if new_sums[i] & new_sums[j]:
                    return False, None
        return True, new_sums

    def greedy_construct_v2(order, deadline):
        elements = []
        sums = [1]  # bit 0 set
        
        for val in order:
            if time.time() > deadline:
                break
            ok, new_sums = can_add_grow(sums, val)
            if ok:
                elements.append(val)
                sums = new_sums
        return elements

    best = []
    best_size = 0

    # Try different orders
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(129, 257)) + list(range(1, 129)),
        list(range(1, 129)) + list(range(256, 128, -1))
    ]
    
    random.seed(42)
    for _ in range(20):
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        res = greedy_construct_v2(order, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Local search / improvement
    # Try to swap elements to allow more additions?
    # Or just try to add more elements if we stopped early?
    # The greedy stops when no more can be added in that order.
    # We can try to fill remaining slots.
    
    # A more powerful search: 
    # If we have a set of size K, can we find a set of size K+1?
    # This is hard.
    
    # Let's just return the best found.
    # Ensure types
    return [int(x) for x in best]
