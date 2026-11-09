import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    all_mask = (1 << n) - 1
    
    # Precompute: for each x, list of (a,b) with a<b, a+b==x
    pairs_sum = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s <= n:
                pairs_sum[s].append((a, b))
    
    def is_valid_addition(mask, x):
        """
        Check if adding element x (1-indexed) to the set represented by mask is valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True  # Already in set
        
        # Case 1: x is a vertex (c=x)
        # Need a, b in S distinct from x such that:
        #   a+x in S, b+x in S, and a+b in S
        
        # Find all a in S such that a+x in S
        candidates_a = []
        for a in range(1, n - x + 1):
            if (mask >> (a - 1)) & 1:
                if (mask >> (a + x - 1)) & 1:
                    if a != x:
                        candidates_a.append(a)
        
        k = len(candidates_a)
        if k >= 2:
            for i in range(k):
                a = candidates_a[i]
                for j in range(i + 1, k):
                    b = candidates_a[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n:
                        if (mask >> (s - 1)) & 1:
                            return False
        
        # Case 2: x is a sum (x = a+b)
        # Need c in S, c distinct from a,b, such that a+c in S and b+c in S
        for (a, b) in pairs_sum[x]:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                C_a_mask = mask & (mask >> a)
                C_b_mask = mask & (mask >> b)
                potential_c_mask = C_a_mask & C_b_mask
                
                # Exclude c=a and c=b
                potential_c_mask &= ~(1 << (a - 1))
                potential_c_mask &= ~(1 << (b - 1))
                
                if potential_c_mask != 0:
                    return False
        
        return True
    
    def get_size(mask):
        return bin(mask).count('1')
    
    def greedy_construct(order, time_limit):
        """Build a valid set by greedily adding vertices in the given order."""
        mask = 0
        for x in order:
            if time.time() > time_limit:
                break
            if is_valid_addition(mask, x):
                mask |= (1 << (x - 1))
        return mask
    
    all_elements = list(range(1, n + 1))
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Greedy construction with various orderings
    greedy_deadline = start_time + 15  # Spend 15s on greedy
    
    for trial in range(200):
        if time.time() > greedy_deadline:
            break
        
        # Different ordering strategies
        if trial == 0:
            order = list(range(65, 129)) + list(range(1, 65))
        elif trial == 1:
            order = all_elements[:]
            random.shuffle(order)
        elif trial == 2:
            order = list(range(n, 0, -1))
        elif trial == 3:
            order = list(range(128, 64, -1)) + list(range(1, 65))
        elif trial % 3 == 0:
            upper = list(range(65, 129))
            lower = list(range(1, 65))
            random.shuffle(upper)
            random.shuffle(lower)
            order = upper + lower
        elif trial % 3 == 1:
            order = all_elements[:]
            random.shuffle(order)
        else:
            mid = list(range(40, 90))
            rest = list(set(all_elements) - set(mid))
            random.shuffle(mid)
            random.shuffle(rest)
            order = mid + rest
        
        mask = greedy_construct(order, greedy_deadline)
        size = get_size(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Local search to improve
    current_mask = best_mask
    current_size = best_size
    
    local_search_deadline = deadline
    
    attempts = 0
    max_attempts = 50000000  # High limit, time is the guard
    
    while time.time() < local_search_deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a random missing element
        x = random.randint(1, n)
        if not (current_mask >> (x - 1)) & 1:
            if is_valid_addition(current_mask, x):
                current_mask |= (1 << (x - 1))
                new_size = get_size(current_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = current_mask
                    current_size = new_size
        
        # Occasionally remove an element to escape local optima
        if random.random() < 0.10:
            present = [i + 1 for i in range(n) if (current_mask >> i) & 1]
            if present:
                idx = random.choice(present)
                current_mask &= ~(1 << (idx - 1))
        
        # If stuck (size dropped significantly), restart from best
        if get_size(current_mask) < best_size - 5:
            current_mask = best_mask
    
    # Convert best_mask to list of ints
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
