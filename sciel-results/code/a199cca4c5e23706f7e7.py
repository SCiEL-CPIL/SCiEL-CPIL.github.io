import random
import time

def construct():
    n = 512
    max_time = 55.0
    start_time = time.time()
    
    def can_add(mask, x):
        idx = x - 1
        if (mask >> idx) & 1:
            return False
        
        # Case 1: x is 4th term
        max_d1 = (x - 1) // 3
        for d in range(1, max_d1 + 1):
            a = x - 3 * d
            if ((mask >> (a - 1)) & 1) and ((mask >> (a + d - 1)) & 1) and ((mask >> (a + 2 * d - 1)) & 1):
                return False

        # Case 2: x is 3rd term
        max_d2 = (x - 1) // 2
        for d in range(1, max_d2 + 1):
            a = x - 2 * d
            b3 = x + d
            if b3 <= n:
                if ((mask >> (a - 1)) & 1) and ((mask >> (a + d - 1)) & 1) and ((mask >> (b3 - 1)) & 1):
                    return False

        # Case 3: x is 2nd term
        # d <= x-1 and x+2d <= n
        max_d3 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d3 + 1):
            a = x - d
            b2 = x + d
            b3 = x + 2 * d
            if ((mask >> (a - 1)) & 1) and ((mask >> (b2 - 1)) & 1) and ((mask >> (b3 - 1)) & 1):
                return False

        # Case 4: x is 1st term
        max_d4 = (n - x) // 3
        for d in range(1, max_d4 + 1):
            b1 = x + d
            b2 = x + 2 * d
            b3 = x + 3 * d
            if ((mask >> (b1 - 1)) & 1) and ((mask >> (b2 - 1)) & 1) and ((mask >> (b3 - 1)) & 1):
                return False

        return True

    def greedy_random():
        mask = 0
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    def greedy_deterministic(reverse=False):
        mask = 0
        order = list(range(1, n + 1))
        if reverse:
            order.reverse()
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    def local_search_improve(mask):
        # Simple hill climbing: try to swap elements to allow more additions
        # We limit the number of attempts to keep it fast
        attempts = 0
        max_attempts = 5000
        current_size = bin(mask).count('1')
        
        while attempts < max_attempts and time.time() - start_time < max_time - 5:
            attempts += 1
            # Pick a random element y to remove
            elements_in = [i+1 for i in range(n) if (mask >> i) & 1]
            if not elements_in:
                break
            y = random.choice(elements_in)
            
            # Pick a random element x to add
            elements_out = [i+1 for i in range(n) if not ((mask >> i) & 1)]
            if not elements_out:
                break
            x = random.choice(elements_out)
            
            # Try swap
            new_mask = (mask & ~(1 << (y - 1))) | (1 << (x - 1))
            
            # Check if new_mask is valid (it should be, since we removed y and added x, 
            # but we must ensure x doesn't form AP with new neighbors? 
            # Wait, removing y might break APs, adding x might create them.
            # We need to check if x can be added to (S - {y}).
            # This is equivalent to can_add(mask & ~(1 << (y-1)), x)
            
            if not can_add(mask & ~(1 << (y - 1)), x):
                continue
                
            # If valid, check if we can add another element z
            # Try to add any element that was previously blocked
            # For speed, just try to add a random few elements from elements_out
            # If we can add one, size increases.
            
            # Actually, if we just swap, size is same.
            # We need to find a z that can be added to new_mask.
            # Let's try to add the first few elements from elements_out that can be added.
            
            improved = False
            # Try a few random candidates for z
            random.shuffle(elements_out)
            for z in elements_out[:10]:
                if z == x:
                    continue
                if can_add(new_mask, z):
                    new_mask |= (1 << (z - 1))
                    improved = True
                    break
            
            if improved:
                mask = new_mask
                current_size = bin(mask).count('1')
                attempts = 0 # Reset attempts on success
        
        return mask

    best_mask = 0
    best_size = 0
    
    # Deterministic baselines
    m1 = greedy_deterministic(False)
    s1 = bin(m1).count('1')
    if s1 > best_size:
        best_size = s1
        best_mask = m1
        
    m2 = greedy_deterministic(True)
    s2 = bin(m2).count('1')
    if s2 > best_size:
        best_size = s2
        best_mask = m2

    # Random greedy with local search on best candidates
    # We keep a list of top candidates
    top_candidates = []
    
    while time.time() - start_time < max_time - 10:
        m = greedy_random()
        s = bin(m).count('1')
        
        # Keep top 5 candidates
        top_candidates.append((s, m))
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        if len(top_candidates) > 5:
            top_candidates.pop()
            
        if s > best_size:
            best_size = s
            best_mask = m

    # Apply local search to the top candidates
    for s, m in top_candidates:
        if time.time() - start_time > max_time - 2:
            break
        m_improved = local_search_improve(m)
        s_improved = bin(m_improved).count('1')
        if s_improved > best_size:
            best_size = s_improved
            best_mask = m_improved

    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
