import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill

    # Bitmask for fast membership and AP checking
    mask = 0
    N_mask = (1 << (N + 1)) - 1  # bits 0..N

    def can_add(x):
        """Check if x can be added to current set (represented by mask) without creating a 3-AP."""
        if mask & (1 << x):
            return False
        # Case 1: x is middle. a + c = 2x. c = 2x - a.
        # For each a in S, if c in S and c != a, invalid.
        # We iterate a from 1 to N. If a in S, check c.
        # Optimization: only check a such that c is in [1, N].
        # c = 2x - a => a = 2x - c.
        # Range of a: if c in [1, N], then a in [2x-N, 2x-1].
        # Also a must be in [1, N].
        # So a in [max(1, 2x-N), min(N, 2x-1)].
        # Also we need c != a => 2x - a != a => a != x.
        
        # Case 2: x is end. 
        # Subcase 2a: x is largest. a, b, x. 2b = a+x. b = (a+x)/2.
        # a must be such that b is integer and in S.
        # a+x must be even. b in [1, N].
        # a = 2b - x.
        # b in [max(1, (x+1)//2 + 1?), ...]. Actually just iterate b?
        # If we iterate b in S, check a = 2b - x. If a in S and a != b and a != x, invalid.
        # This is O(K).
        
        # Subcase 2b: x is smallest. x, b, c. 2b = x+c. c = 2b - x.
        # If we iterate b in S, check c = 2b - x. If c in S and c != b and c != x, invalid.
        # This is O(K).
        
        # Combined efficient check using bit operations?
        # Let's stick to iterating over set bits for clarity and speed in Python if set size is moderate.
        # But bitmask allows fast "is a in S" via (mask >> a) & 1.
        
        # Let's iterate over all possible a in S? No, that's O(N) per check.
        # We can iterate over the set bits of mask.
        
        # Alternative: 
        # 1. x is middle: 
        #    We need to know if there exists a in S such that (2x - a) in S.
        #    This is equivalent to checking if (S & (2x - S)) is non-empty (excluding x itself if 2x-x=x).
        #    Let S_mask = mask.
        #    Let T_mask = 0
        #    For each bit a set in S_mask:
        #       c = 2x - a
        #       if 1 <= c <= N and c != a:
        #           if S_mask has bit c: return False
        #       if c == x: # a=x, then c=x. 2x-x=x. (x,x,x) is constant AP, ignored.
        #           pass
        #
        # 2. x is end (a is middle):
        #    We need to know if there exists b in S such that (2b - x) in S.
        #    Let b be in S. a = 2b - x.
        #    If 1 <= a <= N and a != b and a != x:
        #       if S_mask has bit a: return False
        #
        # This is O(K) per check. K is size of S.
        # With N=256, K is probably ~100-150. O(150) per check is fast.
        
        # Iterate through set bits of mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            
            # Case 1: x is middle. a, x, c. c = 2x - a.
            c = 2 * x - a
            if 1 <= c <= N and c != a:
                if mask & (1 << c):
                    return False
            
            # Case 2: x is end. a, b, x or x, b, a. 
            # Actually, the condition "b is middle" covers both:
            # If b in S, and a = 2b - x is in S, then (a, b, x) or (x, b, a) is AP.
            # Here 'a' from loop is 'b' in this context.
            # So: let b = a (the element from S). 
            # Check if 2*b - x is in S.
            c2 = 2 * a - x
            if 1 <= c2 <= N and c2 != a and c2 != x:
                if mask & (1 << c2):
                    return False
            
            # Note: In Case 1, we checked if c=2x-a is in S.
            # In Case 2, we checked if c2=2a-x is in S.
            # These are distinct.
            
        return True

    def get_score():
        # Sum of reciprocals
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            s += 1.0 / x
            m ^= lsb
        return s

    # 1. Greedy Initialization
    # Try to add numbers in order of decreasing reciprocal weight (smallest first)
    # This is a heuristic: small numbers have high weight, so we prioritize them.
    # But we must avoid APs.
    
    global_mask = 0
    for x in range(1, N + 1):
        if time.time() - start_time > time_limit - 5:
            break
        if can_add(x):
            global_mask |= (1 << x)

    current_mask = global_mask
    current_score = get_score()
    best_mask = current_mask
    best_score = current_score
    
    # 2. Local Search
    # Moves: Add, Remove, Swap
    # We will use a simple hill climbing with some random restarts/perturbations if stuck.
    
    all_numbers = list(range(1, N + 1))
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_mask = current_mask
        valid = True
        
        if move_type == 'add':
            # Find candidates not in mask
            # Efficient way: iterate 1..N, check bit
            candidates = []
            for x in all_numbers:
                if not (current_mask & (1 << x)):
                    candidates.append(x)
            if not candidates:
                continue
            x = random.choice(candidates)
            # Check validity
            # We need to temporarily set x and check? No, can_add checks against current mask.
            # But can_add uses global 'mask' variable. We need to pass mask or make it local.
            # Let's refactor can_add to take mask as argument.
            
            # Refactoring can_add to take mask
            def can_add_mask(x, m):
                if m & (1 << x):
                    return False
                mm = m
                while mm:
                    lsb = mm & -mm
                    a = lsb.bit_length() - 1
                    mm ^= lsb
                    c = 2 * x - a
                    if 1 <= c <= N and c != a:
                        if m & (1 << c):
                            return False
                    c2 = 2 * a - x
                    if 1 <= c2 <= N and c2 != a and c2 != x:
                        if m & (1 << c2):
                            return False
                return True

            if can_add_mask(x, current_mask):
                new_mask |= (1 << x)
            else:
                continue

        elif move_type == 'remove':
            # Pick random x in current_mask
            m = current_mask
            if m == 0:
                continue
            # Pick random set bit
            # Convert to list of set bits for random choice? Or use random index?
            # N is small, list is fine.
            set_bits = []
            mm = m
            while mm:
                lsb = mm & -mm
                x = lsb.bit_length() - 1
                set_bits.append(x)
                mm ^= lsb
            if not set_bits:
                continue
            x = random.choice(set_bits)
            new_mask &= ~(1 << x)

        elif move_type == 'swap':
            # Pick x in S, y not in S
            m = current_mask
            set_bits = []
            mm = m
            while mm:
                lsb = mm & -mm
                x = lsb.bit_length() - 1
                set_bits.append(x)
                mm ^= lsb
            if not set_bits:
                continue
            x = random.choice(set_bits)
            
            candidates = []
            for y in all_numbers:
                if not (current_mask & (1 << y)):
                    candidates.append(y)
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Remove x, check if y can be added
            temp_mask = current_mask & ~(1 << x)
            if can_add_mask(y, temp_mask):
                new_mask = temp_mask | (1 << y)
            else:
                continue

        # Evaluate new state
        # Calculate score incrementally? 
        # get_score is O(K). K~100. O(100) per move is fine.
        new_score = get_score() # Wait, get_score uses global 'mask'. 
        # I need to make get_score take mask as argument too, or update global mask.
        # Let's make a helper.
        
        def score_of_mask(m):
            s = 0.0
            mm = m
            while mm:
                lsb = mm & -mm
                x = lsb.bit_length() - 1
                s += 1.0 / x
                mm ^= lsb
            return s

        new_score = score_of_mask(new_mask)
        
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            stuck_count = 0
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        else:
            stuck_count += 1
            # If stuck for a while, maybe perturb?
            # For now, just continue. Randomness will help.
            # We can also try to accept worse moves with small prob (Simulated Annealing)
            # But simple hill climbing with random restarts is often robust.
            # Let's add a small probability of accepting worse moves to escape local optima.
            # Probability = exp(-delta / T)
            # T starts high, cools down.
            # This requires tracking T.
            
            # Let's implement simple SA.
            # T = 1.0 initially, cool by 0.99 each step.
            # But we need to manage T outside the loop.
            
            # Let's restructure to include SA parameters.
            
            # Since I'm inside the loop, I'll just use a fixed small prob for now 
            # or rely on the 'add'/'remove'/'swap' randomness.
            # Actually, let's implement proper SA in the main loop.
            
            # Breaking out to restructure with SA variables.
            # I will rewrite the loop part to include T.
            
            # For now, let's just return best if time is running low.
            # But I want to improve.
            
            # Let's assume the current hill climbing is sufficient for this small N.
            # The greedy init is strong.
            
    # Extract best set
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        result.append(x)
        m ^= lsb
        
    return result

# Wait, the previous block had a nested function definition inside the loop which is inefficient.
# And I defined can_add_mask inside the loop.
# Let's clean up the code structure.

def construct():
    N = 256
    start_time = time.time()
    time_limit = 80.0

    def can_add_mask(x, m):
        if m & (1 << x):
            return False
        mm = m
        while mm:
            lsb = mm & -mm
            a = lsb.bit_length() - 1
            mm ^= lsb
            # x is middle
            c = 2 * x - a
            if 1 <= c <= N and c != a:
                if m & (1 << c):
                    return False
            # x is end (a is middle)
            c2 = 2 * a - x
            if 1 <= c2 <= N and c2 != a and c2 != x:
                if m & (1 << c2):
                    return False
        return True

    def score_of_mask(m):
        s = 0.0
        mm = m
        while mm:
            lsb = mm & -mm
            x = lsb.bit_length() - 1
            s += 1.0 / x
            mm ^= lsb
        return s

    # Greedy Init
    current_mask = 0
    for x in range(1, N + 1):
        if time.time() - start_time > time_limit - 5:
            break
        if can_add_mask(x, current_mask):
            current_mask |= (1 << x)

    current_score = score_of_mask(current_mask)
    best_mask = current_mask
    best_score = current_score

    all_numbers = list(range(1, N + 1))
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001

    while time.time() - start_time < time_limit and T > min_T:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_mask = current_mask
        valid = True

        if move_type == 'add':
            # Find candidates not in mask
            # Optimization: Precompute list of not-in-mask? 
            # Or just iterate.
            # For N=256, iterating is fine.
            candidates = [x for x in all_numbers if not (current_mask & (1 << x))]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add_mask(x, current_mask):
                new_mask |= (1 << x)
            else:
                continue

        elif move_type == 'remove':
            # Pick random x in current_mask
            set_bits = []
            mm = current_mask
            while mm:
                lsb = mm & -mm
                x = lsb.bit_length() - 1
                set_bits.append(x)
                mm ^= lsb
            if not set_bits:
                continue
            x = random.choice(set_bits)
            new_mask &= ~(1 << x)

        elif move_type == 'swap':
            set_bits = []
            mm = current_mask
            while mm:
                lsb = mm & -mm
                x = lsb.bit_length() - 1
                set_bits.append(x)
                mm ^= lsb
            if not set_bits:
                continue
            x = random.choice(set_bits)
            
            candidates = [y for y in all_numbers if not (current_mask & (1 << y))]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            temp_mask = current_mask & ~(1 << x)
            if can_add_mask(y, temp_mask):
                new_mask = temp_mask | (1 << y)
            else:
                continue

        new_score = score_of_mask(new_mask)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling

    # Extract best set
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        result.append(x)
        m ^= lsb
        
    return result
