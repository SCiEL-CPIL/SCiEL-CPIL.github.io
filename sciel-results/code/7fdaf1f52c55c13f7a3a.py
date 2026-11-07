import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Initial set: {63, ..., 128}
    # Bitmask: bit i is 1 if i is in the set.
    # We use bits 1..128. Bit 0 is 0.
    mask = 0
    for i in range(63, 129):
        mask |= (1 << i)
    
    best_mask = mask
    best_size = 66
    
    all_nums = set(range(1, 129))
    
    def get_size(m):
        return bin(m).count('1')
        
    def can_add(m, x):
        # Check if x can be added to set represented by m
        if (m >> x) & 1:
            return False # x already in set
        
        # N_x = { z in S | z+x in S }
        # z+x in S means bit (z+x) in m is 1.
        # Shift m right by x: bit z becomes 1 if bit (z+x) was 1.
        # But we only care about z in S.
        # So N_x_mask = (m >> x) & m
        Nx_mask = (m >> x) & m
        
        # Check if there are a, b in N_x (a != b) such that a+b in S or a+b == x
        # Iterate over a in N_x
        # To be fast, we can just iterate. N_x is small.
        # Extract indices of set bits in Nx_mask
        # Since max bit is 128, we can just loop 1..128
        # But we can do better.
        
        # For each a in N_x:
        #   We need b in N_x, b != a, such that (m >> (a+b)) & 1 or a+b == x
        
        # Optimization:
        # If a+b == x, then b = x-a.
        # Check if b in N_x and b != a and b > 0.
        
        # If a+b in S, check if (m >> (a+b)) & 1.
        
        # Let's iterate over set bits in Nx_mask
        temp = Nx_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check b = x - a
            b1 = x - a
            if b1 > 0 and b1 != a:
                if (Nx_mask >> b1) & 1:
                    return False
            
            # Check other b in N_x
            # We can just check if there is any b in N_x such that a+b in S
            # This is O(|Nx|) per a, so O(|Nx|^2).
            # |Nx| is likely small.
            
            # To speed up, we can precompute sums? No, dynamic.
            # Let's just loop b.
            temp_b = Nx_mask
            while temp_b:
                lsb_b = temp_b & -temp_b
                b = lsb_b.bit_length() - 1
                temp_b ^= lsb_b
                
                if b == a:
                    continue
                
                s = a + b
                if s == x:
                    return False
                if (m >> s) & 1:
                    return False
        
        return True

    def can_swap(m, y, x):
        # Remove y, add x
        if (m >> x) & 1:
            return False
        if not ((m >> y) & 1):
            return False
            
        # New set S' = S \ {y} U {x}
        # Check violation involving x.
        # a, b in S' such that x+a in S', x+b in S', a+b in S'
        
        # x+a in S' => x+a in S (since x+a > x, cannot be x)
        # So a must be in S \ {y} such that x+a in S.
        # Nx_mask = { z in S \ {y} | z+x in S }
        # (m >> x) & m gives z in S such that z+x in S.
        # We must exclude y.
        Nx_mask = ((m >> x) & m) & ~(1 << y)
        
        # Now check for a, b in Nx_mask (a != b)
        # such that a+b in S'
        # a+b in S' means a+b in S OR a+b == x
        # (a+b cannot be y because y is removed, so if a+b==y, it's not in S')
        
        temp = Nx_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check b = x - a
            b1 = x - a
            if b1 > 0 and b1 != a:
                if (Nx_mask >> b1) & 1:
                    return False
            
            # Check other b in Nx_mask
            temp_b = Nx_mask
            while temp_b:
                lsb_b = temp_b & -temp_b
                b = lsb_b.bit_length() - 1
                temp_b ^= lsb_b
                
                if b == a:
                    continue
                
                s = a + b
                if s == x:
                    return False
                # Check if s in S
                # Note: if s == y, it is NOT in S'. So we should NOT return False if s==y.
                # But we check (m >> s) & 1. If s==y, (m>>y)&1 is 1.
                # So we must explicitly exclude s==y.
                if s != y and ((m >> s) & 1):
                    return False
        
        return True

    # Local Search
    current_mask = mask
    current_size = get_size(current_mask)
    
    # List of elements in current set for easy access
    def get_elements(m):
        elems = []
        for i in range(1, 129):
            if (m >> i) & 1:
                elems.append(i)
        return elems
    
    # Precompute list of all numbers
    all_list = list(range(1, 129))
    
    # Randomized Local Search
    # We will try to add, then swap, then double swap.
    
    # To avoid getting stuck, we use a simple hill climbing with random restarts if stuck?
    # Or just extensive random swaps.
    
    # Phase 1: Greedy Add
    while time.time() < deadline:
        added = False
        # Try to add any element
        missing = [x for x in all_list if not ((current_mask >> x) & 1)]
        random.shuffle(missing)
        for x in missing:
            if time.time() > deadline:
                break
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                added = True
                break
        if not added:
            break
            
    # Phase 2: Random Swaps
    # We try to swap y in S with x not in S.
    # To make it efficient, we pick random x not in S, and random y in S.
    
    stuck_count = 0
    while time.time() < deadline and stuck_count < 500:
        missing = [x for x in all_list if not ((current_mask >> x) & 1)]
        if not missing:
            break
        present = [x for x in all_list if (current_mask >> x) & 1]
        
        if not present:
            break
            
        x = random.choice(missing)
        y = random.choice(present)
        
        if can_swap(current_mask, y, x):
            # Perform swap
            current_mask &= ~(1 << y)
            current_mask |= (1 << x)
            # Size stays same
            stuck_count = 0
            # Try to add after swap
            # (Loop back to top of while? No, we are in swap loop. 
            # We can check add here or just let next iteration handle it if we structure it well.
            # Actually, let's just continue swapping.
        else:
            stuck_count += 1
            
        # Occasionally try to add
        if stuck_count % 10 == 0:
             added = False
             missing_now = [x for x in all_list if not ((current_mask >> x) & 1)]
             random.shuffle(missing_now)
             for x in missing_now:
                 if can_add(current_mask, x):
                     current_mask |= (1 << x)
                     current_size += 1
                     if current_size > best_size:
                         best_size = current_size
                         best_mask = current_mask
                     added = True
                     break
             if added:
                 stuck_count = 0

    # Final check
    if current_size > best_size:
        best_size = current_size
        best_mask = current_mask

    # Convert best_mask to list
    result = []
    for i in range(1, 129):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
