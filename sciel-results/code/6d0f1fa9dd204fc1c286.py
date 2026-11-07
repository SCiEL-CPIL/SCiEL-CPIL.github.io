import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    max_val = n
    
    # Precompute pairs_sum[s] = list of (a,b) with a<b, a+b=s
    pairs_sum = [[] for _ in range(max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= max_val:
                pairs_sum[s].append((a, b))
    
    # Precompute sum_mask[x] = bitmask of y such that x+y <= max_val
    # This is used to quickly find candidates c such that b+c <= max_val
    sum_mask = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        m = 0
        # y can be 1 to max_val - x
        # Bits 0 to max_val - x - 1 are set
        if max_val - x > 0:
            m = (1 << (max_val - x)) - 1
        sum_mask[x] = m

    def is_valid_addition(x, mask):
        """Check if adding x to valid set 'mask' keeps it valid."""
        if mask & (1 << (x - 1)):
            return True
        
        # Case 1: x is a member.
        # Need b,c in S, b!=c, such that x+b in S, x+c in S, and b+c in S (or b+c=x, but b,c in S and b+c=x implies x is sum, handled in Case 2? 
        # Wait, if b+c=x, then the sums are x, x+b, x+c. 
        # The condition is: a,b,c distinct members. Sums a+b, a+c, b+c must be in S.
        # If x is a member, let the triple be {x, b, c}.
        # Sums: x+b, x+c, b+c.
        # All must be in S_new = S U {x}.
        # Since x is new, x+b and x+c cannot be x (b,c>0). So x+b in S, x+c in S.
        # b+c can be in S or b+c=x.
        
        # Subcase 1a: b+c in S.
        # We need b,c in S such that x+b in S, x+c in S, b+c in S.
        # Let A = {b in S | x+b in S}.
        # We need b,c in A, b!=c, such that b+c in S.
        
        A_bits = 0
        # b must be in S and x+b in S.
        # b <= max_val - x
        # Iterate through bits of mask, but only those <= max_val - x
        # Actually, just iterate 1..max_val-x
        # Optimization: Use bit operations if possible, but loop is small (<=128)
        for b in range(1, max_val - x + 1):
            if (mask & (1 << (b - 1))) and (mask & (1 << (x + b - 1))):
                A_bits |= (1 << (b - 1))
        
        # Check if any pair in A_bits sums to an element in S
        # m_a holds bits of A
        m_a = A_bits
        while m_a:
            lsb_b = m_a & -m_a
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m_a ^= lsb_b
            
            # Need c in A_bits, c != b, such that b+c in S
            # c must be in A_bits and c <= max_val - b
            # candidates_c = A_bits & sum_mask[b_val] & ~(1 << b_idx)
            candidates_c = A_bits & sum_mask[b_val] & ~(1 << b_idx)
            if candidates_c:
                # Check if any c in candidates_c has b+c in S
                # We can iterate candidates_c
                m_c = candidates_c
                while m_c:
                    lsb_c = m_c & -m_c
                    c_idx = lsb_c.bit_length() - 1
                    c_val = c_idx + 1
                    sum_bc = b_val + c_val
                    if mask & (1 << (sum_bc - 1)):
                        return False
                    m_c ^= lsb_c

        # Subcase 1b: b+c = x.
        # We need b,c in S, b+c=x, such that x+b in S and x+c in S.
        # Note: if b+c=x, then x+b = x + b = (b+c) + b = 2b+c.
        # x+c = b+2c.
        for a, b in pairs_sum[x]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))):
                # Check if x+a in S and x+b in S
                if x + a <= max_val and x + b <= max_val:
                    if (mask & (1 << (x + a - 1))) and (mask & (1 << (x + b - 1))):
                        return False

        # Case 2: x is a sum.
        # Triple {a, b, c} such that one sum is x.
        # WLOG a+b=x. a,b in S.
        # Then x+a and x+b must be in S_new.
        # And a+c, b+c must be in S_new.
        # Since x is new, x+a != x (a>0), so x+a must be in S.
        # Similarly x+b in S.
        # Wait, the sums are a+b=x, a+c, b+c.
        # So we need:
        # 1. a,b in S, a+b=x.
        # 2. There exists c in S_new \ {a,b} such that a+c in S_new and b+c in S_new.
        
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))): continue
            if not (mask & (1 << (b - 1))): continue
            
            # We need c in S_new \ {a,b}
            # S_new = S U {x}
            # So c can be in S \ {a,b} or c=x.
            
            # If c=x:
            # Need a+x in S_new and b+x in S_new.
            # a+x and b+x cannot be x. So they must be in S.
            if x + a <= max_val and x + b <= max_val:
                if (mask & (1 << (x + a - 1))) and (mask & (1 << (x + b - 1))):
                    return False
            
            # If c in S \ {a,b}:
            # Need a+c in S_new and b+c in S_new.
            # a+c cannot be x? If a+c=x, then c=x-a=b. But c!=b. So a+c != x.
            # So a+c must be in S. Similarly b+c in S.
            # We need c in S \ {a,b} such that a+c in S and b+c in S.
            
            # candidates = S & {c | a+c in S} & {c | b+c in S} & ~{a,b}
            # Let's compute masks for c
            # c must be in S.
            # a+c in S => c in S_a where S_a = {c | a+c in S}
            # b+c in S => c in S_b where S_b = {c | b+c in S}
            
            # We can iterate c in S? Or use bitmasks.
            # Let's iterate c in S \ {a,b}
            # This is O(N).
            # To speed up, precompute? No, S changes.
            
            # Optimization:
            # candidates_mask = mask & ~(1 << (a-1)) & ~(1 << (b-1))
            # We need c in candidates_mask such that (a+c in S) and (b+c in S).
            
            # Let's iterate bits of candidates_mask
            m_c = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
            while m_c:
                lsb_c = m_c & -m_c
                c_idx = lsb_c.bit_length() - 1
                c_val = c_idx + 1
                m_c ^= lsb_c
                
                if a + c_val <= max_val and b + c_val <= max_val:
                    if (mask & (1 << (a + c_val - 1))) and (mask & (1 << (b + c_val - 1))):
                        return False

        return True

    # Initial solution: Greedy
    # Start with empty, add 1..128 if valid?
    # Or start with upper half and try to add lower?
    # Let's start with a random set of size ~60 and improve.
    # Better: Greedy from 1 to 128.
    
    best_mask = 0
    best_size = 0
    current_mask = 0
    current_size = 0
    
    # Greedy Pass 1: Add numbers 1..128
    for i in range(1, n + 1):
        if is_valid_addition(i, current_mask):
            current_mask |= (1 << (i - 1))
            current_size += 1
    
    best_mask = current_mask
    best_size = current_size
    
    # Local Search
    # We will try to add unused elements, and swap if stuck.
    
    # To avoid timeout, limit iterations or time.
    iterations = 0
    max_iterations = 100000 # Will be cut by time
    
    while time.time() < deadline:
        iterations += 1
        
        # Try to add a random unused element
        unused = [i for i in range(1, n + 1) if not (current_mask & (1 << (i - 1)))]
        
        if unused:
            random.shuffle(unused)
            added = False
            # Try first few
            for x in unused[:10]:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    break
            if added:
                continue
        
        # If no add, try a swap (remove 1, add 1) or (remove 2, add 2)
        elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
        
        if elements and unused:
            # Try 1-out 1-in
            x_out = random.choice(elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            # Find valid additions to mask_trial
            valid_adds = []
            for x in unused:
                if is_valid_addition(x, mask_trial):
                    valid_adds.append(x)
            
            if valid_adds:
                x_in = random.choice(valid_adds)
                current_mask = mask_trial | (1 << (x_in - 1))
                # Size remains same
                # We don't update best_size unless it increases, which it doesn't here.
                # But this might allow future additions.
                
        # If stuck, try 2-out 2-in? Or just perturb.
        # Perturbation: remove 1 random element
        if elements:
            if random.random() < 0.5:
                x_out = random.choice(elements)
                current_mask &= ~(1 << (x_out - 1))
                current_size -= 1

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
    
    return sorted(result)
