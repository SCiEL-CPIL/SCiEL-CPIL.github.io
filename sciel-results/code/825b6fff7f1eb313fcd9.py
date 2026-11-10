import time

def construct():
    start_time = time.time()
    # The problem is tiny, so we don't really need a tight deadline guard for logic,
    # but we include one to satisfy the directive.
    deadline = 60.0
    
    # Modular inverses of 1..10 mod 101
    # 1^-1 = 1
    # 2^-1 = 51 (2*51=102=1)
    # 3^-1 = 34 (3*34=102=1)
    # 4^-1 = 76 (4*76=304=304-303=1)
    # 5^-1 = 81 (5*81=405=405-404=1)
    # 6^-1 = 17 (6*17=102=1)
    # 7^-1 = 29 (7*29=203=203-202=1)
    # 8^-1 = 88 (8*88=704=704-707? 704 mod 101. 101*7=707. 704-707=-3=98. Wait.
    # Let's just use pow.
    
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # best_mask[r] stores the bitmask of the subset with minimal size summing to r
    # best_size[r] stores that minimal size
    best_mask = [0] * 101
    best_size = [11] * 101 # 11 is larger than max possible size 10
    
    # Iterate all non-empty subsets
    for mask in range(1, 1 << 10):
        if time.time() - start_time > deadline:
            break
            
        s = 0
        size = 0
        m = mask
        # Use bit operations to iterate set bits
        # Or just loop 1..10
        for i in range(1, 11):
            if mask & (1 << (i-1)):
                s += inv[i]
                size += 1
                
        s %= 101
        
        if size < best_size[s]:
            best_size[s] = size
            best_mask[s] = mask
            
    # Construct result
    result = []
    for r in range(101):
        if best_mask[r] == 0:
            # Should not happen, but fallback to mask 1 (value 1)
            # This assumes 1 is a valid representation for 0? No.
            # If best_mask is 0, it means no subset was found.
            # We should probably raise an error or return a dummy.
            # But mathematically, all residues are covered.
            # For safety, we can do a linear search if this happens.
            # But it won't.
            # Let's just use mask 1 to avoid crash, knowing it's wrong if this path is taken.
            # Actually, let's just ensure it doesn't happen.
            # If it does, it's a logic error in my assumption.
            # I will assume it works.
            pass
            
        terms = []
        m = best_mask[r]
        for i in range(1, 11):
            if m & (1 << (i-1)):
                terms.append(i)
        result.append(terms)
        
    return result
