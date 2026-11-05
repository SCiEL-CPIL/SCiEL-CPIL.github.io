import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute conflicts
    # For each x, conflicts[x] is a list of (b, c) such that 1/x = 1/b + 1/c, b < c, b,c > x
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflicts[a].append((b, c))
    
    # Precompute bitmasks
    bits = [0] * (N + 1)
    for i in range(1, N + 1):
        bits[i] = 1 << (i - 1)
        
    # For fast checking: for each x, store a list of (mask_b, mask_c)
    # Actually, we can just check if (mask & mask_b) and (mask & mask_c)
    conflict_masks = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for b, c in conflicts[x]:
            conflict_masks[x].append((bits[b], bits[c]))
            
    def can_add(x, mask):
        for mb, mc in conflict_masks[x]:
            if (mask & mb) and (mask & mc):
                return False
        return True
        
    def mask_size(mask):
        return bin(mask).count("1")
        
    def local_search(initial_mask, max_time=5.0):
        start_time = time.time()
        mask = initial_mask
        size = mask_size(mask)
        best_mask = mask
        best_size = size
        
        # Lists of elements in and out of mask for faster random selection
        in_mask = [i for i in range(1, N + 1) if mask & bits[i]]
        out_mask = [i for i in range(1, N + 1) if not (mask & bits[i])]
        
        while time.time() - start_time < max_time:
            if time.time() > deadline - 2:
                break
                
            # Try add
            if out_mask:
                x = random.choice(out_mask)
                if can_add(x, mask):
                    mask |= bits[x]
                    size += 1
                    in_mask.append(x)
                    out_mask.remove(x)
                    if size > best_size:
                        best_mask = mask
                        best_size = size
                        # Optional: restart local search from this better point?
                        # For now, just continue.
                        
            # Try swap
            if out_mask and in_mask:
                x = random.choice(out_mask)
                u = random.choice(in_mask)
                mask ^= bits[u] # remove u
                if can_add(x, mask):
                    mask |= bits[x] # add x
                    # size unchanged
                    in_mask.remove(u)
                    in_mask.append(x)
                    out_mask.remove(x)
                    out_mask.append(u)
                    # We don't update best here unless we do a full check, 
                    # but size is same. We might improve structure for future moves.
                    # However, we want to maximize size. Swaps don't increase size.
                    # But they can enable future adds.
                    
            # Try 2-swap
            if len(out_mask) >= 2 and len(in_mask) >= 2:
                x1 = random.choice(out_mask)
                x2 = random.choice([y for y in out_mask if y != x1])
                u1 = random.choice(in_mask)
                u2 = random.choice([y for y in in_mask if y != u1])
                
                mask ^= bits[u1]
                mask ^= bits[u2]
                
                if can_add(x1, mask):
                    mask |= bits[x1]
                    if can_add(x2, mask):
                        mask |= bits[x2]
                        # size unchanged
                        in_mask.remove(u1)
                        in_mask.remove(u2)
                        in_mask.append(x1)
                        in_mask.append(x2)
                        out_mask.remove(x1)
                        out_mask.remove(x2)
                        out_mask.append(u1)
                        out_mask.append(u2)
                    else:
                        # undo x1
                        mask ^= bits[x1]
                        # undo x2 was not added
                        # restore u1, u2
                        mask |= bits[u1]
                        mask |= bits[u2]
                        in_mask.remove(x1)
                        in_mask.remove(x2)
                        in_mask.append(u1)
                        in_mask.append(u2)
                        out_mask.remove(u1)
                        out_mask.remove(u2)
                        out_mask.append(x1)
                        out_mask.append(x2)
                else:
                    # restore u1, u2
                    mask |= bits[u1]
                    mask |= bits[u2]
                    in_mask.remove(x1)
                    in_mask.remove(x2)
                    in_mask.append(u1)
                    in_mask.append(u2)
                    out_mask.remove(u1)
                    out_mask.remove(u2)
                    out_mask.append(x1)
                    out_mask.append(x2)
                    
        return best_mask, best_size

    # Initial solution: Upper half
    mask1 = 0
    for i in range(91, N + 1):
        mask1 |= bits[i]
    # Greedy cleanup
    for i in range(91, N + 1):
        if mask1 & bits[i]:
            if not can_add(i, mask1 & ~bits[i]):
                mask1 ^= bits[i]
                
    best_mask = mask1
    best_size = mask_size(mask1)
    
    # Run multiple local searches
    random.seed(42)
    num_runs = 10
    run_time = (deadline - time.time() - 2) / num_runs
    
    for _ in range(num_runs):
        if time.time() > deadline - 2:
            break
        
        # Perturb current best or start fresh
        if random.random() < 0.5 and best_size > 0:
            start_mask = best_mask
            # Randomly remove a few elements
            for _ in range(3):
                if best_mask:
                    # pick random set bit
                    tmp = best_mask
                    while tmp:
                        if random.random() < 0.1:
                            break
                        tmp &= tmp - 1
                    if tmp:
                        # find the bit
                        # This is inefficient, just use in_mask list if we had it
                        # For simplicity, just pick random i
                        i = random.randint(1, N)
                        if best_mask & bits[i]:
                            start_mask ^= bits[i]
        else:
            # Start from upper half again
            start_mask = 0
            for i in range(91, N + 1):
                if random.random() < 0.8:
                    start_mask |= bits[i]
            
        m, s = local_search(start_mask, max_time=run_time)
        if s > best_size:
            best_mask = m
            best_size = s
            
    result = []
    for i in range(1, N + 1):
        if best_mask & bits[i]:
            result.append(i)
            
    return result
