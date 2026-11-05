import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bits
    bits = [0] * (N + 1)
    for i in range(1, N + 1):
        bits[i] = 1 << (i - 1)
        
    # For each x, precompute a list of (bit_y, bit_z) pairs
    conflict_bits = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_bits[x].append((bits[y], bits[z]))
    
    def can_add(x, mask):
        for by, bz in conflict_bits[x]:
            if (mask & by) and (mask & bz):
                return False
        return True
    
    def mask_size(mask):
        return bin(mask).count("1")
    
    def greedy_from(initial_mask):
        """Add elements greedily in order 1..N that can be added."""
        mask = initial_mask
        size = mask_size(mask)
        for x in range(1, N + 1):
            if not (mask & bits[x]) and can_add(x, mask):
                mask |= bits[x]
                size += 1
        return mask, size
    
    def local_search(mask, size, max_iterations=50000):
        """Local search with add, swap, and kick moves."""
        best_mask = mask
        best_size = size
        random.seed(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random element
            x = random.randint(1, N)
            if not (mask & bits[x]):
                if can_add(x, mask):
                    mask |= bits[x]
                    size += 1
                    if size > best_size:
                        best_mask = mask
                        best_size = size
                    continue
            
            # Try a swap: remove one element u from S, add x
            if not (mask & bits[x]):
                # Find an element u in S such that removing u allows adding x
                # Try a few random candidates
                for _ in range(3):
                    u = random.randint(1, N)
                    if mask & bits[u]:
                        mask ^= bits[u]  # remove u
                        if can_add(x, mask):
                            mask |= bits[x]
                            if size > best_size:
                                best_mask = mask
                                best_size = size
                            continue
                        else:
                            mask ^= bits[u]  # restore u
            
            # Try to add a random element that was previously blocked
            w = random.randint(1, N)
            if not (mask & bits[w]) and can_add(w, mask):
                mask |= bits[w]
                size += 1
                if size > best_size:
                    best_mask = mask
                    best_size = size
                continue
            
            # Kick: remove 1-2 random elements, then greedily refill
            if random.random() < 0.03:
                kick_size = random.randint(1, 2)
                for _ in range(kick_size):
                    while True:
                        u = random.randint(1, N)
                        if mask & bits[u]:
                            mask ^= bits[u]
                            size -= 1
                            break
                
                # Greedily try to add elements back
                for v in range(1, N + 1):
                    if not (mask & bits[v]) and can_add(v, mask):
                        mask |= bits[v]
                        size += 1
                
                if size > best_size:
                    best_mask = mask
                    best_size = size
        
        return best_mask, best_size
    
    # Try multiple initial solutions
    best_mask = 0
    best_size = 0
    
    # Initial solution 1: upper half
    mask1 = 0
    for i in range(91, N + 1):
        mask1 |= bits[i]
    m1, s1 = greedy_from(mask1)
    if s1 > best_size:
        best_mask = m1
        best_size = s1
    
    # Initial solution 2: lower half
    mask2 = 0
    for i in range(1, 91):
        mask2 |= bits[i]
    m2, s2 = greedy_from(mask2)
    if s2 > best_size:
        best_mask = m2
        best_size = s2
    
    # Initial solution 3: random
    random.seed(42)
    mask3 = 0
    for i in range(1, N + 1):
        if random.random() < 0.5:
            mask3 |= bits[i]
    m3, s3 = greedy_from(mask3)
    if s3 > best_size:
        best_mask = m3
        best_size = s3
    
    # Local search from best initial
    best_mask, best_size = local_search(best_mask, best_size)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & bits[i]:
            result.append(i)
            
    return result
