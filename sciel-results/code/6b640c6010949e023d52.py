import time
import random

def construct():
    deadline = time.time() + 80
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
        
    # Precompute conflict checks for speed
    conflict_bits = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_bits[x].append((bits[y], bits[z]))

    # Precompute for each x, the list of (y, z, by, bz) for faster swap analysis
    conflict_detail = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_detail[x].append((y, z, bits[y], bits[z]))

    def can_add(x, mask):
        for by, bz in conflict_bits[x]:
            if (mask & by) and (mask & bz):
                return False
        return True

    def get_active_conflicts(x, mask):
        """Return list of (y, z) pairs where both y and z are in mask and conflict with x."""
        active = []
        for y, z, by, bz in conflict_detail[x]:
            if (mask & by) and (mask & bz):
                active.append((y, z))
        return active

    # Initial Solution: Greedy from 1 to N
    # This tends to include more small numbers than just taking the upper half
    mask = 0
    for i in range(1, N + 1):
        if can_add(i, mask):
            mask |= bits[i]
            
    best_mask = mask
    best_size = bin(mask).count("1")
    
    random.seed(42)
    
    # Local Search
    stuck_count = 0
    max_stuck = 50
    
    for iteration in range(10000000):
        if time.time() > deadline:
            break
            
        # Pick a random element x to try to add
        x = random.randint(1, N)
        
        if mask & bits[x]:
            continue
            
        # Try to add x
        if can_add(x, mask):
            mask |= bits[x]
            size = bin(mask).count("1")
            if size > best_size:
                best_mask = mask
                best_size = size
            stuck_count = 0
            continue
            
        # x is blocked. Try to find a swap or kick.
        found_swap = False
        
        # Get active conflicts
        active_pairs = get_active_conflicts(x, mask)
        
        if active_pairs:
            # Try 1-swap: remove one element from an active conflict
            for _ in range(min(3, len(active_pairs))):
                if time.time() > deadline:
                    break
                y, z = random.choice(active_pairs)
                
                # Try removing y
                mask ^= bits[y]
                if can_add(x, mask):
                    mask |= bits[x]
                    found_swap = True
                else:
                    mask ^= bits[y]
                    
                if found_swap:
                    break
                    
                # Try removing z
                mask ^= bits[z]
                if can_add(x, mask):
                    mask |= bits[x]
                    found_swap = True
                else:
                    mask ^= bits[z]
                    
                if found_swap:
                    break
            
            # If 1-swap worked, try to grow
            if found_swap:
                for _ in range(3):
                    w = random.randint(1, N)
                    if not (mask & bits[w]):
                        if can_add(w, mask):
                            mask |= bits[w]
                            size = bin(mask).count("1")
                            if size > best_size:
                                best_mask = mask
                                best_size = size
                                stuck_count = 0
                                break
                continue
        
        # If no swap worked, try a Kick
        if stuck_count >= max_stuck or random.random() < 0.08:
            # Targeted kick: remove elements that are blocking x
            removed = []
            kick_size = random.randint(2, 4)
            
            # Prefer removing elements from active conflicts
            if active_pairs:
                for y, z in active_pairs:
                    if len(removed) >= kick_size:
                        break
                    if mask & bits[y]:
                        removed.append(y)
                        mask ^= bits[y]
                    if mask & bits[z]:
                        removed.append(z)
                        mask ^= bits[z]
            
            # Fill remaining kick slots with random elements
            while len(removed) < kick_size:
                u = random.randint(1, N)
                if (mask & bits[u]) and u not in removed:
                    removed.append(u)
                    mask ^= bits[u]
            
            # Try to add x
            if can_add(x, mask):
                mask |= bits[x]
                
            # Greedily add back as many as possible, prioritizing removed ones
            random.shuffle(removed)
            for u in removed:
                if can_add(u, mask):
                    mask |= bits[u]
                    
            # Then try random others
            for _ in range(8):
                w = random.randint(1, N)
                if not (mask & bits[w]):
                    if can_add(w, mask):
                        mask |= bits[w]
                        
            stuck_count = 0
            size = bin(mask).count("1")
            if size > best_size:
                best_mask = mask
                best_size = size
            continue
            
        stuck_count += 1
        
    # Final polish pass: try to add any remaining elements
    polish_mask = best_mask
    for i in range(1, N + 1):
        if not (polish_mask & bits[i]):
            if can_add(i, polish_mask):
                polish_mask |= bits[i]
    
    polish_size = bin(polish_mask).count("1")
    if polish_size > best_size:
        best_mask = polish_mask
        best_size = polish_size
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & bits[i]:
            result.append(i)
            
    return result
