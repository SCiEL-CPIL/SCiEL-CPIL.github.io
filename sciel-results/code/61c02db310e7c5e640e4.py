import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflicts using bitmasks for speed
    # conflicts_mask[i] is a bitmask of j such that (i*j) % (i+j) == 0
    conflicts_mask = [0] * (n + 1)
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts_mask[a] |= (1 << b)
                conflicts_mask[b] |= (1 << a)
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_valid_set(s):
        """Check if set s is a valid independent set using bitmask."""
        mask = 0
        for a in s:
            if conflicts_mask[a] & mask:
                return False
            mask |= (1 << a)
        return True
    
    def greedy_build(order, start_mask=0):
        """Build a set by greedily adding elements from order that don't conflict."""
        s = set()
        mask = start_mask
        for e in order:
            if not (conflicts_mask[e] & mask):
                s.add(e)
                mask |= (1 << e)
        return s, mask
    
    def local_search(s, max_iters=2000):
        """Local search with add, 1-swap (1-for-1), and 2-swap (2-for-1) moves."""
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        best_size = len(s)
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # 1. Try to add any non-member
            for b in range(1, n + 1):
                if not (mask & (1 << b)):
                    if not (conflicts_mask[b] & mask):
                        s.add(b)
                        mask |= (1 << b)
                        best_size = len(s)
                        improved = True
                        break
            if improved:
                continue
            
            # 2. Try 1-swap: remove a, add b
            # b must conflict with exactly one element a in S
            for b in range(1, n + 1):
                if mask & (1 << b):
                    continue
                conflicts_with_b = conflicts_mask[b] & mask
                if conflicts_with_b == 0:
                    continue  # Should have been caught above
                if conflicts_with_b & (conflicts_with_b - 1) == 0:
                    # Exactly one conflict
                    a_bit = conflicts_with_b
                    a = a_bit.bit_length() - 1
                    new_mask = (mask ^ a_bit) | (1 << b)
                    s.discard(a)
                    s.add(b)
                    mask = new_mask
                    best_size = len(s)
                    improved = True
                    break
            if improved:
                continue
            
            # 3. Try 2-swap: remove a, c, add b
            # b must conflict with exactly two elements a, c in S
            for b in range(1, n + 1):
                if mask & (1 << b):
                    continue
                conflicts_with_b = conflicts_mask[b] & mask
                if conflicts_with_b == 0:
                    continue
                # Check if exactly 2 bits set
                if conflicts_with_b & (conflicts_with_b - 1) == 0:
                    continue  # Only 1 conflict, handled above
                if (conflicts_with_b & (conflicts_with_b - 1)) & (conflicts_with_b & (conflicts_with_b - 1) - 1) == 0:
                    # Exactly 2 conflicts
                    # Extract the two bits
                    a_bit = conflicts_with_b & (-conflicts_with_b)
                    c_bit = conflicts_with_b ^ a_bit
                    a = a_bit.bit_length() - 1
                    c = c_bit.bit_length() - 1
                    # After removing a and c, check that b doesn't conflict with remaining S
                    # b's conflicts with S are exactly {a, c}, so after removing them, no conflict
                    new_mask = (mask & ~a_bit & ~c_bit) | (1 << b)
                    s.discard(a)
                    s.discard(c)
                    s.add(b)
                    mask = new_mask
                    # Size stays same (remove 2, add 1), but might enable further improvements
                    improved = True
                    break
            if improved:
                continue
            
            if not improved:
                break
                
        return s, best_size

    best = []
    best_size = 0
    
    # Phase 1: Initial Construction
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))
    
    # Sort evens by number of conflicts with odds
    def conflict_count(e):
        # Count bits in conflicts_mask[e] that are in odds
        odds_mask = 0
        for o in odds:
            odds_mask |= (1 << o)
        return bin(conflicts_mask[e] & odds_mask).count('1')
    
    evens_sorted = sorted(evens, key=conflict_count)
    
    s, mask = greedy_build(evens_sorted, 0)
    # Start with odds
    s_full = set(odds)
    odds_mask = 0
    for o in odds:
        odds_mask |= (1 << o)
    for e in evens_sorted:
        if not (conflicts_mask[e] & (odds_mask | 0)):
            # Check against current s_full
            pass
    
    # Simpler: build from scratch
    s = set(odds)
    mask = 0
    for o in odds:
        mask |= (1 << o)
    for e in evens_sorted:
        if not (conflicts_mask[e] & mask):
            s.add(e)
            mask |= (1 << e)
    
    s, sz = local_search(s, max_iters=3000)
    if sz > best_size:
        best = list(s)
        best_size = sz
        
    # Phase 2: Restart with Randomized Kick
    all_elements = list(range(1, n + 1))
    
    for trial in range(80):
        if time.time() > deadline - 5:
            break
            
        # Start from best and kick
        s = set(best)
        
        # Kick: remove K random elements
        K = random.randint(5, 15)
        to_remove = random.sample(list(s), min(K, len(s)))
        for r in to_remove:
            s.discard(r)
            
        # Rebuild mask
        mask = 0
        for a in s:
            mask |= (1 << a)
            
        # Greedily re-add from all elements not in s, in random order
        remaining = [e for e in all_elements if e not in s]
        random.shuffle(remaining)
        
        for e in remaining:
            if not (conflicts_mask[e] & mask):
                s.add(e)
                mask |= (1 << e)
                
        # Local search
        s, sz = local_search(s, max_iters=1500)
        
        if sz > best_size:
            best = list(s)
            best_size = sz
            
    return best
