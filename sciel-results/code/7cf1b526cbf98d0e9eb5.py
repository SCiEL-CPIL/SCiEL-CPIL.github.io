import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    n = 512
    deadline = time.time() + 50  # Safety margin for 60s limit
    
    # 1. Precompute conflict masks
    # conflicts[x] is a list of bitmasks. If (S & mask) == mask, x cannot be added.
    # x is 1-indexed in problem, 0-indexed in bit operations for ease?
    # Let's use 0-indexed for bits: bit i corresponds to element i+1.
    # So element x (1..512) is bit (x-1).
    
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def greedy_construct():
        """Generate a random greedy set."""
        S = 0
        # Shuffle indices
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if can_add(i, S):
                S |= (1 << i)
        return S

    def local_search(S):
        """Improve S using local search."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any element
            # We can iterate through all not-in-S elements
            # To speed up, we can just loop 0..n-1
            for i in range(n):
                if not (S & (1 << i)):
                    if can_add(i, S):
                        S |= (1 << i)
                        improved = True
            if improved:
                continue
                
            # If no direct addition, try a random swap (remove one, try to add others)
            # Pick a random element to remove
            # We need a list of set bits
            # Since n=512, converting to list is fast
            set_bits = []
            temp_S = S
            for i in range(n):
                if temp_S & (1):
                    set_bits.append(i)
                temp_S >>= 1
            
            if not set_bits:
                break
                
            # Try removing a random element
            # We try a few random removals
            for _ in range(10): # Limit random swaps per stuck state to avoid infinite loops
                if time.time() > deadline - 5:
                    return S
                rem_idx = random.choice(set_bits)
                S_new = S & ~(1 << rem_idx)
                
                # Try to fill S_new greedily/randomly
                # Just try to add any available
                added = 0
                # We can use a simple greedy pass on S_new
                # But we want to see if we can recover size
                # Let's just do a few random additions
                # Actually, just try to add ALL valid elements to S_new
                # This is O(n * deg). Fast enough.
                
                # Check if size increases
                # Original size: S.bit_count()
                # New size after filling: ?
                
                # Let's just try to add valid elements to S_new
                # We iterate through all not-in-S_new
                # To avoid bias, we can shuffle or just go 0..n
                # Since we removed one, we have "space"
                
                # Simple heuristic: try to add elements that were blocked by rem_idx?
                # No, just try all.
                
                # Optimization: Only try if S_new has potential.
                # Just run the "try to add" loop on S_new
                # If size(S_new_filled) > size(S), accept.
                
                # To make it faster, we can just try to add one element that was previously blocked?
                # Let's stick to: Remove one, then try to add any valid.
                # If we can add more than 1, we gain.
                
                # Let's implement a fast "fill" that stops if it can't beat current size?
                # No, just fill completely.
                
                # But filling completely is the same as the "try to add" loop above.
                # So:
                S_candidate = S_new
                # Try to add all possible
                for i in range(n):
                    if not (S_candidate & (1 << i)):
                        if can_add(i, S_candidate):
                            S_candidate |= (1 << i)
                
                if S_candidate.bit_count() > S.bit_count():
                    S = S_candidate
                    improved = True
                    break # Restart the while loop with new S
            
            # If we didn't improve in this round of swaps, we might be stuck.
            # We can try a different random removal or break.
            # The loop above tries 10 random removals. If none worked, improved remains False.
            
        return S

    best_S = 0
    best_size = 0
    
    # Initial greedy
    S = greedy_construct()
    best_S = S
    best_size = S.bit_count()
    
    # Local search on initial
    S = local_search(S)
    if S.bit_count() > best_size:
        best_S = S
        best_size = S.bit_count()
    
    # Multiple restarts
    while time.time() < deadline - 5:
        S = greedy_construct()
        # Quick check if this seed is already better
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
        
        S = local_search(S)
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
            
        # If we hit a very high size, we can reduce restart frequency or just continue
        # The local search is the main improver.
        
    # Convert bitset to sorted list
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
