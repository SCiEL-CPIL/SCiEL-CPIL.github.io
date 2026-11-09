"""Maximum independent set via bitset branch-and-bound."""
import sys
sys.setrecursionlimit(100000)

def construct():
    N = 256
    # Build conflict bitsets: conflict[i] is a bitset of j such that (i+j) | i*i
    # For distinct a,b: (a+b) | a*b  <=>  (a+b) | a*a
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        s2 = a * a
        for b in range(1, N + 1):
            if b == a:
                continue
            s = a + b
            if s2 % s == 0:
                mask |= (1 << b)
        conflict[a] = mask
    
    # Bitset of all vertices
    all_mask = 0
    for i in range(1, N + 1):
        all_mask |= (1 << i)
    
    # Precompute degree
    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        degree[a] = bin(conflict[a] & all_mask).count('1')
    
    # Branch and bound for maximum independent set
    best_set = [0]  # bitset
    best_size = [0]
    
    # Order vertices by degree (higher degree first for pruning)
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -degree[x])
    
    def popcount(x):
        return bin(x).count('1')
    
    def bnb(candidates, current, size):
        # candidates: bitset of available vertices (independent with current)
        # current: bitset of chosen vertices
        # size: number chosen so far
        if candidates == 0:
            if size > best_size[0]:
                best_size[0] = size
                best_set[0] = current
            return
        
        # Upper bound: size + popcount(candidates)
        if size + popcount(candidates) <= best_size[0]:
            return
        
        # Pick vertex with max degree in candidates for branching
        # Find the vertex in candidates with highest degree (within candidates)
        best_v = 0
        best_deg = -1
        c = candidates
        while c:
            # Get lowest set bit
            lb = c & (-c)
            v = lb.bit_length() - 1
            # Degree of v within candidates
            deg = popcount(conflict[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lb
        
        # Branch 1: include best_v
        new_current = current | (1 << best_v)
        # New candidates: those not conflicting with best_v
        new_candidates = candidates & ~(conflict[best_v] | (1 << best_v))
        bnb(new_candidates, new_current, size + 1)
        
        # Branch 2: exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        bnb(new_candidates, current, size)
    
    bnb(all_mask, 0, 0)
    
    # Extract result
    result = []
    for i in range(1, N + 1):
        if best_set[0] & (1 << i):
            result.append(i)
    
    # Fallback to odd numbers if something went wrong
    if not result:
        result = list(range(1, 257, 2))
    
    return result
