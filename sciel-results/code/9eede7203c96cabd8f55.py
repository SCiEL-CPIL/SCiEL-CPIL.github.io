import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Tighter safety margin
    
    n = 128
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    # 1. Precompute conflict structures
    # bad_triples: list of (a, b, c) such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # conflict_pairs[x]: set of (u, v) such that {u, v, x} is a bad triple.
    # This means if u and v are in the set, x cannot be added.
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == L_ab and _lcm(b, c) == L_ab:
                    bad_triples.append((a, b, c))
                    
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute degrees (number of conflicting pairs) for each element
    degrees = [len(conflict_pairs[i]) for i in range(n + 1)]
    
    # Bitmask representation: element i is represented by bit (i-1)
    # mask is an integer where bit j is 1 if element j+1 is in the set.
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for u, v in conflict_pairs[x]:
            # Check if both u and v are in mask
            # Bit index for element k is k-1
            if (mask >> (u - 1)) & 1 and (mask >> (v - 1)) & 1:
                return False
        return True

    def popcount(mask):
        return bin(mask).count('1')
        
    def greedy(order, mask=0):
        """Greedy construction given an order of elements."""
        selected = mask
        for x in order:
            if not (selected >> (x - 1)) & 1:
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def local_search(mask):
        """
        Fast local search:
        1. Greedy add any valid element.
        2. If stuck, try removing one random element and greedily filling again.
        """
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Greedy add pass
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            if not improved:
                # 2. Perturbation: Remove one, fill greedily
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                if not s_list:
                    break
                
                # Try a few random removals to escape local optima
                for _ in range(20):
                    if time.time() > deadline - 2:
                        break
                    x = random.choice(s_list)
                    mask_no_x = mask & ~(1 << (x - 1))
                    new_mask = mask_no_x
                    # Greedy fill
                    for y in range(1, n + 1):
                        if not (new_mask >> (y - 1)) & 1:
                            if can_add(y, new_mask):
                                new_mask |= (1 << (y - 1))
                    
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        break # Restart the while loop with the new mask
                        
        return mask

    best_mask = 0
    best_size = 0
    
    # Main loop: Multi-start Greedy + Fast Local Search
    num_restarts = 0
    while time.time() < deadline - 2:
        num_restarts += 1
        
        # Generate order
        r = random.random()
        if r < 0.4:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.7:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x])
        else:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
            
        mask = greedy(order)
        
        # Local Search
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Final polish
    best_mask = local_search(best_mask)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# Helper for popcount if not using bin (bin is fast enough for 128 bits)
# Already defined inside construct

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
