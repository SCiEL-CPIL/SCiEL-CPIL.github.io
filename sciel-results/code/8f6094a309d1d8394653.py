import random
import time
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    for i in range(1, N + 1):
        gcd_table[i][i] = i

    # Precompute conflicts: for each y, which pairs (a,b) in [1..N] form a bad triple with y
    # Use bitmasks for speed: conflict_mask[y] is a set of (a,b) pairs encoded as a
    # For fast checking: for each y, build a list of "forbidden pairs" as frozensets or use a 2D boolean
    # Better approach: for each y, build a bitset of "blocking" elements
    # Actually, the key structural insight: a triple {a,b,y} is bad iff gcd(a,b)==gcd(a,y)==gcd(b,y)
    # Let g = gcd(a,y) = gcd(b,y). Then a = g*a', b = g*b', y = g*y' with gcd(a',y')=gcd(b',y')=1
    # and gcd(a',b')=1. So the condition is: a/g, b/g, y/g are pairwise coprime.
    
    # For fast membership testing, precompute for each y:
    #   block[y] = set of all (a,b) with a<b, a!=y, b!=y, that form bad triple with y
    # But |block[y]| can be large. Instead, use a different approach:
    # For a given set S (as bitmask), adding y is valid iff no pair (a,b) in S is in block[y].
    # We can check this by: for each a in S (a != y), count how many b in S (b != y, b != a) 
    # have gcd(a,b)==gcd(a,y)==gcd(b,y). If any such b exists, invalid.
    
    # Even faster: precompute for each y and each a != y, the set of b such that (a,b,y) is bad.
    # Then adding y to S is valid iff for all a in S, no b in S is in bad_b[y][a].
    # Represent bad_b[y][a] as a bitmask over elements.
    
    # bad_b[y][a] = bitmask of b such that {a,b,y} is a bad triple
    bad_b = [[0] * (N + 1) for _ in range(N + 1)]
    for y in range(1, N + 1):
        for a in range(1, N + 1):
            if a == y:
                continue
            ga = gcd_table[a][y]
            mask = 0
            for b in range(1, N + 1):
                if b == y or b == a:
                    continue
                if gcd_table[a][b] == ga == gcd_table[b][y]:
                    mask |= (1 << (b - 1))
            bad_b[y][a] = mask
    
    # Also build the reverse: for a given set S (bitmask) and candidate y,
    # check validity: for each a in S, check if (S & bad_b[y][a]) != 0
    # But we need to exclude a itself from the mask... actually bad_b[y][a] doesn't include a or y.
    # Wait, it might include elements equal to a? No, we skip b==a. And b==y is skipped.
    # So bad_b[y][a] is a mask over valid b values.
    
    # For checking: given S as a bitmask (bits 0..127 for elements 1..128), and candidate y:
    #   For each a in S (a != y): if (S & bad_b[y][a]) != 0, then invalid.
    #   But we also need to make sure we're not counting a itself. Since bad_b[y][a] doesn't
    #   include a (we skip b==a), this is fine.
    
    def can_add(S_mask, y):
        """Check if element y can be added to set represented by S_mask (bitmask, bit i = element i+1)."""
        y_idx = y - 1
        if S_mask & (1 << y_idx):
            return True  # already in set
        # For each a in S (a != y), check if any b in S is in bad_b[y][a]
        # We only need to check a in S where a != y
        s_without_y = S_mask & ~(1 << y_idx)
        m = s_without_y
        while m:
            # get lowest set bit
            a_idx = (m & (-m)).bit_length() - 1
            a = a_idx + 1
            # Check if S intersects bad_b[y][a]
            if s_without_y & bad_b[y][a]:
                return False
            m &= m - 1  # clear lowest set bit
        return True
    
    # Greedy initialization with multiple random orderings
    start_time = time.time()
    deadline = start_time + 55.0
    random.seed(42)
    
    best_mask = 0
    best_size = 0
    
    def greedy_build(order):
        mask = 0
        size = 0
        for y in order:
            if can_add(mask, y):
                mask |= (1 << (y - 1))
                size += 1
        return mask, size
    
    # Try several greedy orderings
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = list(elements)
        if attempt == 0:
            pass  # natural order
        elif attempt == 1:
            order.reverse()
        elif attempt == 2:
            # Sort by number of divisors (heuristic: numbers with more divisors are more "connected")
            order.sort(key=lambda x: sum(1 for d in range(1, x+1) if x % d == 0), reverse=True)
        elif attempt == 3:
            # Sort by number of "bad triple partners"
            def num_partners(x):
                count = 0
                for a in range(1, N+1):
                    if a == x: continue
                    if bad_b[x][a]:
                        count += bad_b[x][a].bit_count()
                return count
            order.sort(key=num_partners, reverse=True)
        else:
            random.shuffle(order)
        
        mask, size = greedy_build(order)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Now do local search: try to add elements, and if stuck, do swaps
    # Represent current set as bitmask
    cur_mask = best_mask
    cur_size = best_size
    
    def popcount(x):
        return x.bit_count()
    
    while time.time() < deadline:
        # Try to add any element
        added = False
        remaining = [x for x in elements if not (cur_mask & (1 << (x-1)))]
        random.shuffle(remaining)
        
        for y in remaining:
            if can_add(cur_mask, y):
                cur_mask |= (1 << (y - 1))
                cur_size += 1
                added = True
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                break
        
        if not added:
            # Try swaps: remove one, add another
            swapped = False
            in_set = [x for x in elements if (cur_mask & (1 << (x-1)))]
            random.shuffle(in_set)
            
            for x in in_set:
                if time.time() > deadline:
                    break
                new_mask_without_x = cur_mask & ~(1 << (x - 1))
                candidates = [y for y in elements if not (cur_mask & (1 << (y-1)))]
                random.shuffle(candidates)
                
                for y in candidates:
                    if can_add(new_mask_without_x, y):
                        cur_mask = new_mask_without_x | (1 << (y - 1))
                        # cur_size stays same
                        swapped = True
                        break
                
                if swapped:
                    break
            
            if not swapped:
                # Kick: remove 2 random elements, refill greedily
                if cur_size >= 2:
                    in_set = [x for x in elements if (cur_mask & (1 << (x-1)))]
                    random.shuffle(in_set)
                    x1 = in_set[0]
                    x2 = in_set[1]
                    cur_mask &= ~(1 << (x1 - 1))
                    cur_mask &= ~(1 << (x2 - 1))
                    cur_size -= 2
                    
                    # Greedy refill
                    for y in elements:
                        if not (cur_mask & (1 << (y-1))) and can_add(cur_mask, y):
                            cur_mask |= (1 << (y - 1))
                            cur_size += 1
                    
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
    
    # Convert best_mask back to list
    result = [x for x in range(1, N + 1) if best_mask & (1 << (x - 1))]
    return result
