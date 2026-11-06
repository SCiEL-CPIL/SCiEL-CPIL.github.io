import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    k = 4
    all_ones = (1 << n) - 1
    
    # Precompute all k-subsets as bitmasks
    all_masks = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_masks.append(mask)
        
    num_masks = len(all_masks)
    
    # Precompute submasks of size 0,1,2,3 for all masks
    # submasks_by_size[mask][size] = list of submasks
    # This is memory heavy but fast access.
    # Alternatively, just use a function.
    # Let's try to optimize the function first.
    
    def popcount(x):
        return bin(x).count('1')
        
    # Precompute submasks for each mask and size?
    # Let's just use the recursive generator but optimized.
    # Or, since U is small, we can just iterate all 1820 masks and filter?
    # 1820 checks per pair.
    # 250 pairs per addition.
    # 250 additions.
    # 1820 * 250 * 250 = 113,750,000 checks.
    # Each check: (M_prime & J) == J and (M_prime & (A^J)) == 0 and (M_prime & (mask^J)) == 0.
    # This is 3 bitwise ops.
    # 113M * 3 = 339M ops.
    # In Python, this might take ~10-20 seconds per restart.
    # Too slow for 500 restarts.
    
    # So we MUST use the submask generation.
    # Let's precompute the list of submasks for each possible U and size.
    # U can be any mask.
    # We only need sizes 0, 1, 2, 3.
    
    submasks_cache = {}
    def get_submasks_cached(mask, size):
        key = (mask, size)
        if key not in submasks_cache:
            # Generate
            res = []
            bits = []
            m = mask
            while m:
                lsb = m & -m
                bits.append(lsb)
                m ^= lsb
                
            if len(bits) >= size:
                # Use itertools.combinations on bits
                from itertools import combinations as comb
                for c in comb(bits, size):
                    val = 0
                    for b in c:
                        val |= b
                    res.append(val)
            submasks_cache[key] = res
        return submasks_cache[key]
        
    def greedy(order):
        counts = [0] * (1 << n)
        family = []
        
        for mask in order:
            if counts[mask] > 0:
                continue
            
            family.append(mask)
            
            for A in family[:-1]:
                J = A & mask
                U = all_ones ^ (A | mask)
                need = k - popcount(J)
                
                submasks = get_submasks_cached(U, need)
                for Y in submasks:
                    M_prime = J | Y
                    counts[M_prime] += 1
                    
        return family
        
    best_family = []
    best_size = 0
    
    fam = greedy(all_masks)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam[:]
        
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        random.seed(12345 + restart)
        shuffled = all_masks[:]
        random.shuffle(shuffled)
        
        fam = greedy(shuffled)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam[:]
            
        if best_size >= 250:
            break
            
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
