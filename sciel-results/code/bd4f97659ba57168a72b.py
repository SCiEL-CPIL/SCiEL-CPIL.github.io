import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    # We only need to cover up to some limit. The task caps at 10^5, but 
    # with sum(1/p)<=1, L is likely much smaller. 
    # Let's estimate a reasonable limit for the bitset. 
    # If we cover 1..L, and L is around 1000-5000, a bitset of size 10000 is cheap.
    # Let's use a limit of 20000 to be safe, but we can stop early if we find a gap.
    MAX_L = 20000
    
    # Precompute bitsets for each prime and residue? 
    # That would be sum(p) bits ~ 76127 * 16 bytes ~ 1.2 MB. 
    # But we can generate them on the fly or precompute masks for each p.
    # Actually, for a given p, the mask for residue r is a periodic pattern.
    
    # Let's precompute the mask for each prime p for each residue r.
    # mask[p][r] is an integer where bit i is 1 if i % p == r and 1 <= i <= MAX_L.
    # This might be memory heavy if we store all, but 76127 primes * avg 500 residues? No.
    # There are 168 primes <= 1000.
    # Sum of p for p<=1000 is ~ 76127.
    # So storing 76127 integers of ~20000 bits (2500 bytes) is ~ 190 MB. This is too much.
    
    # Alternative: Generate mask on the fly.
    # To generate mask for p, r:
    #   bits = 0
    #   for i in range(r if r>0 else p, MAX_L+1, p):
    #       bits |= (1 << i)
    # This is O(MAX_L/p). For small p, this is fast. For large p, it's very fast.
    # Total cost to generate all masks for a given p is O(MAX_L * sum(1/p)) ~ O(MAX_L * log log 1000).
    
    # Let's precompute the list of masks for each prime, but only store them if we decide to use that prime?
    # Or just compute the "best" residue for a given p against a current uncovered set U.
    
    def get_mask(p, r):
        mask = 0
        # i must be in [1, MAX_L] and i % p == r
        # If r == 0, i is multiple of p. First is p.
        # If r > 0, first is r (if r <= MAX_L).
        start = r
        if start == 0:
            start = p
        if start > MAX_L:
            return 0
        for i in range(start, MAX_L + 1, p):
            mask |= (1 << i)
        return mask

    def greedy_cover():
        # Returns (config, sum_fraction, covered_bitset)
        covered = 0
        budget = Fraction(1, 1)
        used_primes = set()
        config = []
        
        # Pre-compute masks for all primes? 
        # Let's compute them on demand and cache.
        masks_cache = {}
        
        def get_masks_for_prime(p):
            if p not in masks_cache:
                masks = []
                for r in range(p):
                    masks.append(get_mask(p, r))
                masks_cache[p] = masks
            return masks_cache[p]
            
        # We want to cover 1..MAX_L
        # Initial uncovered set is all 1..MAX_L
        # But we don't need to represent the full set if we just track covered.
        
        # To evaluate gain, we need to know how many NEW bits are set.
        # new_bits = mask[p][r] & ~covered
        # gain = popcount(new_bits)
        
        # We iterate until no prime can improve coverage or budget is exhausted.
        
        while True:
            best_gain = 0
            best_p = None
            best_r = None
            best_mask = 0
            
            # Iterate through all available primes
            for p in primes:
                if p in used_primes:
                    continue
                
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                masks = get_masks_for_prime(p)
                # Find residue r that maximizes gain
                # Optimization: We can break early if max possible gain (p) is <= best_gain? No, p varies.
                # But we can check if the total number of bits in mask is <= best_gain.
                
                for r in range(p):
                    m = masks[r]
                    # If mask is 0, skip
                    if m == 0:
                        continue
                    new_bits = m & ~covered
                    # Count bits
                    gain = new_bits.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                        best_r = r
                        best_mask = m
                        
            if best_p is None or best_gain == 0:
                break
                
            # Add the best prime
            used_primes.add(best_p)
            config.append([best_p, best_r])
            budget -= Fraction(1, best_p)
            covered |= best_mask
            
        return config, budget, covered

    def local_search(config, covered, budget, time_limit):
        # Optimize residues and potentially swap primes
        # We will try to improve coverage by changing residues or swapping primes.
        
        # Precompute masks for primes in config and nearby primes
        # This is expensive if done naively.
        
        # Let's focus on residue optimization first.
        # For each prime in config, try all residues to see if it improves coverage.
        
        # We need to be able to quickly calculate the coverage if we change a residue.
        # new_covered = (covered & ~old_mask) | new_mask
        # gain = popcount(new_covered) - popcount(covered)
        
        # Since we only care about improving L, we can just check if new_covered has more bits.
        
        # To make this fast, we should cache masks for the primes we are using.
        masks_cache = {}
        
        def get_masks_for_prime(p):
            if p not in masks_cache:
                masks = []
                for r in range(p):
                    masks.append(get_mask(p, r))
                masks_cache[p] = masks
            return masks_cache[p]
            
        # Initialize cache for current config
        for p, r in config:
            get_masks_for_prime(p)
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle config to avoid bias
            random.shuffle(config)
            
            for i in range(len(config)):
                p, r = config[i]
                masks = get_masks_for_prime(p)
                old_mask = masks[r]
                
                # Try all other residues
                best_r = r
                best_mask = old_mask
                current_cover_count = covered.bit_count()
                
                # We can prune: if the mask has fewer bits than current best, skip?
                # No, it depends on overlap.
                
                for new_r in range(p):
                    if new_r == r:
                        continue
                    new_mask = masks[new_r]
                    # Calculate new covered set
                    # We don't want to create a huge new integer every time if we can avoid it?
                    # But bit operations on Python ints are fast.
                    new_covered = (covered & ~old_mask) | new_mask
                    new_count = new_covered.bit_count()
                    
                    if new_count > current_cover_count:
                        current_cover_count = new_count
                        best_r = new_r
                        best_mask = new_mask
                        
                if best_r != r:
                    config[i][1] = best_r
                    covered = (covered & ~old_mask) | best_mask
                    improved = True
                    
        # After residue optimization, try swapping primes?
        # This is more complex. We can try to replace a prime p with a prime p' not in config
        # if p' allows for a better residue choice or covers more.
        # Given time constraints, residue optimization is the biggest win.
        
        return config, covered, budget

    # Run Greedy
    config, budget, covered = greedy_cover()
    
    # Run Local Search
    time_left = deadline - time.time()
    if time_left > 5:
        config, covered, budget = local_search(config, covered, budget, deadline - 5)
        
    # Calculate final L
    # We need the largest L such that 1..L are all covered.
    # We can iterate from 1 upwards until we find a gap.
    # Or use bit manipulation to find the first 0 bit after bit 0?
    # Bit 0 is unused (we start at 1).
    # We want the first i >= 1 such that bit i is 0.
    
    # Trick: ~covered has 1s where covered has 0s.
    # We want the lowest set bit in ~covered that is >= 1.
    # Let's just loop, it's fast for L ~ 1000.
    
    L = 0
    # Cap at MAX_L
    for i in range(1, MAX_L + 1):
        if (covered >> i) & 1:
            L = i
        else:
            break
            
    # Ensure we don't return a config that violates budget (shouldn't happen, but check)
    # The greedy ensures budget is respected.
    
    return config

if __name__ == "__main__":
    # Simple test
    res = construct()
    # Calculate L for debugging
    covered = 0
    MAX_L = 20000
    for p, r in res:
        mask = 0
        start = r if r > 0 else p
        for i in range(start, MAX_L + 1, p):
            mask |= (1 << i)
        covered |= mask
        
    L = 0
    for i in range(1, MAX_L + 1):
        if (covered >> i) & 1:
            L = i
        else:
            break
    print(f"Covered up to L = {L}")
    s = sum(Fraction(1, p) for p, r in res)
    print(f"Sum 1/p = {s}")
