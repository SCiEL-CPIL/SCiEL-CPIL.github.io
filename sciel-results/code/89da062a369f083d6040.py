import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (150s kill)
    
    primes = _sieve(1000)
    N = 100000
    
    # Precompute bitset masks for each (p, r)
    # mask[p][r] is an integer where bit i is set if i % p == r and 1 <= i <= N
    masks = {}
    for p in primes:
        masks[p] = []
        for r in range(p):
            m = 0
            # Start from the smallest positive integer congruent to r mod p
            # If r == 0, smallest is p. If r > 0, smallest is r.
            start = r if r > 0 else p
            # We need to set bits for start, start+p, start+2p, ... <= N
            # Setting bits one by one is too slow for small p.
            # We can use a pattern.
            # But for simplicity and speed in precomp (which is one-time), 
            # we can generate the indices and set bits.
            # Actually, for p=2, there are 50000 bits to set. 
            # For p=2, r=0: 2, 4, 6...
            # For p=2, r=1: 1, 3, 5...
            
            # A faster way:
            # Create a pattern of length p, then repeat it.
            # But Python integers don't support easy slicing/pattern filling.
            # Let's stick to the loop but optimize it.
            # Actually, 1000 primes * avg 100 residues * 1000 steps = 10^8 operations? 
            # That might be too slow in Python.
            
            # Alternative: Use numpy to generate the mask?
            # No, we need Python integers for fast OR/XOR.
            
            # Let's estimate time.
            # Sum of p for p <= 1000 is approx 1000^2 / (2 ln 1000) ~ 72000.
            # For each residue, we iterate N/p times.
            # Total iterations = Sum_p (p * N/p) = N * NumPrimes = 100000 * 168 = 16.8 million.
            # 16.8 million bit sets in Python might take ~1-2 seconds. Acceptable.
            
            for i in range(start, N + 1, p):
                m |= (1 << i)
            masks[p].append(m)
        
    # Helper to calculate L given a set of (p, r) pairs using bitsets
    def get_L(pairs):
        covered = 0
        for p, r in pairs:
            covered |= masks[p][r]
        
        # Find the first 0 bit in 1..N
        # We want the largest L such that bits 1..L are all 1.
        # If covered has a 0 at position k, then L = k-1.
        # If all bits 1..N are 1, L = N.
        
        # Check if all bits 1..N are set
        # full_mask = (1 << (N+1)) - 2  # Bits 1..N set
        # if (covered & full_mask) == full_mask: return N
        
        # Find lowest zero bit >= 1
        # ~covered has 1s where covered has 0s.
        # We want the lowest set bit in ~covered that is >= 1.
        # Actually, we want the first missing integer.
        
        # Let's construct a mask for 1..N
        # full = (1 << (N+1)) - 2
        # missing = full & ~covered
        # If missing == 0, return N
        # Else, find lowest set bit in missing.
        
        # Creating (1 << 100001) is fine.
        # But doing it every call is expensive? 
        # 100000 bit integer operations are fast in C (Python big ints).
        
        full_mask = (1 << (N + 1)) - 2
        missing = full_mask & ~covered
        if missing == 0:
            return N
        
        # Find lowest set bit
        # (missing & -missing).bit_length() - 1 gives the index of the lowest set bit.
        # Let's verify: missing = 0b1000 (8). -missing = ...11111000. 
        # missing & -missing = 0b1000. bit_length = 4. index = 3. Correct.
        # If missing = 0b0001 (1). -missing = ...11111111. 
        # missing & -missing = 1. bit_length = 1. index = 0.
        # But we start from 1. If bit 0 is set, it doesn't matter.
        # Our full_mask has bit 0 as 0. So missing bit 0 is 0.
        
        lowest_zero_bit = (missing & -missing).bit_length() - 1
        return lowest_zero_bit - 1 if lowest_zero_bit > 0 else 0

    # Strategy:
    # 1. Select a set of primes.
    #    We want sum(1/p) <= 1.
    #    To maximize L, we generally want small primes, but they are expensive.
    #    Let's try a few different prime sets.
    
    def select_primes(min_p=2):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel, s

    # Try multiple prime sets
    prime_sets = [
        select_primes(2)[0],
        select_primes(3)[0],
        select_primes(5)[0],
        select_primes(7)[0],
        select_primes(11)[0],
        select_primes(13)[0],
    ]
    
    unique_sets = []
    seen = set()
    for ps in prime_sets:
        key = tuple(ps)
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)
    
    best_L = 0
    best_pairs = []
    
    for ps in unique_sets:
        if not ps:
            continue
        
        # Greedy Initialization
        # Start with empty coverage
        # Repeatedly pick the (p, r) that adds the most new coverage in the prefix [1, current_L + 1]
        # Actually, we want to extend L.
        
        # Let's try a few random initializations and local search
        # Local search with bitsets is much faster.
        
        inits = [
            {p: 1 for p in ps},
            {p: 0 for p in ps},
            {p: random.randint(0, p-1) for p in ps},
            {p: random.randint(0, p-1) for p in ps},
        ]
        
        for init in inits:
            current_pairs = [(p, init[p]) for p in ps]
            current_L = get_L(current_pairs)
            
            if current_L > best_L:
                best_L = current_L
                best_pairs = current_pairs.copy()
            
            # Local Search
            T = 1.0
            cooling = 0.999
            no_improve = 0
            max_no_improve = 5000 # Restart if stuck
            
            # Precompute indices for pairs for faster access?
            # current_pairs is a list of (p, r).
            
            while time.time() < deadline - 2:
                # Pick a random prime to change
                idx = random.randint(0, len(ps) - 1)
                p = ps[idx]
                old_r = init[p]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                init[p] = new_r
                # Update pair
                current_pairs[idx] = (p, new_r)
                
                new_L = get_L(current_pairs)
                
                if new_L > current_L:
                    current_L = new_L
                    no_improve = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_pairs = current_pairs.copy()
                else:
                    delta = new_L - current_L
                    if delta < 0:
                        # Accept with probability exp(delta / T)
                        # delta is negative.
                        if random.random() < np.exp(delta / T):
                            current_L = new_L
                        else:
                            # Revert
                            init[p] = old_r
                            current_pairs[idx] = (p, old_r)
                    # If delta == 0, accept?
                    # Usually yes, to wander.
                    
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break # Restart this prime set with new random init? Or just break SA.
                
                T *= cooling
                if T < 0.01:
                    T = 0.01
            
            # If we broke due to no_improve, we might try another init for the same prime set
            # But for now, we just move to the next init.

    # Convert best_pairs to required format [[p, r], ...]
    return best_pairs

# EVOLVE-BLOCK-END
