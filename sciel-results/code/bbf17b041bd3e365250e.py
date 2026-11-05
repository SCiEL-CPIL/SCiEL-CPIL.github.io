import random
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute masks
_primes_list = _primes(N)
_num_primes = len(_primes_list)
# bad_masks[i][r] is a bitmask for prime _primes_list[i], residue r
# We store as list of lists of ints
_bad_masks = []
for p in _primes_list:
    masks = []
    for r in range(p):
        mask = 0
        # Set bits for n in [0, N] such that n % p == r
        start = r
        if start > N:
            continue
        for n in range(start, N + 1, p):
            mask |= (1 << n)
        masks.append(mask)
    _bad_masks.append(masks)

_all_mask = (1 << (N + 1)) - 1

def construct():
    best_score = 0
    best_r = None
    
    # Use multiple restarts
    num_restarts = 10
    steps_per_restart = 3000
    
    for _ in range(num_restarts):
        # Random initialization
        r = [random.randint(0, p - 1) for p in _primes_list]
        
        # Compute initial removed mask
        removed = 0
        for i in range(_num_primes):
            removed |= _bad_masks[i][r[i]]
            
        current_score = (_all_mask & ~removed).bit_count()
        
        if current_score > best_score:
            best_score = current_score
            best_r = r[:]
            
        T = 1.0
        for step in range(steps_per_restart):
            # Pick a random prime index
            i = random.randint(0, _num_primes - 1)
            p = _primes_list[i]
            
            # Pick a random new residue
            old_r = r[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Calculate new removed mask
            # others = OR of all masks except i
            others = 0
            for j in range(_num_primes):
                if j != i:
                    others |= _bad_masks[j][r[j]]
            
            new_removed = others | _bad_masks[i][new_r]
            new_score = (_all_mask & ~new_removed).bit_count()
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                # Accept
                r[i] = new_r
                removed = new_removed
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_r = r[:]
            
            T *= 0.9995
            
    # Reconstruct S from best_r
    removed = 0
    for i in range(_num_primes):
        removed |= _bad_masks[i][best_r[i]]
        
    S_mask = _all_mask & ~removed
    S = []
    n = 0
    while S_mask:
        if S_mask & 1:
            S.append(n)
        S_mask >>= 1
        n += 1
        
    return S
# EVOLVE-BLOCK-END
