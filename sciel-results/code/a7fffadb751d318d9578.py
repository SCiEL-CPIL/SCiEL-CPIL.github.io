"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import lru_cache

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
P_INDEX = {p: i for i, p in enumerate(PRIMES)}
NUM_DIMS = len(PRIMES)

# Build exponent vectors for all integers 2..40
def get_vector(n):
    v = [0] * NUM_DIMS
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

VEC = {n: get_vector(n) for n in range(2, 41)}
CANDIDATES = list(range(2, 41))

def subset_sums_distinct(vecs):
    """Check if all subset sums of the given exponent vectors are distinct."""
    if not vecs:
        return True
    sums = {tuple([0]*NUM_DIMS)}
    for v in vecs:
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(NUM_DIMS))
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Initial best: primes (always valid)
    primes = [a for a in CANDIDATES if all(a % d for d in range(2, int(a**0.5) + 1))]
    best_set = set(primes)
    best_size = len(best_set)
    
    if best_size >= 18:
        return [int(x) for x in sorted(best_set)]
    
    def try_add(S, cand):
        """Try to add cand to S, return True if valid."""
        if cand in S:
            return True
        S.add(cand)
        ok = subset_sums_distinct([VEC[x] for x in S])
        if not ok:
            S.discard(cand)
        return ok
    
    def greedy_fill(S):
        """Greedily add candidates that keep S valid."""
        added = True
        while added and time.time() < deadline:
            added = False
            remaining = [c for c in CANDIDATES if c not in S]
            random.shuffle(remaining)
            for c in remaining:
                if try_add(S, c):
                    added = True
                    break
        return S
    
    def local_search(S, max_iter=2000):
        """Local search: try swaps and adds to improve set size."""
        S = set(S)
        current_size = len(S)
        no_improve = 0
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if current_size >= 18:
                break
            
            improved = False
            
            # Try to add a random candidate
            remaining = [c for c in CANDIDATES if c not in S]
            if remaining:
                random.shuffle(remaining)
                for c in remaining[:10]:
                    if try_add(S, c):
                        current_size += 1
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    continue
            
            # Try swaps: remove one, try to add one
            if S:
                elems = list(S)
                random.shuffle(elems)
                for rem in elems[:5]:
                    S.discard(rem)
                    # Now try to add
                    remaining = [c for c in CANDIDATES if c not in S]
                    random.shuffle(remaining)
                    for add in remaining[:15]:
                        if try_add(S, add):
                            current_size = len(S)
                            improved = True
                            no_improve = 0
                            break
                    if not improved:
                        S.add(rem)
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
        
        return S, len(S)
    
    # Main search loop
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        if restart == 0:
            S = set(primes)
        elif restart % 10 == 0:
            S = set(best_set)
        else:
            # Perturb best_set
            S = set(best_set)
            # Remove 1-3 random elements
            for _ in range(random.randint(1, 3)):
                if S:
                    S.discard(random.choice(list(S)))
            # Add 1-3 random composites
            composites = [c for c in CANDIDATES if c not in primes]
            for _ in range(random.randint(1, 3)):
                if composites:
                    S.add(random.choice(composites))
            # Ensure validity by removing problematic elements
            if not subset_sums_distinct([VEC[x] for x in S]):
                # Repair: remove elements until valid
                while S and not subset_sums_distinct([VEC[x] for x in S]):
                    S.discard(random.choice(list(S)))
        
        # Greedy fill
        S = greedy_fill(S)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size >= 18:
                break
        
        # Local search
        S2, size2 = local_search(S)
        if size2 > best_size:
            best_set = S2
            best_size = size2
            if best_size >= 18:
                break
    
    # Final greedy fill on best_set
    best_set = greedy_fill(best_set)
    
    result = [int(x) for x in sorted(best_set)]
    return result

# EVOLVE-BLOCK-END
