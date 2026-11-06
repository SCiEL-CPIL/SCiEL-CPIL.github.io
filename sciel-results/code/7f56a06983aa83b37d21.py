import math
import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0
    
    best_result = []
    
    # Run multiple randomized greedy attempts
    # Each attempt consists of a greedy pass and a local improvement pass
    num_restarts = 20
    
    for seed in range(num_restarts):
        if time.time() > limit_time:
            break
            
        random.seed(seed)
        
        # 1. Greedy Construction
        active = [True] * (N + 1)
        
        for p in PRIMES:
            if time.time() > limit_time:
                break
                
            # Count active points in each residue class mod p
            # We only care about primes p where it's possible that all classes are covered.
            # If the number of active points is less than p, it's impossible to cover all p classes.
            # So we can skip if count_active < p.
            
            # Fast check: count active points
            # This is O(N). We can maintain a count.
            # But N=5000 is small.
            
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            
            # Check if all classes are non-empty
            if all(c > 0 for c in counts):
                # Find min count
                min_c = min(counts)
                # Find all indices with min count
                min_indices = [i for i, c in enumerate(counts) if c == min_c]
                
                # Randomly pick one
                if len(min_indices) > 1:
                    chosen_idx = random.choice(min_indices)
                else:
                    chosen_idx = min_indices[0]
                
                # Deactivate points in this class
                for x in range(chosen_idx, N + 1, p):
                    active[x] = False
        
        # 2. Local Improvement (Add-back)
        # Current set T
        T = [x for x in range(N + 1) if active[x]]
        k = len(T)
        
        # Precompute empty classes for primes <= k
        # empty_classes[p] is a set of residues r such that no x in T has x % p == r
        empty_classes = {}
        for p in PRIMES:
            if p > k:
                break
            covered = set()
            for x in T:
                covered.add(x % p)
            empty_classes[p] = set(range(p)) - covered
        
        # Try to add points back
        # We iterate through all inactive points
        # To make it efficient, we can just loop through 0..N
        # We repeat this until no more points can be added
        
        improved = True
        while improved and time.time() < limit_time:
            improved = False
            for x in range(N + 1):
                if active[x]:
                    continue
                
                # Check if x can be added
                can_add = True
                
                # Check existing primes p <= k
                for p in PRIMES:
                    if p > k:
                        break
                    if p not in empty_classes:
                        continue
                    e_p = empty_classes[p]
                    if len(e_p) == 1:
                        # The only empty class is e
                        e = next(iter(e_p))
                        if x % p == e:
                            can_add = False
                            break
                    # If len(e_p) >= 2, adding x might fill one, but one remains empty. Safe.
                
                if not can_add:
                    continue
                
                # Check if k+1 is prime
                new_k = k + 1
                if new_k > 1 and all(new_k % i != 0 for i in range(2, int(new_k ** 0.5) + 1)):
                    # new_k is prime
                    # Check if T U {x} covers all residues mod new_k
                    # T has k elements. x is 1 element. Total k+1 elements.
                    # We need to check if there is an empty residue.
                    covered = set()
                    for y in T:
                        covered.add(y % new_k)
                    if x % new_k in covered:
                        # x doesn't add a new residue
                        # We need to check if covered is all residues
                        # But wait, if x % new_k is already in covered, then the set of residues is still 'covered'.
                        # If len(covered) == new_k, then it's full.
                        # If len(covered) < new_k, it's not full.
                        if len(covered) == new_k:
                            can_add = False
                    else:
                        # x adds a new residue
                        # The set of residues becomes covered U {x % new_k}
                        # Size is len(covered) + 1
                        # If len(covered) + 1 == new_k, then it's full.
                        if len(covered) + 1 == new_k:
                            can_add = False
                
                if can_add:
                    # Add x
                    active[x] = True
                    T.append(x)
                    k = new_k
                    improved = True
                    
                    # Update empty_classes for existing primes
                    for p in PRIMES:
                        if p > k:
                            break
                        if p in empty_classes:
                            r = x % p
                            if r in empty_classes[p]:
                                empty_classes[p].remove(r)
                    
                    # If k+1 is prime, add it to empty_classes for future checks?
                    # The loop "for p in PRIMES" in the next iteration will handle it if we re-calculate or update.
                    # But we only update existing ones.
                    # If new_k is prime, we should compute its empty classes.
                    if new_k > 1 and all(new_k % i != 0 for i in range(2, int(new_k ** 0.5) + 1)):
                        p = new_k
                        covered = set()
                        for y in T:
                            covered.add(y % p)
                        empty_classes[p] = set(range(p)) - covered

        # Update best result
        if len(T) > len(best_result):
            best_result = T
    
    return best_result
# EVOLVE-BLOCK-END
