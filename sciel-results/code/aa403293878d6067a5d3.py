import time
import random
import math

N = 5000
DEADLINE = 290.0  # Safety margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def _is_admissible(S, primes):
    """
    Check if S is admissible.
    """
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return False
    return True

def _get_smallest_violated_prime(S, primes):
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _greedy_remove(S, primes, rng):
    """
    Remove elements to make S admissible for all primes.
    """
    S = list(S)
    while True:
        p = _get_smallest_violated_prime(S, primes)
        if p is None:
            break
        freq = {}
        for x in S:
            r = x % p
            freq[r] = freq.get(r, 0) + 1
        
        min_freq = min(freq.values())
        candidates = [r for r, c in freq.items() if c == min_freq]
        r_to_remove = rng.choice(candidates)
        
        S = [x for x in S if x % p != r_to_remove]
        
    return S

def _greedy_add(S, candidates, primes, rng, time_limit=0.5):
    """
    Add elements from candidates to S if they maintain admissibility.
    """
    S = list(S)
    candidates = list(candidates)
    rng.shuffle(candidates)
    
    covered = {}
    for p in primes:
        if p > N: break
        covered[p] = set(x % p for x in S)
    
    added_any = True
    start_t = time.time()
    while added_any and candidates and (time.time() - start_t < time_limit):
        added_any = False
        new_candidates = []
        for x in candidates:
            new_n = len(S) + 1
            valid = True
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                if r not in covered[p]:
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            if valid:
                S.append(x)
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_candidates.append(x)
        candidates = new_candidates
        
    return S

def _local_search(S, primes, rng, time_limit=1.0):
    """
    Perform local search to improve S.
    Strategy: 1-out-2-in swaps.
    """
    S = list(S)
    S_set = set(S)
    all_elements = set(range(N + 1))
    complement = list(all_elements - S_set)
    
    start_t = time.time()
    improved = True
    while improved and (time.time() - start_t < time_limit):
        improved = False
        if not S: break
        
        # Try to add any element from complement (greedy add pass)
        # Use a very short time limit here to just clear easy adds
        S = _greedy_add(S, complement, primes, rng, time_limit=0.05)
        S_set = set(S)
        complement = list(all_elements - S_set)
        
        if not S: break
        
        # 1-out-2-in
        # Pick a random x from S
        x = rng.choice(S)
        S_no_x = S_set - {x}
        
        # Check if we can add 2 elements from complement to S_no_x
        # We need to check if S_no_x U {y, z} is admissible.
        
        # Compute covered residues for S_no_x
        covered_no_x = {}
        for p in primes:
            if p > N: break
            covered_no_x[p] = set(x2 % p for x2 in S_no_x)
        
        # Find elements in complement that can be added to S_no_x individually
        addable = []
        for y in complement:
            valid = True
            new_n = len(S_no_x) + 1
            for p in primes:
                if p > new_n:
                    break
                r = y % p
                if r not in covered_no_x[p]:
                    if len(covered_no_x[p]) == p - 1:
                        valid = False
                        break
            if valid:
                addable.append(y)
        
        if len(addable) >= 2:
            # Try to find two that can be added together
            found = False
            # Shuffle addable to randomize
            rng.shuffle(addable)
            for i in range(len(addable)):
                for j in range(i + 1, len(addable)):
                    y, z = addable[i], addable[j]
                    new_S = list(S_no_x) + [y, z]
                    if _is_admissible(new_S, primes):
                        S = new_S
                        S_set = set(S)
                        complement = list(all_elements - S_set)
                        improved = True
                        found = True
                        break
                if found:
                    break
            
            if found:
                continue
        
        # If no improvement found in this iteration, break to avoid stalling
        # In a more advanced version, we might try different x's, but for now, break.
        break
        
    return S

def construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Branching primes: We explicitly choose the missing residue for these
    # This explores different structural configurations
    branch_primes = [2, 3, 5, 7]
    
    while time.time() - start_time < DEADLINE - 5:
        rng = random.Random()
        
        # 1. Initialize S with all integers
        S = list(range(N + 1))
        
        # 2. Branching phase: Force missing residues for small primes
        for p in branch_primes:
            # Pick a random residue to remove
            r_to_remove = rng.choice(range(p))
            S = [x for x in S if x % p != r_to_remove]
            
        # 3. Greedy removal for remaining primes
        # S is now admissible for primes in branch_primes.
        # We need to ensure it is admissible for primes > 7 (up to len(S))
        S = _greedy_remove(S, PRIMES, rng)
        
        # 4. Greedy add back
        S_set = set(S)
        removed = [x for x in range(N + 1) if x not in S_set]
        
        S_final = _greedy_add(S, removed, PRIMES, rng, time_limit=0.5)
        
        # 5. Local search
        S_final = _local_search(S_final, PRIMES, rng, time_limit=1.0)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
    return best_S
