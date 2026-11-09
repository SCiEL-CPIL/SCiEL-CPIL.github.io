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
    Remove elements to make S admissible.
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
    Candidates is a list of elements not in S.
    """
    S = list(S)
    candidates = list(candidates)
    rng.shuffle(candidates)
    
    # Precompute covered residues for efficiency
    # We only care about primes <= len(S) + max_adds
    # Let's just check on the fly or maintain a dict.
    # For speed, let's maintain a dict of sets for primes <= N.
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
    Strategy: 
    1. Identify "blocked" elements (elements that cannot be added because they would violate admissibility).
    2. Try to remove elements from S to unblock them.
    3. If removing an element allows adding more elements than removed, accept.
    """
    S = list(S)
    S_set = set(S)
    all_elements = set(range(N + 1))
    complement = list(all_elements - S_set)
    
    # Precompute covered residues for S
    covered = {}
    for p in primes:
        if p > N: break
        covered[p] = set(x % p for x in S)
    
    start_t = time.time()
    improved = True
    while improved and (time.time() - start_t < time_limit):
        improved = False
        if not S: break
        
        # Find elements in complement that are "blocked"
        # An element x is blocked if adding it to S would violate admissibility.
        # This happens if for some p <= len(S)+1, x covers a new residue r such that len(covered[p]) == p-1.
        
        # Instead of checking all, let's try to find a swap.
        # Pick a random x from S.
        # Check if we can replace x with two elements y, z from complement.
        # This is 1-out-2-in.
        
        # Simpler: 1-out-1-in swap? No, size must increase.
        # So 1-out-2-in or 0-out-1-in (greedy add).
        
        # Let's try to find a pair (y, z) in complement that can be added if we remove some x.
        # This is expensive.
        
        # Alternative: 
        # 1. Try to add any element from complement (greedy add pass).
        # 2. If stuck, try to remove a random x from S, then try to add elements from complement.
        #    If we can add more than 1 element, accept.
        
        # Step 1: Greedy add
        S = _greedy_add(S, complement, primes, rng, time_limit=0.1)
        S_set = set(S)
        complement = list(all_elements - S_set)
        
        if not S: break
        
        # Step 2: 1-out-2-in
        # Pick random x from S
        x = rng.choice(S)
        S_no_x = S_set - {x}
        
        # Check if we can add 2 elements from complement to S_no_x
        # We need to check if S_no_x U {y, z} is admissible.
        # This is equivalent to checking if y and z can be added to S_no_x.
        
        # Let's compute covered residues for S_no_x
        covered_no_x = {}
        for p in primes:
            if p > N: break
            covered_no_x[p] = set(x2 % p for x2 in S_no_x)
        
        # Find elements in complement that can be added to S_no_x
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
            # This is O(k^2) where k is len(addable)
            # Since k is likely small, this is okay.
            found = False
            for i in range(len(addable)):
                for j in range(i + 1, len(addable)):
                    y, z = addable[i], addable[j]
                    # Check if S_no_x U {y, z} is admissible
                    # We can just check admissibility of the new set
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
        
        # If no improvement, break to avoid infinite loop if no swaps possible
        # But we might have just failed this iteration. 
        # To prevent stalling, we can shuffle or try different x.
        # For now, if no swap found, we break the local search loop for this restart.
        # This is a simple local search.
        break
        
    return S

def construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Precompute residues for all numbers for all relevant primes?
    # This might speed up _greedy_add and local search.
    # Let's try to optimize _greedy_add first.
    
    while time.time() - start_time < DEADLINE - 5:
        rng = random.Random()
        
        # Strategy 1: Full set start
        if best_size == 0 or rng.random() < 0.5:
            S_init = list(range(N + 1))
        else:
            # Strategy 2: Random subset start
            # Density between 0.5 and 0.9
            density = rng.uniform(0.5, 0.9)
            target_size = int(N * density)
            S_init = rng.sample(range(N + 1), target_size)
            
        S_greedy = _greedy_remove(S_init, PRIMES, rng)
        
        S_set = set(S_greedy)
        removed = [x for x in range(N + 1) if x not in S_set]
        
        S_final = _greedy_add(S_greedy, removed, PRIMES, rng, time_limit=1.0)
        
        # Local search
        S_final = _local_search(S_final, PRIMES, rng, time_limit=2.0)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
    return best_S
