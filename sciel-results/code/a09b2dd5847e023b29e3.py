import math
import time
import random
import bisect

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
# Map prime to index for fast lookup if needed, but we mostly iterate
# We need primes up to N (5000)
# For a tuple of size K, we need primes p <= K.

def construct():
    start_time = time.time()
    limit_time = start_time + 280.0 # Safety margin
    
    best_T = []
    best_size = 0
    
    # We will run multiple restarts
    # Each restart: Greedy Init + Local Search
    # Number of restarts depends on time.
    # Local search is fast, so we can do many.
    
    while time.time() < limit_time:
        # 1. Greedy Initialization
        active = [True] * (N + 1)
        
        # Process primes in increasing order
        # For each p, if all residues present, remove one class
        # To be smarter, remove the class with minimum count
        for p in PRIMES:
            if time.time() > limit_time: break
            if p > N: break
            
            # Count residues
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            
            # Check if any residue is missing
            min_count = min(counts)
            if min_count > 0:
                # All residues present, must remove one
                # Find residue with min count
                min_r = counts.index(min_count)
                # Remove all x such that x % p == min_r
                for x in range(min_r, N + 1, p):
                    active[x] = False
        
        T = [x for x in range(N + 1) if active[x]]
        
        # 2. Local Search
        # Data structures
        # counts[i] for primes[i]
        # We only need primes up to max possible size (5000)
        # But dynamically, we only care about p <= len(T)
        
        # Precompute counts for current T
        # This is O(|T| * pi(|T|))
        # |T| ~ 2000, pi(2000) ~ 300. 600k ops. Fast.
        
        # We will maintain counts for ALL primes up to N?
        # No, just up to N is fine. 669 primes.
        # counts is a list of lists.
        # counts[i] has length PRIMES[i]
        
        counts = []
        missing_counts = []
        for p in PRIMES:
            c = [0] * p
            for x in T:
                c[x % p] += 1
            counts.append(c)
            missing_counts.append(sum(1 for val in c if val == 0))
        
        in_T = [False] * (N + 1)
        for x in T:
            in_T[x] = True
            
        # Local Search Loop
        # We alternate between Add and Remove
        # Add: pick random x not in T, check validity
        # Remove: pick random x in T, remove it
        
        # To check validity of adding x:
        # For all p <= len(T) + 1:
        #   r = x % p
        #   if counts[i][r] == 0 and missing_counts[i] == 1:
        #       invalid
        
        # We need index of prime p.
        # Since we iterate p in PRIMES, we can just loop i while PRIMES[i] <= len(T)+1
        
        stall_counter = 0
        max_stall = 1000
        
        while time.time() < limit_time:
            current_size = len(T)
            
            # Find max prime index to check
            # bisect_right(PRIMES, current_size + 1)
            end_idx = bisect.bisect_right(PRIMES, current_size + 1)
            
            # Try Add
            x = random.randint(0, N)
            if not in_T[x]:
                valid = True
                # Check constraints
                for i in range(end_idx):
                    p = PRIMES[i]
                    r = x % p
                    if counts[i][r] == 0:
                        if missing_counts[i] == 1:
                            valid = False
                            break
                
                if valid:
                    # Commit Add
                    for i in range(end_idx):
                        p = PRIMES[i]
                        r = x % p
                        if counts[i][r] == 0:
                            missing_counts[i] -= 1
                        counts[i][r] += 1
                    
                    T.append(x)
                    in_T[x] = True
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1
                
            # If stalled, try Remove
            if stall_counter > max_stall:
                # Remove a random element
                # To be effective, maybe remove one that is "least useful"?
                # For now, random.
                if T:
                    idx = random.randint(0, len(T) - 1)
                    y = T[idx]
                    # Remove y
                    in_T[y] = False
                    T.pop(idx)
                    
                    # Update counts
                    new_size = len(T)
                    new_end_idx = bisect.bisect_right(PRIMES, new_size + 1)
                    # We might have primes that are no longer relevant?
                    # But counts are maintained for all primes up to N.
                    # We just need to update counts for primes p <= new_size + 1?
                    # Actually, removing y decreases count for y%p.
                    # If count becomes 0, missing_count increases.
                    
                    for i in range(len(PRIMES)):
                         # Optimization: only update if PRIMES[i] <= N?
                         # We need to keep counts consistent for future adds.
                         # When we add, we check p <= size+1.
                         # So we must maintain counts for all p that *could* be checked.
                         # Max size is N, so max p is N.
                         # So we must update all primes up to N.
                         # This is slow: 669 primes per removal.
                         # 669 * 5000 removals = 3.3M ops. OK.
                         
                         p = PRIMES[i]
                         r = y % p
                         counts[i][r] -= 1
                         if counts[i][r] == 0:
                             missing_counts[i] += 1
                             
                stall_counter = 0
                max_stall = int(max_stall * 1.5) # Increase threshold to avoid too many removals
                if max_stall > 10000: max_stall = 10000

            # Update best
            if len(T) > best_size:
                best_size = len(T)
                best_T = T.copy()
                
        # End of one restart
        # If we have time, loop again with new random seed?
        # The random seed is global, so it continues.
        # But we might want to reset the greedy init?
        # Greedy init is deterministic.
        # So all restarts start from the same T?
        # That's bad.
        
        # Improvement: Randomize the greedy init.
        # When removing a residue class, pick a random one among those with min count?
        # Or just pick a random residue class?
        # Let's modify greedy init to be randomized.
        
    return best_T
