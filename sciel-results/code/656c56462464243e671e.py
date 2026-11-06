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
        # 1. Greedy Initialization (randomized)
        active = [True] * (N + 1)
        
        # Process primes in increasing order
        # For each p, if all residues present, remove one class
        # Randomize: pick randomly among min-count residues
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
                # Find all residues with min count
                min_residues = [r for r in range(p) if counts[r] == min_count]
                # Pick one randomly
                min_r = random.choice(min_residues)
                # Remove all x such that x % p == min_r
                for x in range(min_r, N + 1, p):
                    active[x] = False
        
        T = [x for x in range(N + 1) if active[x]]
        
        # 2. Local Search
        # Data structures
        # counts[i] for primes[i]
        
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
            
        # Local Search Loop with batch adds
        stall_counter = 0
        max_stall = 200
        batch_size = 20
        
        while time.time() < limit_time:
            current_size = len(T)
            
            # Find max prime index to check
            end_idx = bisect.bisect_right(PRIMES, current_size + 1)
            
            # Try batch of adds
            added = 0
            for _ in range(batch_size):
                if time.time() > limit_time: break
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
                        added += 1
                        # Update end_idx since size changed
                        end_idx = bisect.bisect_right(PRIMES, len(T) + 1)
                    else:
                        stall_counter += 1
                else:
                    stall_counter += 1
            
            if added > 0:
                stall_counter = 0
            
            # If stalled, try Remove
            if stall_counter > max_stall:
                # Remove an element that blocks the most adds
                # Heuristic: remove element whose removal creates the most new zero residues
                if T:
                    # Sample a few candidates and pick the best
                    best_y = None
                    best_gain = -1
                    sample_size = min(10, len(T))
                    for _ in range(sample_size):
                        idx = random.randint(0, len(T) - 1)
                        y = T[idx]
                        # Estimate gain: how many primes would get a new zero residue
                        gain = 0
                        for i in range(end_idx):
                            p = PRIMES[i]
                            r = y % p
                            if counts[i][r] == 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_y = idx
                    
                    if best_y is not None:
                        y = T[best_y]
                        in_T[y] = False
                        T.pop(best_y)
                        
                        # Update counts
                        new_end_idx = bisect.bisect_right(PRIMES, len(T) + 1)
                        for i in range(len(PRIMES)):
                            p = PRIMES[i]
                            r = y % p
                            counts[i][r] -= 1
                            if counts[i][r] == 0:
                                missing_counts[i] += 1
                        
                        end_idx = new_end_idx
                
                stall_counter = 0
                max_stall = int(max_stall * 1.3)
                if max_stall > 5000: max_stall = 5000

            # Update best
            if len(T) > best_size:
                best_size = len(T)
                best_T = T.copy()
                
    return best_T
