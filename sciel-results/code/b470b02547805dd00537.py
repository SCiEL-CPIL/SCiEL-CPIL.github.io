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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0
    
    # 1. Greedy Initialization
    # Start with all numbers
    active = [True] * (N + 1)
    primes = _primes(N)
    
    # Greedy removal
    # We process primes in increasing order.
    # For each p, if all residues are present, remove the smallest residue class.
    # To do this efficiently, we can just iterate and remove.
    # However, the counts change as we remove.
    # A simpler greedy:
    # For p in primes:
    #   counts = [0]*p
    #   for x in range(N+1):
    #     if active[x]: counts[x%p] += 1
    #   if min(counts) > 0:
    #      min_r = counts.index(min(counts))
    #      for x in range(min_r, N+1, p):
    #         active[x] = False
    
    # This is O(N * pi(N) * avg_p) which might be slow in Python.
    # N=5000, pi(N)=669.
    # Sum of p is ~ 2.5M.
    # Total ops ~ 5000 * 669? No.
    # For each p, we scan N. 669 * 5000 = 3.3M ops. Very fast.
    
    for p in primes:
        if time.time() > limit_time: break
        counts = [0] * p
        for x in range(N + 1):
            if active[x]:
                counts[x % p] += 1
        
        if min(counts) > 0:
            min_val = min(counts)
            min_idx = counts.index(min_val)
            for x in range(min_idx, N + 1, p):
                active[x] = False
                
    T = [x for x in range(N + 1) if active[x]]
    
    # 2. Local Search
    # Data structures for fast validity check
    # counts[p_idx][r]
    # We only care about primes p <= len(T)
    # Let's precompute mods
    # mods[x][p_idx] = x % primes[p_idx]
    
    # To save memory and time, we can compute on the fly or precompute for all x, p.
    # 5000 * 669 integers is ~ 3.3M integers. ~ 25MB. OK.
    # But we only need primes up to current max size.
    
    # Let's maintain:
    # prime_indices: list of primes
    # counts: list of lists. counts[i] is counts for primes[i]
    # zero_counts[i]: number of residues with count 0 for primes[i]
    
    max_prime_idx = len(primes)
    # Initialize counts for the current T
    # We need to map prime value to index? No, just use list of primes.
    
    # Filter primes relevant to max possible size (5000)
    # All primes up to 5000 are relevant.
    
    counts = []
    zero_counts = []
    for p in primes:
        c = [0] * p
        zc = 0
        for x in T:
            r = x % p
            c[r] += 1
        # Count zeros
        for val in c:
            if val == 0:
                zc += 1
        counts.append(c)
        zero_counts.append(zc)
        
    # Helper to check if T is admissible for size k
    # Admissible if for all p <= k, zero_counts[idx(p)] > 0
    
    # We need a way to quickly find index of prime p.
    # prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Add Move
    # Pick x not in T
    # Check if T U {x} is admissible.
    # New size k' = k + 1.
    # Must check primes p <= k'.
    # For each such p, x adds to class x%p.
    # If counts[idx][x%p] was 0, zero_counts[idx] decreases.
    # If zero_counts[idx] becomes 0, invalid.
    
    # We will simulate the add, and if valid, commit.
    
    # To allow escaping local minima, we can also remove.
    # But adding is the goal.
    
    # Random state
    random.seed(42)
    
    best_size = len(T)
    best_T = T.copy()
    
    # Active set for O(1) lookup
    in_T = [False] * (N + 1)
    for x in T:
        in_T[x] = True
        
    # Candidates for addition: not in T
    # Candidates for removal: in T
    
    # Precompute mods for all x and all primes?
    # This might be too slow/memory intensive to precompute fully.
    # Let's compute on the fly.
    
    # Optimization:
    # Only primes p <= current_size + 1 need to be checked.
    # Let max_p_idx = index of largest prime <= current_size + 1.
    
    # We will iterate.
    # To speed up, we can maintain a list of "active" primes indices.
    
    # Let's just loop through all primes up to 5000?
    # No, only those <= size.
    
    # We can find the cutoff index using bisect.
    import bisect
    
    while time.time() < limit_time:
        current_size = len(T)
        if current_size > best_size:
            best_size = current_size
            best_T = T.copy()
            
        # Find max prime index to check
        # Primes are sorted.
        # We need primes <= current_size + 1
        max_p_to_check = current_size + 1
        # Find index
        # bisect_right(primes, max_p_to_check) - 1
        end_idx = bisect.bisect_right(primes, max_p_to_check)
        
        # Try to add a random element
        # If we fail many times, try removing one to reshuffle?
        # Let's just try adding.
        
        # Pick random x
        # We can keep a list of not_in_T?
        # Or just pick random x in [0, N] and check if in_T[x]
        
        x = random.randint(0, N)
        if in_T[x]:
            continue
            
        # Check validity of adding x
        valid = True
        # Store changes to revert if invalid
        # changes: list of (p_idx, r, old_count, old_zero)
        changes = []
        
        for i in range(end_idx):
            p = primes[i]
            r = x % p
            old_count = counts[i][r]
            old_zero = zero_counts[i]
            
            new_count = old_count + 1
            new_zero = old_zero
            if old_count == 0:
                new_zero -= 1
            
            if new_zero == 0:
                # Invalid
                valid = False
                # Revert previous changes in this loop
                # We haven't committed yet, so we just need to revert what we changed in memory?
                # We are checking, so we can just break and not apply.
                # But we need to be careful if we applied some.
                # Better: Check first, then apply.
                # But checking requires knowing the state.
                # Since we are just reading `counts` and `zero_counts`, we can simulate.
                break
            
            # Apply tentatively?
            # If we apply and then find invalid, we must revert.
            # It is safer to collect changes and apply if all pass.
            # But we need to know if a *later* prime fails.
            # So we simulate the full path.
            
            # Wait, if I break, I haven't applied any changes to `counts` or `zero_counts` yet.
            # So I can just `continue` the outer loop (next x).
            
            # However, I need to apply the changes if valid.
            # So I should store the changes.
            
            changes.append((i, r, old_count, old_zero))
            
        if valid:
            # Commit changes
            for i, r, old_count, old_zero in changes:
                counts[i][r] += 1
                if old_count == 0:
                    zero_counts[i] -= 1
            
            T.append(x)
            in_T[x] = True
        else:
            # Do nothing
            pass
            
        # Optional: Remove move to escape local minima
        # If we are stuck (no additions for a while), remove a random element.
        # This increases the "zero_counts" for some primes, potentially allowing other additions.
        # We can track "stall" count.
        
    return best_T
# EVOLVE-BLOCK-END
