import time
import math

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

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # present[x] is True if x is in the set
    present = [True] * (N + 1)
    current_size = N + 1
    
    # Initialize counts and min info for all primes
    # counts[p][c] = number of elements in S congruent to c mod p
    # We only need this for primes that are currently "bad"
    # A prime is bad if min(counts[p]) > 0
    
    # Precompute indices for each prime to speed up candidate selection?
    # For a prime p, elements in class c are c, c+p, c+2p...
    
    # State for each prime:
    # min_count[p]
    # min_class[p]
    # is_bad[p]
    
    min_count = {}
    min_class = {}
    is_bad = {}
    counts = {} # p -> list of counts for residues 0..p-1
    
    for p in PRIMES:
        counts[p] = [0] * p
        for x in range(N + 1):
            counts[p][x % p] += 1
        
        m = min(counts[p])
        if m > 0:
            is_bad[p] = True
            min_count[p] = m
            # Find the class with min count
            # If multiple, pick the first one? Or the one with most "overlap"?
            # For init, just pick first.
            min_class[p] = counts[p].index(m)
        else:
            is_bad[p] = False
            min_count[p] = 0
            min_class[p] = 0
            
    bad_primes = [p for p in PRIMES if is_bad[p]]
    
    while bad_primes:
        if time.time() > deadline:
            break
            
        k = current_size
        # Filter bad primes to p <= k
        active_bad = [p for p in bad_primes if p <= k]
        if not active_bad:
            break
            
        # Pick the prime with the smallest min_count
        # Tie-breaker: larger p? (Cheaper to fix?)
        # Smaller min_count is the primary goal.
        p_star = min(active_bad, key=lambda p: (min_count[p], -p))
        
        c_star = min_class[p_star]
        
        # Find candidates in S with x % p_star == c_star
        # And pick the one with highest "benefit"
        # Benefit: number of other active bad primes q where x % q == min_class[q]
        
        best_x = -1
        best_score = -1
        
        # Iterate through elements in class c_star mod p_star
        # x = c_star + j * p_star
        # Check if present[x]
        
        # Optimization: If p_star is small, there are many candidates.
        # If p_star is large, few candidates.
        
        for x in range(c_star, N + 1, p_star):
            if not present[x]:
                continue
                
            score = 0
            for q in active_bad:
                if q == p_star:
                    continue
                if x % q == min_class[q]:
                    score += 1
                    
            # Add a small bias to remove elements that are "more central"?
            # Or just max score.
            if score > best_score:
                best_score = score
                best_x = x
                
        if best_x == -1:
            # Should not happen if is_bad is true
            break
            
        # Remove best_x
        present[best_x] = False
        current_size -= 1
        
        # Update state for all active bad primes
        # Note: k changes, so active_bad might change in next iter.
        # But for update, we only care about primes that were active bad.
        # Removing an element cannot make a good prime bad.
        # So we only update active_bad.
        
        for p in active_bad:
            c = best_x % p
            counts[p][c] -= 1
            
            if counts[p][c] == 0:
                is_bad[p] = False
                # Remove from bad_primes list? 
                # We filter active_bad each time, so just set flag.
                # But we need to remove it from the main bad_primes list to keep it small?
                # Or just filter.
                # Let's remove from bad_primes list for efficiency.
                if p in bad_primes:
                    bad_primes.remove(p)
            else:
                # Update min_count and min_class if necessary
                # If c was the min_class, min_count decreases.
                if c == min_class[p]:
                    min_count[p] -= 1
                    # It might no longer be the min if another class was tied and is now smaller?
                    # No, we decreased it. So it's still min or tied min.
                    # However, if there was another class with the same min count, 
                    # and we didn't touch it, the min count is the same.
                    # So min_count[p] is correct.
                else:
                    # We decreased a non-min class.
                    # The min class is unchanged.
                    # But what if the non-min class was tied for min?
                    # Then min_count is still correct.
                    # If it was strictly larger, min_count is correct.
                    # So no change needed for min_count.
                    # But wait, if we decrease a non-min class, could it become the new min?
                    # No, we are decreasing it. It gets smaller.
                    # If it was > min, it might become == min or < min?
                    # If it becomes < min, then the min has changed!
                    # Example: counts [5, 5]. Min is 5.
                    # Decrease class 1 by 1 -> [5, 4]. New min is 4.
                    # So we MUST check if the new count is less than current min_count.
                    if counts[p][c] < min_count[p]:
                        min_count[p] = counts[p][c]
                        min_class[p] = c
                    # If it becomes equal, we don't need to update min_class (tie is fine).
                    
    # Construct result
    result = [x for x in range(N + 1) if present[x]]
    return result
