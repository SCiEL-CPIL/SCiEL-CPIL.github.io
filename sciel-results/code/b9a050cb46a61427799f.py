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

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Initial set
    S = set(range(N + 1))
    
    # Precompute primes up to N
    primes = _primes(N)
    prime_set = set(primes)
    
    # Data structures for fast updates
    # counts[p] is a list of size p
    counts = {}
    # elements_in_class[p][r] is a set of elements in S with x % p == r
    elements_in_class = {}
    
    for p in primes:
        counts[p] = [0] * p
        elements_in_class[p] = [set() for _ in range(p)]
        
    for x in S:
        for p in primes:
            r = x % p
            counts[p][r] += 1
            elements_in_class[p][r].add(x)
            
    # Identify initial bad primes
    # Bad if p <= |S| and min(counts[p]) > 0
    bad_primes = set()
    for p in primes:
        if p <= len(S):
            if min(counts[p]) > 0:
                bad_primes.add(p)
                
    # Candidates: elements that are the sole occupant of a class for some bad prime
    # We will maintain a list of (score, x) or just recalculate if small
    # Given the size, recalculating score for all x in S might be too slow if S is large.
    # But we can restrict to "active" candidates.
    
    while bad_primes:
        if time.time() > deadline:
            break
            
        L = len(S)
        # Update bad_primes definition based on current L
        # Primes p > L are not bad by definition
        # Primes p <= L are bad if min(counts[p]) > 0
        
        # Re-evaluate bad primes? 
        # As we remove elements, min(counts[p]) can only decrease or stay same.
        # It can hit 0, removing p from bad_primes.
        # It cannot go from 0 to > 0.
        # So bad_primes only shrinks (plus the effect of L shrinking).
        
        # Check if any prime p <= L has min > 0
        # We maintain bad_primes dynamically.
        
        # Find candidates
        # An element x is a candidate if there exists p in bad_primes such that counts[p][x%p] == 1
        # Let's collect all such x.
        
        candidate_scores = {}
        for p in bad_primes:
            if p > L:
                bad_primes.discard(p)
                continue
            # Check if p is still bad
            # min(counts[p]) > 0 is true if bad_primes contains it, 
            # but we should verify if it became 0.
            # Actually, we update bad_primes when counts change.
            # Let's assume bad_primes is accurate.
            
            for r in range(p):
                if counts[p][r] == 1:
                    # The single element in this class is a candidate
                    # elements_in_class[p][r] has size 1
                    x = next(iter(elements_in_class[p][r]))
                    if x in candidate_scores:
                        candidate_scores[x] += 1
                    else:
                        candidate_scores[x] = 1
                        
        if candidate_scores:
            # Pick x with max score
            x_to_remove = max(candidate_scores, key=candidate_scores.get)
        else:
            # No immediate candidates (no class has count 1 for any bad prime)
            # We must reduce counts to create candidates.
            # Strategy: Remove an element that is in the minimum-count class for the most bad primes.
            # Or simply: For each bad prime p, find the min count m_p.
            # Pick x that is in a class with count m_p for the most p.
            
            target_scores = {}
            for p in bad_primes:
                if p > L:
                    bad_primes.discard(p)
                    continue
                m_p = min(counts[p])
                # Find residues r where counts[p][r] == m_p
                # For each such r, pick an element x from elements_in_class[p][r]
                # And add p to target_scores[x]
                # To avoid picking the same x multiple times for the same p, 
                # we just need to know that x is in a min-class for p.
                
                # Optimization: Just pick one element from a min class for p and credit it.
                # This is a heuristic.
                min_residues = [r for r in range(p) if counts[p][r] == m_p]
                if min_residues:
                    # Pick random min residue to avoid bias? Or first one?
                    # Let's pick the one with the largest set of elements? No, they are all min.
                    # Just pick the first one.
                    r = min_residues[0]
                    # Pick an element from this class
                    # We need to be consistent. Let's just pick the smallest element in the class?
                    # Or random.
                    x = min(elements_in_class[p][r]) # Deterministic
                    if x in target_scores:
                        target_scores[x] += 1
                    else:
                        target_scores[x] = 1
                        
            if target_scores:
                x_to_remove = max(target_scores, key=target_scores.get)
            else:
                # Should not happen if bad_primes is not empty
                break
                
        # Remove x_to_remove
        S.remove(x_to_remove)
        
        # Update data structures
        for p in primes:
            if p > len(S):
                # We don't strictly need to update counts for p > len(S) 
                # because they won't be in bad_primes.
                # But if len(S) increases (it doesn't), we would.
                # Since len(S) only decreases, primes that are > len(S) will never become <= len(S) again.
                # So we can skip updating counts for p > len(S) to save time?
                # Wait, if p was <= old_len and is > new_len, it drops out of consideration.
                # We still need to update counts for p <= new_len.
                pass
            
            r = x_to_remove % p
            # Update count
            counts[p][r] -= 1
            elements_in_class[p][r].discard(x_to_remove)
            
            # Update bad_primes status
            if p in bad_primes:
                # If count became 0, this class is missed.
                # Check if min(counts[p]) == 0
                # This requires a min() call which is O(p).
                # We can optimize by keeping track of min, but O(p) is small for small p.
                # For large p, min() is expensive.
                # However, large p have large counts, so they are less likely to hit 0 quickly.
                # Let's just do min() for now.
                if min(counts[p]) == 0:
                    bad_primes.discard(p)
                    
            # If p was not in bad_primes, it might become bad?
            # No, removing elements only decreases counts.
            # If min was 0, it stays 0.
            # So bad_primes only shrinks.
            
    return list(S)
