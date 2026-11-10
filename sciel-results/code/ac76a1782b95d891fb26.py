import time
import random
from fractions import Fraction

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _check_budget(primes):
    total = Fraction(0, 1)
    for p in primes:
        total += Fraction(1, p)
    return total <= 1

def _first_gap(residues, primes, limit=10000):
    """
    Returns the first integer n >= 1 not covered by the classes.
    If all 1..limit are covered, returns limit + 1.
    """
    # residues is a list of r_p corresponding to primes
    for n in range(1, limit + 1):
        covered = False
        for i, p in enumerate(primes):
            if n % p == residues[i]:
                covered = True
                break
        if not covered:
            return n
    return limit + 1

def _local_search(primes, initial_residues, max_iterations=500, limit=5000):
    """
    Hill climbing to maximize the first gap.
    """
    current_res = initial_residues[:]
    current_gap = _first_gap(current_res, primes, limit)
    
    for _ in range(max_iterations):
        best_res = current_res[:]
        best_gap = current_gap
        
        # Try changing one residue
        # Shuffle order of primes to avoid bias
        indices = list(range(len(primes)))
        random.shuffle(indices)
        
        for i in indices:
            p = primes[i]
            old_r = current_res[i]
            # Try all other residues
            for new_r in range(p):
                if new_r == old_r:
                    continue
                current_res[i] = new_r
                gap = _first_gap(current_res, primes, limit)
                if gap > best_gap:
                    best_gap = gap
                    best_res = current_res[:]
                # Restore for next trial in this loop? 
                # No, we can keep the best found so far in this pass
                # But to be a proper hill climb, we should revert if not better
                # Or we can do a "best improvement" step
                # Let's just track the best move in this full scan
            # Revert to old_r after trying all? 
            # Actually, the above loop overwrites current_res. 
            # We need to be careful. 
            # Let's reset current_res[i] to old_r before trying the next prime
            current_res[i] = old_r
            
        if best_gap > current_gap:
            current_res = best_res
            current_gap = best_gap
        else:
            break
            
    return current_res, current_gap

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Generate primes up to 1000
    all_primes = _sieve_primes(1000)
    
    # Define candidate sets of primes that satisfy the budget
    # We will manually select sets that are likely good
    # Set 1: {2, 5, 7, 11, 13, 17}
    set1 = [2, 5, 7, 11, 13, 17]
    # Set 2: {2, 7, 11, 13, 17, 19, 23, 29}
    set2 = [2, 7, 11, 13, 17, 19, 23, 29]
    # Set 3: {3, 5, 7, 11, 13, 17, 19, 23}
    set3 = [3, 5, 7, 11, 13, 17, 19, 23]
    # Set 4: {2, 3, 11, 13}
    set4 = [2, 3, 11, 13]
    # Set 5: {2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997}
    # Obviously Set 5 is too big. We need to select a subset.
    
    # Let's just use the small sets.
    candidate_sets = [set1, set2, set3, set4]
    
    # Verify budgets
    valid_sets = []
    for s in candidate_sets:
        if _check_budget(s):
            valid_sets.append(s)
    
    best_classes = []
    best_score = 0
    best_primes = []
    
    # Run local search on each valid set
    for primes in valid_sets:
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random residues
            res = [random.randint(0, p - 1) for p in primes]
            
            # Run local search
            res, gap = _local_search(primes, res, max_iterations=100, limit=5000)
            
            # If we hit the limit, try to extend?
            # For now, just record the gap.
            if gap > best_score:
                best_score = gap
                best_primes = primes
                # Store residues corresponding to best_primes
                # We need to map back to classes
                best_res = res
                
    # Construct the output
    if best_primes:
        classes = []
        for i, p in enumerate(best_primes):
            classes.append([p, best_res[i]])
        return classes
    else:
        # Fallback: return empty or a trivial valid set
        # The seed returns a set. We must return a valid set.
        # Let's return the seed's logic if we failed?
        # Or just a single prime class [2, 0] which covers evens.
        return [[2, 0]]

