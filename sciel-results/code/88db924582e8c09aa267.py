import time
import random

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _find_first_gap(residues, primes, limit=10000):
    """
    Find the smallest x >= 1 such that coverage < 2.
    Returns x if found, else limit + 1.
    """
    # residues is a list of ints
    # primes is a list of ints
    for x in range(1, limit + 1):
        count = 0
        for i, p in enumerate(primes):
            if x % p == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            return x
    return limit + 1

def _find_best_residue_for_gap(x, residues, primes, indices, current_score, eval_limit):
    """
    For a given gap x, try all possible residues for a specific prime index i.
    Returns (best_r, best_score) for that prime.
    """
    i = indices[0]
    p = primes[i]
    old_r = residues[i]
    
    best_r = old_r
    best_score = current_score
    
    # We only need to check residues that cover x, because if it doesn't cover x,
    # the gap remains x (or gets worse if it breaks other coverage? No, it only reduces coverage).
    # Actually, if we change r_i, we might break coverage for other numbers.
    # But if the new r_i doesn't cover x, the gap is still x (score is same or worse).
    # If the new r_i covers x, the gap might move forward.
    # So we only need to try r = x % p.
    
    new_r = x % p
    if new_r != old_r:
        residues[i] = new_r
        score = _find_first_gap(residues, primes, eval_limit) - 1
        if score > best_score:
            best_score = score
            best_r = new_r
        residues[i] = old_r
        
    return best_r, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    eval_limit = 50000 
    
    best_residues = None
    best_score = -1
    
    # Greedy Sequential Repair
    # Start with a random or simple initialization
    # r_p = 0 for all p is a common simple start, but might be bad.
    # Let's try a few random starts and repair each.
    
    for seed in range(5):
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        
        current_score = _find_first_gap(residues, primes, eval_limit) - 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Find the first gap
            gap = current_score + 1
            if gap > eval_limit:
                break
            
            # Try to fix the gap by changing one residue
            # We want to pick a prime p such that changing r_p to gap % p helps.
            # Any prime can be changed.
            # Let's try all primes? That's 168 evaluations.
            # Or random subset?
            # Since we have 115s and 168 evaluations is fast, let's try all.
            
            best_new_score = current_score
            best_new_r = -1
            best_idx = -1
            
            # Randomize order of primes to avoid bias
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for i in indices:
                p = primes[i]
                old_r = residues[i]
                new_r = gap % p
                
                if new_r == old_r:
                    continue
                    
                residues[i] = new_r
                score = _find_first_gap(residues, primes, eval_limit) - 1
                residues[i] = old_r
                
                if score > best_new_score:
                    best_new_score = score
                    best_new_r = new_r
                    best_idx = i
            
            if best_idx != -1:
                residues[best_idx] = best_new_r
                current_score = best_new_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
        
        # If we used up time, break outer loop too
        if time.time() > deadline - 10:
            break

    return best_residues
