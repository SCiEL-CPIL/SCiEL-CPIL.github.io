import time
import random

def _get_primes(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _backtrack_find_sequence(target_len, max_val=128, start_val=2, current_seq=None, current_products=None, timeout=None, start_time=None):
    """
    Recursively search for a strictly increasing sequence of length target_len 
    with distinct consecutive block products.
    """
    if current_seq is None:
        current_seq = []
    if current_products is None:
        current_products = set()
    if start_time is None:
        start_time = time.time()
        
    if len(current_seq) == target_len:
        return list(current_seq)
        
    if time.time() - start_time > timeout:
        return None
        
    last_val = current_seq[-1] if current_seq else start_val - 1
    
    # Candidates must be > last_val and <= max_val
    # We need to pick (target_len - len(current_seq)) more numbers
    # So the last number must be at least max_val - (remaining - 1)
    remaining = target_len - len(current_seq)
    min_last_candidate = max_val - (remaining - 1)
    
    # We iterate through possible next values
    # To increase diversity, we can shuffle the candidates
    candidates = list(range(last_val + 1, max_val + 1))
    # Prune candidates that are too small to allow enough remaining numbers?
    # If we pick x, we need 'remaining-1' numbers larger than x up to max_val.
    # So x <= max_val - (remaining - 1)
    candidates = [x for x in candidates if x <= max_val - (remaining - 1)]
    
    if not candidates:
        return None
        
    # Shuffle to find different solutions
    random.shuffle(candidates)
    
    for x in candidates:
        # Calculate new products
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(current_seq):
            prod *= a
            new_prods.append(prod)
            
        # Check collision
        if any(p in current_products for p in new_prods):
            continue
            
        # Also check if new_prods have duplicates among themselves?
        # Since seq is strictly increasing integers >= 2, products are strictly increasing?
        # x, x*a, x*a*b...
        # a,b >= 2, so x < x*a < x*a*b.
        # So new_prods are distinct.
        
        # Recurse
        new_seq = current_seq + [x]
        new_prods_set = current_products.union(new_prods)
        
        result = _backtrack_find_sequence(
            target_len, 
            max_val, 
            start_val, 
            new_seq, 
            new_prods_set, 
            timeout, 
            start_time
        )
        
        if result is not None:
            return result
            
    return None

def construct():
    """
    Use Backtracking with Increasing Target Length to maximize sequence length.
    """
    start_time = time.time()
    time_limit = 55.0
    
    # Baseline: Primes
    primes = _get_primes(128)
    best_seq = primes
    best_len = len(best_seq)
    
    # Try to extend
    # We know 30 is valid.
    # Try 31, 32, ...
    # We use a time limit per search attempt to avoid getting stuck.
    # If we find a solution of length L, we know L is possible.
    # We stop when we fail to find a solution of length L+1 within the time limit?
    # No, we should try hard to find L+1. If we can't, maybe it doesn't exist or is hard.
    # Given the small search space, if it exists, it should be found quickly with randomization.
    
    current_target = best_len + 1
    
    while current_target < 50: # Arbitrary upper bound
        if time.time() - start_time > time_limit:
            break
            
        # Allow some time for this specific length search
        # If we just found L, we have plenty of time.
        # If we are stuck, we might waste time.
        # Let's allocate 5 seconds per length attempt initially, reducing if time is low.
        attempt_timeout = min(5.0, time_limit - (time.time() - start_time))
        
        if attempt_timeout < 0.5:
            break
            
        found_seq = _backtrack_find_sequence(
            current_target, 
            max_val=128, 
            timeout=attempt_timeout, 
            start_time=time.time()
        )
        
        if found_seq:
            best_seq = found_seq
            best_len = current_target
            current_target += 1
        else:
            # If we failed to find a sequence of this length, 
            # it might be due to time limit or non-existence.
            # We can try again with a different random seed/shuffle?
            # The function shuffles internally.
            # Let's try one more time with a slightly longer timeout if we have time?
            # Or just assume it's not possible for now and stop?
            # To be safe, if we fail, we might miss a valid longer sequence.
            # But if we can't find it in 5s, it's likely hard or non-existent.
            # Let's try 2 more times with 2s each.
            retry_success = False
            for _ in range(2):
                if time.time() - start_time > time_limit - 1.0:
                    break
                attempt_timeout = min(2.0, time_limit - (time.time() - start_time))
                if attempt_timeout < 0.5:
                    break
                found_seq = _backtrack_find_sequence(
                    current_target, 
                    max_val=128, 
                    timeout=attempt_timeout, 
                    start_time=time.time()
                )
                if found_seq:
                    best_seq = found_seq
                    best_len = current_target
                    retry_success = True
                    break
            
            if not retry_success:
                # Assume we reached the maximum or time is too tight
                break
                
    return best_seq
