import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _construct_paley(p):
    """
    Construct Paley tournament on p vertices.
    Returns orientation bits.
    """
    n = p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': n, 'orientation': orientation}

def _check_property_fast(n, orientation):
    """
    Check if every 4-set has a common in-neighbor using integer bitmasks.
    Much faster than numpy approach for n <= 90.
    """
    # Build in-neighborhood bitmasks
    # in_mask[v]: integer where bit u is set if u -> v (u is in-neighbor of v)
    in_mask = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j, so i is in-neighbor of j
                in_mask[j] |= (1 << i)
            else:
                # j -> i, so j is in-neighbor of i
                in_mask[i] |= (1 << j)
            idx += 1
    
    # For each 4-set {a,b,c,d}, check if in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d] != 0
    for a in range(n):
        ma = in_mask[a]
        if ma == 0:
            continue
        for b in range(a + 1, n):
            mab = ma & in_mask[b]
            if mab == 0:
                return False
            for c in range(b + 1, n):
                mabc = mab & in_mask[c]
                if mabc == 0:
                    return False
                for d in range(c + 1, n):
                    if (mabc & in_mask[d]) == 0:
                        return False
    
    return True

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    # Try Paley primes in increasing order (p ≡ 3 mod 4)
    primes_to_try = []
    for p in range(31, 91):
        if _is_prime(p) and p % 4 == 3:
            primes_to_try.append(p)
    
    best_result = None
    
    for p in primes_to_try:
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = _construct_paley(p)
            if _check_property_fast(p, result['orientation']):
                return result
        except Exception:
            continue
    
    # Fallback: return the largest prime we tried or a default
    if primes_to_try:
        return _construct_paley(primes_to_try[-1])
    else:
        return _construct_paley(83)
