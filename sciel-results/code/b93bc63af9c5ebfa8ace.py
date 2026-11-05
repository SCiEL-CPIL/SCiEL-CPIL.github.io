import time
import sys
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # keep a safety margin before the 90s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

# 12! = 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
TARGET = factorial(12)
PRIMES = [2, 3, 5, 7, 11]
EXPONENTS = [10, 5, 2, 1, 1]

def _factorize(n):
    """Return exponents of PRIMES in n (None if n has other prime factors)."""
    exponents = []
    for p in PRIMES:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        exponents.append(e)
    if n != 1:
        return None
    return exponents

def _divisors_from_exponents(exponents):
    """Generate all divisors from prime exponent vector."""
    divs = [1]
    for i, e in enumerate(exponents):
        p = PRIMES[i]
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    return divs

def _divisors_in_range(rem, lo, hi):
    """Get divisors of rem in [lo, hi], sorted."""
    exps = _factorize(rem)
    if exps is None:
        return []
    divs = _divisors_from_exponents(exps)
    result = [d for d in divs if lo <= d <= hi]
    result.sort()
    return result

def _dfs(rem, min_val, max_val, chosen, path, best):
    """
    DFS to find factors.
    rem: product remaining
    min_val: smallest factor allowed (for distinctness/ordering)
    max_val: largest factor allowed (the bound M)
    chosen: set of chosen factors
    path: list of factors in current path
    best: list to store best solution found
    """
    if _time_left() < 2:
        return False
    
    if rem == 1:
        # Found a valid decomposition
        if len(path) >= 1:
            best[0] = list(path)
        return True
    
    if len(path) >= 12:
        return False
    
    if min_val > max_val:
        return False
        
    if rem < min_val:
        return False
        
    # Get divisors in range
    divs = _divisors_in_range(rem, min_val, max_val)
    
    if not divs:
        return False
    
    for d in divs:
        if d in chosen:
            continue
        
        chosen.add(d)
        path.append(d)
        
        if _dfs(rem // d, d + 1, max_val, chosen, path, best):
            return True
        
        path.pop()
        chosen.remove(d)
        
    return False

def construct():
    # Search for minimal M such that a valid decomposition exists
    # All factors must be > 12 and <= M
    # Lower bound: 13
    # Upper bound: TARGET (trivial solution [TARGET])
    
    best = [None]
    
    # First, try small M values where solutions are likely to exist
    # 12! = 479001600
    # We need distinct factors all > 12
    # Let's try M from 13 upward
    
    for M in range(13, 2000):
        if _time_left() < 5:
            break
        chosen = set()
        path = []
        if _dfs(TARGET, 13, M, chosen, path, best):
            return path
    
    # Fallback: return a valid solution
    # Try to construct one greedily
    # 12! = 479001600
    # Let's try to find a decomposition manually
    # 479001600 / 13 = 36846276.9... not integer
    # 479001600 / 14 = 34214400 exactly
    # 34214400 / 15 = 2280960 exactly
    # 2280960 / 16 = 142560 exactly
    # 142560 / 17 = 8385.88... not integer
    # 142560 / 18 = 7920 exactly
    # 7920 / 19 = 416.84... not integer
    # 7920 / 20 = 396 exactly
    # 396 / 21 = 18.857... not integer
    # 396 / 22 = 18 exactly
    # 18 / 23 = 0.78... not integer
    # So: 14 * 15 * 16 * 18 * 20 * 22 * 18... wait, 18 is repeated
    # Let me redo: 12! = 479001600
    # 479001600 = 14 * 15 * 16 * 18 * 20 * 22 * ?
    # 14*15*16*18*20*22 = 14*15=210, 210*16=3360, 3360*18=60480, 60480*20=1209600, 1209600*22=26611200
    # 479001600 / 26611200 = 18.0... wait that's 18 again
    # Let me try different factors
    
    # Actually, let's just return a known valid decomposition
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # We can group these into distinct factors > 12
    
    # Try: 16, 18, 20, 22, 25, 27, 35, 55, 72
    # 16 = 2^4
    # 18 = 2 * 3^2
    # 20 = 2^2 * 5
    # 22 = 2 * 11
    # 25 = 5^2
    # 27 = 3^3
    # 35 = 5 * 7
    # 55 = 5 * 11
    # 72 = 2^3 * 3^2
    # Total: 2^(4+1+2+1+0+0+0+0+3) = 2^11 - too many 2s
    # We need 2^10
    
    # Let me try: 16, 18, 20, 22, 25, 27, 35, 55, 36
    # 36 = 2^2 * 3^2
    # Total 2s: 4+1+2+1+0+0+0+0+2 = 10 ✓
    # Total 3s: 0+2+0+0+0+3+0+0+2 = 7 - too many (need 5)
    
    # Try: 16, 18, 20, 22, 25, 9, 35, 55, 36 - 9 is not > 12
    
    # Let me just use a simple fallback
    return [TARGET]
