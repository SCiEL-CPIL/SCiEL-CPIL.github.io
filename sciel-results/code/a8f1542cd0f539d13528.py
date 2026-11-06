import time
from fractions import Fraction

_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _search(num, den, max_terms, min_den, max_den, out):
    """
    Recursively search for an Egyptian fraction representation.
    num/den is the remaining sum (in reduced form).
    max_terms is the number of terms left to find.
    min_den is the minimum allowed denominator (strictly increasing).
    max_den is the global maximum denominator.
    out is a list to store the denominators.
    Returns True if found, False otherwise.
    """
    if not _deadline_ok():
        return False
    
    # Reduce fraction
    g = _gcd(num, den)
    num //= g
    den //= g

    if max_terms == 0:
        return num == 0

    if max_terms == 1:
        # Need 1/d = num/den => d = den/num
        if den % num == 0:
            d = den // num
            if min_den <= d <= max_den:
                out.append(d)
                return True
        return False

    # Determine range for next denominator d
    # 1. 1/d < num/den => d > den/num, so d >= floor(den/num) + 1
    # 2. We need num/den >= max_terms/d => d >= max_terms * den / num
    #    Actually: num/den <= max_terms/d => d <= max_terms * den / num
    
    d_min = max(min_den, den // num + 1)
    d_max = min(max_den, (max_terms * den) // num)
    
    if d_min > d_max:
        return False

    # Try to find a solution by iterating through possible denominators
    for d in range(d_min, d_max + 1):
        if not _deadline_ok():
            return False
        
        # Remaining: num/den - 1/d = (num*d - den) / (den*d)
        rem_num = num * d - den
        rem_den = den * d
        if rem_num <= 0:
            continue
        
        # Pruning: Check if remaining sum can be achieved with max_terms-1 terms
        # starting from d+1.
        # Max possible sum of (max_terms-1) terms with denominators >= d+1 is (max_terms-1)/(d+1)
        # We need rem_num/rem_den <= (max_terms-1)/(d+1)
        # rem_num * (d+1) <= rem_den * (max_terms-1)
        if rem_num * (d + 1) > rem_den * (max_terms - 1):
            break
        
        # Also check minimum possible sum: (max_terms-1)/max_den
        # rem_num/rem_den >= (max_terms-1)/max_den
        # rem_num * max_den >= rem_den * (max_terms-1)
        if rem_num * max_den < rem_den * (max_terms - 1):
            continue
        
        out.append(d)
        if _search(rem_num, rem_den, max_terms - 1, d + 1, max_den, out):
            return True
        out.pop()
        
    return False

def _validate(dens):
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    for d in dens:
        if d < 5 or d > 10000000:
            return False
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def construct():
    # The minimum number of terms is 8.
    # We search for a solution with exactly 8 terms.
    out = []
    if _search(1, 1, 8, 5, 10000000, out):
        if len(out) == 8 and _validate(out):
            return out
    
    # Fallback: If the search fails (unlikely), try 9 terms
    out = []
    if _search(1, 1, 9, 5, 10000000, out):
        if len(out) == 9 and _validate(out):
            return out
            
    # Hardcoded fallback (should never be reached if search works)
    # This is a valid 8-term solution found via search logic:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/2520? No.
    # Let's just raise an error if we fail, but the search should succeed.
    raise RuntimeError("No representation found")
