import time

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
    # 2. We need num/den >= 1/d + 1/(d+1) + ... + 1/(d+max_terms-1)
    #    A simpler bound: num/den >= max_terms/d => d >= max_terms * den / num
    #    Actually: num/den <= max_terms/d => d <= max_terms * den / num
    
    d_min = max(min_den, den // num + 1)
    d_max = min(max_den, (max_terms * den) // num)
    
    if d_min > d_max:
        return False

    # Try to find a solution by iterating through possible denominators
    # Use a smarter approach: try values that are more likely to work
    # First, try the minimum denominator, then increase
    
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
            # As d increases, rem_num/rem_den increases (since we're subtracting smaller fractions)
            # but (max_terms-1)/(d+1) decreases
            # So if this fails, it will fail for all larger d
            break
        
        # Also check minimum possible sum: (max_terms-1)/max_den
        # rem_num/rem_den >= (max_terms-1)/max_den
        # rem_num * max_den >= rem_den * (max_terms-1)
        if rem_num * max_den < rem_den * (max_terms - 1):
            # This means even with the largest possible denominators, we can't reach the sum
            # As d increases, rem increases, so this might become true later
            # But for now, skip this d
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
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def construct():
    # Try 8 terms first (minimum possible)
    # Use a time budget for each k to avoid spending too long on one value
    for k in range(8, 13):
        if not _deadline_ok():
            break
        out = []
        if _search(1, 1, k, 5, 10000000, out):
            if len(out) == k and _validate(out):
                return out
    
    # Fallback: try a known solution or construct one
    # For k=9, we can try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/x
    # Sum of first 8: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12
    # Let's compute this and see if we can find x
    
    # Actually, let's try a different approach: use the greedy algorithm with backtracking
    # Start with the smallest denominators and adjust
    
    # Known valid solution with 9 terms:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/x = 1
    # Sum of first 8: let's calculate
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in [5, 6, 7, 8, 9, 10, 12, 18]:
        s += Fraction(1, d)
    # s = 1 - 1/x, so 1/x = 1 - s
    if s < 1:
        rem = 1 - s
        if rem.denominator == 1:
            x = rem.numerator
        elif rem.numerator == 1:
            x = rem.denominator
        else:
            x = None
        
        if x and x >= 19 and x <= 10000000 and x not in [5, 6, 7, 8, 9, 10, 12, 18]:
            dens = [5, 6, 7, 8, 9, 10, 12, 18, x]
            if _validate(dens):
                return dens
    
    # Try another known solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/x
    s = Fraction(0, 1)
    for d in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
        s += Fraction(1, d)
    if s < 1:
        rem = 1 - s
        if rem.numerator == 1:
            x = rem.denominator
            if x >= 14 and x <= 10000000 and x not in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
                dens = [5, 6, 7, 8, 9, 10, 11, 12, 13, x]
                if _validate(dens):
                    return dens
    
    # If all else fails, raise an error
    raise RuntimeError("No representation found within deadline")
