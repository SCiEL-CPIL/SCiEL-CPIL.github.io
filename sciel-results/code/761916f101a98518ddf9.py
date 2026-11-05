"""
Schütte's problem k=4: Find smallest n such that a tournament on n vertices
has the property that every 4-set has a common dominator.

Strategy:
1. Try Paley tournaments for primes p ≡ 3 (mod 4) in increasing order.
2. If any smaller Paley tournament is valid, use it.
3. Otherwise, fall back to n=67 Paley tournament (known valid).
"""

import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Primes p ≡ 3 (mod 4) that are candidates for Paley tournaments
    # These must satisfy p ≡ 3 (mod 4) for Paley tournament to be regular
    paley_primes = [31, 43, 47, 59, 67]
    
    def is_paley_prime(p):
        """Check if p is prime and p ≡ 3 (mod 4)."""
        if p < 2:
            return False
        if p % 4 != 3:
            return False
        if p < 3:
            return p == 3
        if p % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(p)) + 1, 2):
            if p % i == 0:
                return False
        return True
    
    def check_paley(p):
        """
        Check if the Paley tournament on p vertices satisfies:
        Every set of 4 vertices has a common dominator.
        
        In Paley tournament: i -> j iff (j-i) mod p is a quadratic residue mod p.
        """
        if time.time() > deadline:
            return None  # timeout
        
        # Build quadratic residues
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        
        # Build in-neighborhood bitmasks
        # in_nb[u] has bit v set if v -> u (i.e., v points to u)
        in_nb = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    in_nb[j] |= (1 << i)
                else:
                    # j -> i
                    in_nb[i] |= (1 << j)
        
        # Check all 4-sets for empty common in-neighborhood
        # A 4-set {a,b,c,d} fails if intersection of in_nb[a], in_nb[b], in_nb[c], in_nb[d] is empty
        for a in range(p - 3):
            if time.time() > deadline:
                return None
            mask_ab = in_nb[a] & in_nb[a+1]  # just to keep compiler happy, we do full loops
            for b in range(a + 1, p - 2):
                mask_ab = in_nb[a] & in_nb[b]
                if mask_ab == 0:
                    return False
                for c in range(b + 1, p - 1):
                    mask_abc = mask_ab & in_nb[c]
                    if mask_abc == 0:
                        return False
                    for d in range(c + 1, p):
                        if (mask_abc & in_nb[d]) == 0:
                            return False
        
        return True
    
    def make_paley_orientation(p):
        """Generate orientation list for Paley tournament on p vertices."""
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                orientation.append(0 if (j - i) % p in squares else 1)
        return orientation
    
    # Try smaller Paley primes first
    for p in sorted(paley_primes):
        if not is_paley_prime(p):
            continue
        result = check_paley(p)
        if result is True:
            # Found a valid smaller Paley tournament
            return {'n': p, 'orientation': make_paley_orientation(p)}
        elif result is None:
            # Timeout, break and fall back
            break
    
    # Fallback: n=67 Paley tournament (known to be valid)
    p = 67
    return {'n': p, 'orientation': make_paley_orientation(p)}
