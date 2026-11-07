import math
import time

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def sf_kernel(n):
        """Return squarefree kernel of n."""
        if n <= 1:
            return 1
        result = 1
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if f > 1:
            result *= f
        return result
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Target: squarefree kernel of 210 = 2*3*5*7 = 210
    target = 210
    
    # Check size 1: find n in [211, 10210] with sf[n] == 210
    # sf[n] == 210 means n = 210 * k^2 for some k, and n > 210
    # smallest: 210*4 = 840
    best_max = max_val  # worst case
    best_list = None
    
    # Size 1
    for n in range(211, max_val + 1):
        if sf[n] == target:
            if n < best_max:
                best_max = n
                best_list = [n]
            break  # first one is smallest
    
    # Size 2: find a < b in [211, M] such that sf[a] * sf[b] has squarefree part 210
    # i.e., sf[a] * sf[b] = 210 * s^2 for some s
    # Equivalently: sf[a] * sf[b] / gcd(sf[a]*sf[b], 210) ... 
    # Better: the squarefree part of sf[a]*sf[b] must be 210
    # sf[a]*sf[b] is not necessarily squarefree. Its squarefree part = sf[a]*sf[b] / (product of common primes)^2
    # Actually: sf_part(a*b) = sf[a] * sf[b] / (sf[gcd(a,b)]^2) ... this is getting complex
    # Simpler: just check if 210 * a * b is a perfect square directly for pairs
    
    def is_perfect_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n
    
    # For size 2, check pairs. To be efficient, iterate b from 211 upward, and for each b,
    # check if there exists a < b such that 210*a*b is a perfect square.
    # 210*a*b = perfect square => a*b must have the right prime parities.
    # Let's just do it directly for a bounded window.
    
    # Search for size 2 in increasing window
    found_size2 = False
    for M in range(212, max_val + 1):
        if time.time() > deadline:
            break
        if M <= best_max and best_list is not None:
            # We can't improve on size 1 with max < best_max for size 2 since both > 210
            # Actually size 2 could have max < best_max
            pass
        # Check pairs (a, M) where a in [211, M-1]
        # 210 * a * M must be perfect square
        # This is equivalent to: a * M must have squarefree part = 210 / sf[210] ... 
        # Let's just compute: is 210 * a * M a perfect square?
        # To speed up: for fixed M, we need a such that sf[a] * sf[M] has squarefree part 210
        # i.e., sf[a] * sf[M] = 210 * k^2
        # Since sf[a] and sf[M] are squarefree, let g = gcd(sf[a], sf[M])
        # Then sf[a]*sf[M] = g * (sf[a]/g) * (sf[M]/g), and the squarefree part is (sf[a]/g)*(sf[M]/g) if g is squarefree... 
        # This is getting complicated. Let me just use the direct check but limit the search.
        
        # Actually, let's use a different approach: for each M, compute what sf[a] should be.
        # 210 * a * M = square => sf[210*a*M] = 1
        # sf[210*a*M] = sf[210] * sf[a] * sf[M] / (corrections for shared primes)
        # This is hard to invert. Let me just do direct perfect square check for limited pairs.
        
        # For efficiency, only check a in a small range or use the sf relationship
        # sf[a] must satisfy: the squarefree part of (sf[a] * sf[M]) equals 210
        # Let's precompute: for each possible sf value, what's the smallest number with that sf?
        
        break  # Don't do exhaustive size 2 in this loop, do it separately
    
    # Better approach: build a map from sf value to smallest number
    sf_to_min = {}
    for n in range(211, max_val + 1):
        s = sf[n]
        if s not in sf_to_min or n < sf_to_min[s]:
            sf_to_min[s] = n
    
    # For size 2: need sf[a] * sf[b] to have squarefree part 210
    # The squarefree part of product of two squarefree numbers x*y is x*y/gcd(x,y)^2 * gcd(x,y) = x*y/gcd(x,y)
    # Wait: if x and y are squarefree, x*y has each prime to power 0, 1, or 2. The squarefree part removes primes with even exponent.
    # So sf_part(x*y) = x*y / (product of primes in both x and y)^2 * (product of primes in both) = x*y / gcd(x,y)
    # No: if p is in both x and y, p^2 divides x*y, so p is removed. If p is in exactly one, p remains.
    # So sf_part(x*y) = (x/gcd(x,y)) * (y/gcd(x,y)) = x*y/gcd(x,y)^2 * gcd(x,y) = x*y/gcd(x,y)
    # Let me verify: x=6, y=10, gcd=2. x*y=60, sf_part(60)=sf_part(2^2*3*5)=15. x*y/gcd=60/2=30. Wrong.
    # x*y/gcd(x,y)^2 = 60/4 = 15. Correct!
    # So sf_part(x*y) = x*y / gcd(x,y)^2 when x,y are squarefree.
    
    # So we need: sf[a] * sf[b] / gcd(sf[a], sf[b])^2 = 210
    # Let x = sf[a], y = sf[b], g = gcd(x,y)
    # x*y/g^2 = 210 => (x/g)*(y/g) = 210
    # Let x' = x/g, y' = y/g, then x'*y' = 210 and gcd(x',y')=1
    # So we need to factor 210 into two coprime factors x', y'
    # 210 = 2*3*5*7. Coprime factor pairs: (1,210), (2,105), (3,70), (5,42), (6,35), (7,30), (10,21), (14,15), and reverses.
    
    # For each coprime factor pair (x', y') of 210, we need:
    # sf[a] = x' * g, sf[b] = y' * g for some g that is coprime to 210 (since x',y' are coprime to each other and contain all primes of 210)
    # Wait, g can have other primes. x' and y' are coprime, and x'*y'=210.
    # sf[a] = x'*g, sf[b] = y'*g. For sf[a] to be squarefree, g must be coprime to x'. Similarly g coprime to y'.
    # Since x' and y' are coprime and x'*y'=210, g must be coprime to 210.
    
    # So for each coprime factor pair (x', y') of 210, and for each g coprime to 210:
    #   sf[a] = x'*g, sf[b] = y'*g
    #   Find smallest a with sf[a]=x'*g and smallest b with sf[b]=y'*g, a != b
    #   Max(a,b) is a candidate
    
    # Generate coprime factor pairs of 210
    def coprime_factor_pairs(n):
        pairs = []
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                e = n // d
                if math.gcd(d, e) == 1:
                    pairs.append((d, e))
                    if d != e:
                        pairs.append((e, d))
        return pairs
    
    pairs_210 = coprime_factor_pairs(210)
    
    # For size 2, try all coprime factor pairs and g values
    # g must be coprime to 210. sf[a] = x'*g <= 10210, so g <= 10210/x'
    # Let's enumerate g for each pair
    
    for x_prime, y_prime in pairs_210:
        if time.time() > deadline:
            break
        max_g = max_val // max(x_prime, y_prime)
        # g must be coprime to 210
        # Enumerate g from 1 upward
        g = 1
        while g <= max_g:
            if math.gcd(g, 210) == 1:
                sa = x_prime * g
                sb = y_prime * g
                if sa in sf_to_min and sb in sf_to_min:
                    a = sf_to_min[sa]
                    b = sf_to_min[sb]
                    if a != b:
                        m = max(a, b)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b]
                # Also try swapping: a from sb, b from sa (but we already have both in the map)
                # Actually sf_to_min gives the smallest number for each sf value, so we've covered it
            g += 1
    
    # Size 3 and beyond: more complex. Let's also try size 3 with a similar approach.
    # For size 3: sf[a]*sf[b]*sf[c] should have squarefree part 210.
    # This is harder. Let's try a different approach: BFS over squarefree values.
    
    # Alternative: for small subset sizes, do a direct search with pruning.
    # Let's try size 3 by iterating over pairs and checking if a third completes it.
    
    # Actually, let's try a more general approach: 
    # Build a map from sf value to list of numbers (sorted) with that sf value.
    # Then for size 2, we already did it. For size 3, we need sf[a]*sf[b]*sf[c] to have sf part 210.
    
    # Given time constraints, let's also try: for each pair (a,b) with max(a,b) < best_max,
    # check if there's a c such that 210*a*b*c is a perfect square.
    # 210*a*b*c = square => c must have the right sf.
    # sf[210*a*b*c] = 1 => sf[c] must "complete" 210*a*b to a square.
    # Let P = 210*a*b. We need c such that P*c is a perfect square.
    # This means sf[c] must equal sf[P] (the squarefree part of P).
    # So we need a number c with sf[c] = sf[P], c distinct from a,b, and c > 210.
    
    # Let's implement this for size 3.
    
    # First, compute sf for all numbers up to 10210 (already done)
    # For size 3: iterate over pairs (a,b) with a<b, both in [211, best_max-1]
    # Compute P = 210*a*b, find sf[P], look up if there's a c with sf[c]=sf[P]
    
    # But this could be O(M^2) which is too slow for large M.
    # Let's limit: only check pairs where max(a,b) < current best_max.
    
    # Precompute sf for products? No, too large.
    # Instead: sf[210*a*b] = sf[210] * sf[a] * sf[b] / (corrections)
    # This is complex. Let me just compute sf of the product directly for limited cases.
    
    # Actually, for the product of numbers, the squarefree part can be computed from the individual sf parts.
    # sf[a*b] = sf[a] * sf[b] / gcd(sf[a], sf[b])^2 * gcd(sf[a], sf[b]) = sf[a]*sf[b]/gcd(sf[a],sf[b])
    # Wait, I showed earlier that sf_part(x*y) = x*y/gcd(x,y)^2 for squarefree x,y.
    # Let me re-verify: x=6=2*3, y=10=2*5, gcd=2. x*y=60=2^2*3*5, sf_part=3*5=15. x*y/gcd^2=60/4=15. Correct.
    # x=6, y=15=3*5, gcd=3. x*y=90=2*3^2*5, sf_part=2*5=10. x*y/gcd^2=90/9=10. Correct.
    # So sf_part(x*y) = x*y / gcd(x,y)^2 for squarefree x,y.
    
    # So sf[210*a*b] where 210, a, b may not be squarefree...
    # sf[n1*n2] = sf[n1]*sf[n2]/gcd(sf[n1],sf[n2])^2 * gcd(sf[n1],sf[n2]) ... no.
    # Actually sf[n1*n2] = sf_part(n1*n2). And sf_part(n1*n2) = sf_part(sf[n1] * sf[n2]) since sf[n] captures the odd-exponent primes.
    # And sf_part(x*y) for squarefree x,y = x*y/gcd(x,y)^2.
    # So sf[210*a*b] = sf_part(sf[210] * sf[a] * sf[b])
    # = sf_part(210 * sf[a] * sf[b])
    # Let x = 210 * sf[a], y = sf[b]. Both squarefree? 210 is squarefree, sf[a] is squarefree, but 210*sf[a] may not be squarefree if they share primes.
    # This is getting complicated. Let me just compute sf of the product directly for the limited search.
    
    # For size 3, limit the search to pairs with small max value.
    if best_list is None or best_max > 1000:
        # Try size 3 with limited pair enumeration
        # For a in [211, 500], b in [a+1, 500]:
        for a in range(211, min(500, best_max)):
            if time.time() > deadline:
                break
            for b in range(a + 1, min(500, best_max)):
                if time.time() > deadline:
                    break
                # Compute sf of 210*a*b
                prod = 210 * a * b
                sfp = sf_kernel(prod)
                if sfp in sf_to_min:
                    c = sf_to_min[sfp]
                    if c > 210 and c != a and c != b:
                        m = max(a, b, c)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b, c]
    
    # If we found a good result, return it. Otherwise, fall back to size 1.
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
