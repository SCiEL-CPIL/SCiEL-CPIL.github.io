import math
import time
from collections import defaultdict

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
    
    def is_perfect_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Build map from sf value to sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for n in range(211, max_val + 1):
        sf_to_nums[sf[n]].append(n)
    # Each list is already sorted since we iterate in order
    
    # Also build sf_to_min for quick lookup
    sf_to_min = {}
    for s, nums in sf_to_nums.items():
        sf_to_min[s] = nums[0]
    
    # Target: squarefree kernel of 210 = 210
    target = 210
    
    best_max = max_val + 1
    best_list = None
    
    # Size 1: find n in [211, 10210] with sf[n] == 210
    if target in sf_to_min:
        n = sf_to_min[target]
        best_max = n
        best_list = [n]
    
    # Size 2: need sf[a]*sf[b]/gcd(sf[a],sf[b])^2 = 210
    # For coprime factor pairs (x', y') of 210, g coprime to 210:
    #   sf[a] = x'*g, sf[b] = y'*g
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
    
    for x_prime, y_prime in pairs_210:
        if time.time() > deadline:
            break
        max_g = max_val // max(x_prime, y_prime)
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
            g += 1
    
    # Size 3: For pair (a,b), need c such that sf[210*a*b*c]=1
    # sf[210*a*b*c] = sf_part(210 * sf[a] * sf[b] * sf[c])
    # We need sf_part(210 * sf[a] * sf[b] * sf[c]) = 1
    # Let R = sf_part(210 * sf[a] * sf[b]). Then we need sf_part(R * sf[c]) = 1,
    # which means sf[c] must equal R (since both are squarefree, sf_part(R*sf[c])=1 iff R=sf[c]).
    # Actually: sf_part(R * sf[c]) where R and sf[c] are squarefree = R*sf[c]/gcd(R,sf[c])^2
    # For this to be 1: R*sf[c] = gcd(R,sf[c])^2, so R and sf[c] must be equal.
    # So sf[c] = R = sf_part(210 * sf[a] * sf[b]).
    
    # Compute sf_part of product of squarefree numbers efficiently
    def sf_part_product(*squarefree_nums):
        """Compute squarefree part of product of squarefree numbers."""
        # Combine all: for each prime, count how many times it appears
        # If count is even, it's removed; if odd, it remains
        from collections import Counter
        prime_counts = Counter()
        for num in squarefree_nums:
            n = num
            d = 2
            while d * d <= n:
                if n % d == 0:
                    prime_counts[d] += 1
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                prime_counts[n] += 1
        result = 1
        for p, cnt in prime_counts.items():
            if cnt % 2 == 1:
                result *= p
        return result
    
    # For size 3, search more aggressively
    # Iterate over pairs (a, b) with a < b, compute required sf for c
    # Limit to pairs where max(a,b) < best_max to find improvements
    
    search_limit = min(best_max, 5000)
    
    # Precompute sf[a] for all a in range for quick access
    # For each a in [211, search_limit], for each b in [a+1, search_limit]:
    #   R = sf_part_product(210, sf[a], sf[b])
    #   If R in sf_to_min and sf_to_min[R] != a and sf_to_min[R] != b:
    #     c = sf_to_min[R]
    #     if c < best_max: update
    
    # This is O(search_limit^2) which could be too slow. Let's be smarter.
    # Instead, for each a, and for each possible sf value s_a = sf[a], 
    # group b by sf[b] = s_b. For each (s_a, s_b) pair, compute R.
    
    # Actually, let's just do it with time checks and limit the range.
    # First, try a smaller range to find good solutions quickly.
    
    for a in range(211, min(search_limit, 1000)):
        if time.time() > deadline:
            break
        sa = sf[a]
        for b in range(a + 1, min(search_limit, 1000)):
            if time.time() > deadline:
                break
            sb = sf[b]
            # Compute R = sf_part(210 * sa * sb)
            # Since 210, sa, sb are squarefree, we can compute this
            prod = 210 * sa * sb
            # Compute squarefree part of prod
            # prod may not be squarefree, so compute sf_kernel(prod)
            R = sf_kernel(prod)
            if R in sf_to_min:
                c = sf_to_min[R]
                if c != a and c != b and c < best_max:
                    m = max(a, b, c)
                    if m < best_max:
                        best_max = m
                        best_list = [a, b, c]
    
    # Try larger range if time permits
    if time.time() < deadline - 20:
        for a in range(211, min(search_limit, 3000)):
            if time.time() > deadline - 10:
                break
            sa = sf[a]
            for b in range(a + 1, min(search_limit, 3000)):
                if time.time() > deadline - 10:
                    break
                sb = sf[b]
                prod = 210 * sa * sb
                R = sf_kernel(prod)
                if R in sf_to_min:
                    c = sf_to_min[R]
                    if c != a and c != b and c < best_max:
                        m = max(a, b, c)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b, c]
    
    # Size 4: For triple (a,b,c), need d such that sf[210*a*b*c*d]=1
    # Similar approach but O(M^3) which is too slow. Skip unless best is still large.
    
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Verify the result
    prod = 210
    for x in best_list:
        prod *= x
    if not is_perfect_square(prod):
        # Fallback to known valid solution
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
