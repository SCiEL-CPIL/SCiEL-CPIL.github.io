import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 75  # 75s safety margin within 90s limit

    def sf_kernel(n):
        """Return squarefree kernel of n using trial division."""
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
    min_val = 211
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)

    target = 210  # sf[210]

    # Map: squarefree value -> list of numbers (sorted) with that sf, in range [211, 10210]
    sf_to_nums = defaultdict(list)
    for n in range(min_val, max_val + 1):
        sf_to_nums[sf[n]].append(n)

    # Map: squarefree value -> smallest number with that sf
    sf_to_min = {}
    for n in range(min_val, max_val + 1):
        s = sf[n]
        if s not in sf_to_min or n < sf_to_min[s]:
            sf_to_min[s] = n

    def is_perfect_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n

    def sf_part_product(x, y):
        """Squarefree part of x*y where x,y are squarefree."""
        g = math.gcd(x, y)
        return x * y // (g * g)

    # Size 1: find n in [211, 10210] with sf[n] == 210
    # n = 210 * k^2, smallest > 210 is 210*4 = 840
    best_max = max_val
    best_list = None

    if 210 in sf_to_nums:
        best_max = sf_to_nums[210][0]
        best_list = [best_max]

    # Size 2: find a < b such that sf[210*a*b] = 1
    # Equivalent: sf_part_product(210, sf[a]) * sf[b] adjusted... 
    # Actually: sf[210*a*b] = sf_part(sf[210] * sf[a] * sf[b])
    # Let x = sf[210] * sf[a], but this may not be squarefree.
    # Better: the condition is that 210*a*b is a perfect square.
    # This means sf[a] * sf[b] must have squarefree part = 210.
    # i.e., sf_part_product(sf[a], sf[b]) == 210

    # For each coprime factor pair (x', y') of 210:
    #   sf[a] = x' * g, sf[b] = y' * g, where g is coprime to 210
    #   sf_part_product(x'*g, y'*g) = x'*y' = 210 ✓

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
    for x_prime, y_prime in pairs_210:
        if time.time() > deadline:
            break
        max_g = max_val // max(x_prime, y_prime)
        # g must be coprime to 210
        for g in range(1, max_g + 1):
            if math.gcd(g, 210) != 1:
                continue
            sa = x_prime * g
            sb = y_prime * g
            if sa in sf_to_nums and sb in sf_to_nums:
                a = sf_to_nums[sa][0]
                b = sf_to_nums[sb][0]
                if a != b:
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_list = [a, b]

    # Size 3: find a, b, c such that 210*a*b*c is a perfect square
    # Approach: for each M (candidate max value), check if there exist a, b < M
    # such that sf[210*a*b*M] = 1
    # sf[210*a*b*M] = sf_part(sf[210] * sf[a] * sf[b] * sf[M])
    # Let sM = sf[M], s210 = 210
    # We need: sf_part(210 * sf[a] * sf[b] * sM) = 1
    # Let X = sf_part(210 * sM). Then we need sf_part(sf[a] * sf[b] * X) = 1
    # i.e., sf[a] * sf[b] must have squarefree part = X (when combined with X to make square)
    # More precisely: sf_part(sf[a] * sf[b] * X) = 1 means sf[a]*sf[b] "complements" X
    # This means: sf_part(sf[a] * sf[b]) = X (since X is squarefree, and we need the total to be square)
    # Wait, that's not quite right. Let me think more carefully.
    # sf[n1*n2*n3] = 1 iff n1*n2*n3 is a perfect square.
    # The parity of each prime's exponent in n1*n2*n3 must be even.
    # Let P(p) be the parity of p's exponent in n. Then we need P_210(p) + P_a(p) + P_b(p) + P_M(p) ≡ 0 (mod 2) for all p.
    # This is equivalent to: the xor of the squarefree "signatures" of 210, a, b, M is 0.
    # In terms of sf: sf[210*a*b*M] = 1.
    # 
    # For fixed M, we need sf[a] * sf[b] to have squarefree part equal to sf[210*M] (adjusted).
    # Actually: sf[210*a*b*M] = sf_part(sf[210] * sf[a] * sf[b] * sf[M])
    # Let sM = sf[M]. We need sf_part(210 * sf[a] * sf[b] * sM) = 1.
    # Let Y = sf_part(210 * sM). Then we need sf_part(sf[a] * sf[b] * Y) = 1.
    # This means sf[a] * sf[b] must be such that when multiplied by Y, the result is a perfect square.
    # So sf[a] * sf[b] must have squarefree part = Y (since Y is squarefree).
    # i.e., sf_part(sf[a] * sf[b]) = Y.

    # For size 3, iterate M from min_val upward, and for each M, check if we can find a, b.
    # To be efficient, for each M, compute Y = sf_part(210 * sM).
    # Then we need sf[a] * sf[b] to have squarefree part Y.
    # This is similar to the size-2 problem but with target Y instead of 210.
    # For fixed Y, we need sf[a] * sf[b] to have squarefree part Y.
    # Using the same technique: coprime factor pairs of Y.

    # But Y can vary. Let's precompute for each M what Y is, and then check.

    # Actually, let's just do a direct search for size 3 with limited M range.
    # For M in range(min_val, best_max):
    #   Compute what sf[a] * sf[b] should be.
    #   Check if there exist a, b < M with that property.

    # For efficiency, precompute for each possible sf value the count of numbers with that sf.
    sf_counts = defaultdict(int)
    for n in range(min_val, max_val + 1):
        sf_counts[sf[n]] += 1

    # For size 3, iterate M from min_val to best_max - 1
    for M in range(min_val, min(best_max, max_val + 1)):
        if time.time() > deadline:
            break
        sM = sf[M]
        # Compute Y = sf_part(210 * sM)
        Y = sf_part_product(210, sM) if math.gcd(210, sM) > 1 else 210 * sM
        # Actually, sf_part_product works for squarefree inputs. 210 is squarefree, sM is squarefree.
        # But 210 and sM may share primes, so we need the proper formula.
        # sf_part(210 * sM) = 210 * sM / gcd(210, sM)^2
        g210sM = math.gcd(210, sM)
        Y = 210 * sM // (g210sM * g210sM)

        # Now we need sf[a] * sf[b] to have squarefree part Y.
        # Using coprime factor pairs of Y:
        # For each coprime factor pair (x', y') of Y:
        #   sf[a] = x' * g, sf[b] = y' * g, where g is coprime to Y
        #   We need a, b < M and a != b and a, b != M
        
        # To be fast, just check if there exist a, b < M with the right sf values.
        # For each coprime factor pair (x', y') of Y:
        pairs_Y = coprime_factor_pairs(Y)
        found = False
        for x_prime, y_prime in pairs_Y:
            if found:
                break
            max_g = min(best_max, M - 1) // max(x_prime, y_prime) if max(x_prime, y_prime) > 0 else 0
            if max_g < 1:
                continue
            for g in range(1, max_g + 1):
                if math.gcd(g, Y) != 1:
                    continue
                sa = x_prime * g
                sb = y_prime * g
                # Check if there exist a, b < M with sf[a]=sa, sf[b]=sb, a!=b
                if sa in sf_to_nums and sb in sf_to_nums:
                    # Find smallest a < M with sf[a]=sa
                    a_candidates = [n for n in sf_to_nums[sa] if n < M and n != M]
                    b_candidates = [n for n in sf_to_nums[sb] if n < M and n != M]
                    if a_candidates and b_candidates:
                        a = a_candidates[0]
                        b = b_candidates[0]
                        if a != b:
                            found = True
                            best_max = M
                            best_list = [a, b, M]
                            break
                        elif len(a_candidates) > 1 or len(b_candidates) > 1:
                            # Try different combinations
                            if sa == sb:
                                if len(a_candidates) >= 2:
                                    found = True
                                    best_max = M
                                    best_list = [a_candidates[0], a_candidates[1], M]
                                    break
                            else:
                                # a from a_candidates, b from b_candidates, they have different sf so a!=b always
                                found = True
                                best_max = M
                                best_list = [a, b, M]
                                break
        if found:
            break

    # If best_max didn't improve, try size 4 briefly
    # For size 4, we need 210*a*b*c*d to be a perfect square.
    # This is even more complex. Given time constraints, skip if not needed.

    # Fallback
    if best_list is None:
        best_list = [840]
        best_max = 840

    # Verify the solution
    prod = 210
    for x in best_list:
        prod *= x
    x = math.isqrt(prod)
    if x * x != prod:
        # Fallback to known valid
        best_list = [840]

    # Ensure all values are distinct Python ints
    result = [int(x) for x in best_list]
    return result
