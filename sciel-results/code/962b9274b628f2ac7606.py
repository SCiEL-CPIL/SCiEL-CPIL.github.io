"""Structural redesign: meet-in-the-middle via powerful-number tables + vectorized
pair-counting over a bounded prefix, with a hard wall-clock deadline guard."""
import time
import numpy as np
from math import isqrt


def _powerful_numbers_upto(L):
    """All powerful numbers <= L (1 is powerful)."""
    if L < 1:
        return []
    # powerful numbers are of the form prod p_i^{e_i} with e_i >= 2.
    # Generate by iterating over squares and cubic contributions.
    # Approach: for each "core" c = prod p_i^{f_i} with f_i >= 1,
    # and each square factor s^2, the number is c * s^2, but we must ensure
    # every prime in c has even total exponent... actually simpler:
    # A number is powerful iff for every prime p | n, p^2 | n.
    # Equivalent: n = a * b^2 where a is squarefree AND every prime in a
    # appears to power >= 2 in n... that's circular.
    #
    # Cleaner generation: powerful numbers are exactly those n where
    # n = prod p_i^{k_i}, k_i >= 2.
    # Generate via: for each square d = m^2, and for each squarefree kernel k
    # supported on the same primes with each exponent at least 1 more... 
    #
    # Simplest correct method: enumerate all n of the form r^2 * s where
    # s is squarefree and every prime dividing s also divides r. Equivalently,
    # n = a^2 * b where b is squarefree and rad(b) | a.
    #
    # Direct enumeration by prime exponents is slow for large L. Use the
    # characterization: n is powerful iff n = x^2 * y where y is squarefree
    # and every prime factor of y divides x.
    #
    # For practical generation up to L: iterate over all squares s^2 <= L,
    # then multiply by a squarefree number t such that every prime of t divides s.
    #
    # Alternative pragmatic approach used in literature: generate by
    # iterating over all "strongly powerful" forms. But for our range we
    # can use: powerful(n) iff for the squarefree part sf(n), sf(n) | isqrt(n)^2.
    #
    # Let's use a direct sieve-like approach for moderate L, and for large L
    # use the parametric form: n = a * b^2 where a is a "2-rough" adjustment.
    #
    # For our purposes, L will be up to ~10^8-10^9. Use the form:
    # n = c * m^2, where c is squarefree and rad(c) | m.
    # Equivalently, let m = prod p_i^{e_i}, then c can include any subset of
    # the primes p_i to odd additional powers (i.e., c's primes are a subset of m's primes).
    # So n = prod p_i^{2e_i + f_i} where f_i in {0,1} and the set of primes with f_i=1
    # is a subset of those with e_i >= 1.
    #
    # For generation: iterate m from 1..isqrt(L), factor m to get its primes,
    # then for each subset of those primes, multiply m^2 by the product of the subset.
    # This is feasible if m is up to ~31623 (for L=10^9).

    result = set()
    limit_sqrt = isqrt(L)
    for m in range(1, limit_sqrt + 1):
        m2 = m * m
        if m2 > L:
            break
        # Factor m to get distinct primes
        mm = m
        primes = []
        d = 2
        while d * d <= mm:
            if mm % d == 0:
                primes.append(d)
                while mm % d == 0:
                    mm //= d
            d += 1 if d == 2 else 2
        if mm > 1:
            primes.append(mm)
        # For each subset of primes, multiply m^2 by the product
        # Use bitmask for small number of primes
        k = len(primes)
        if k <= 12:
            for mask in range(1 << k):
                val = m2
                ok = True
                for i in range(k):
                    if mask & (1 << i):
                        val *= primes[i]
                        if val > L:
                            ok = False
                            break
                if ok and val <= L:
                    result.add(val)
        else:
            # m has many distinct primes; only m^2 itself is safe (and maybe a few)
            result.add(m2)
            # Add m^2 * p for single primes if <= L
            for p in primes:
                v = m2 * p
                if v <= L:
                    result.add(v)
    result.add(1)
    return sorted(result)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    # Strategy: search over N in a range, counting representations as sum of two powerful numbers.
    # We need powerful numbers up to max N. Since N up to 10^18 is too large to enumerate,
    # we restrict to a practical range where we can generate the powerful list.
    #
    # Key insight: the number of powerful numbers up to X is ~ X^{1/2} * (some constant).
    # For X = 10^9, that's roughly 31623 * small factor.
    # For X = 10^12, that's ~10^6.
    #
    # We'll generate powerful numbers up to L_max, then for N up to 2*L_max,
    # count pairs (a, b) with a <= b, a + b = N, both powerful.
    #
    # To maximize representations, larger N tends to help (more room for pairs).
    # But we need N <= 10^18. If L_max = 5*10^17, we can't generate that many.
    #
    # Practical approach: generate powerful numbers up to L_max = 10^9 or so,
    # find the best N in [2, 2*L_max], return it. This gives a valid but modest score.
    #
    # To do better: we can generate powerful numbers up to a larger limit.
    # Let's try L_max = 10^8 first, measure, then scale up if time permits.

    # Try increasing limits until time runs out, keeping best result.
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1

    # Start with a moderate limit and increase
    limits = [10**6, 10**7, 10**8, 3*10**8, 5*10**8, 8*10**8, 10**9]
    
    for L in limits:
        if elapsed() > deadline - 30:
            break
        t0 = time.time()
        powerful = _powerful_numbers_upto(L)
        t_gen = time.time() - t0
        if elapsed() > deadline - 20:
            break
        
        P = len(powerful)
        if P < 2:
            continue
        
        # Convert to numpy for vectorized counting
        pw_arr = np.array(powerful, dtype=np.int64)
        # We want N up to 2*L, count pairs a<=b, a+b=N
        # Use convolution-like approach: for each a in powerful, b = N - a must be in powerful and >= a
        # Efficient: for each a, count how many b >= a with b = N-a in powerful
        # Better: compute all pairwise sums and use histogram? Too expensive for large P.
        #
        # Vectorized approach: for a given N, the count is |{a in powerful : a <= N/2 and N-a in powerful}|
        # We can precompute a boolean array of powerful up to 2L, then for each N slide a window.
        #
        # Even better: use FFT/convolution? powerful_arr * delta function.
        # convolve(indicator, indicator) gives count of pairs (a,b) with a+b=n, but counts ordered pairs.
        # We want unordered with a<=b.
        
        max_N = 2 * L
        # Create indicator array
        indicator = np.zeros(max_N + 1, dtype=np.int8)
        indicator[pw_arr] = 1
        
        # Convolution: conv[n] = sum_a indicator[a] * indicator[n-a]
        # This counts ordered pairs (a, b) with a + b = n.
        # For a <= b: count = (conv[n] + (1 if n is sum of two equal powerfuls)) // 2 ... 
        # Actually: ordered pairs with a+b=n. If a != b, each unordered pair {a,b} with a<b contributes 2.
        # If a = b = n/2, contributes 1.
        # So unordered count with a<=b = (conv[n] + same_count) // 2 where same_count = 1 if n/2 is powerful and n even.
        
        # Use FFT-based convolution
        n_fft = 1
        while n_fft < 2 * (max_N + 1):
            n_fft *= 2
        
        ind_f = np.fft.rfft(indicator, n=n_fft)
        conv = np.fft.irfft(ind_f * ind_f, n=n_fft)
        conv = np.round(conv[:max_N + 1]).astype(np.int64)
        
        t_conv = time.time() - t0
        if elapsed() > deadline - 15:
            break
        
        # For each n, compute unordered pairs a<=b
        # same[n] = 1 if n is even and n//2 is powerful
        half = np.arange(max_N + 1) // 2
        even_mask = (np.arange(max_N + 1) % 2 == 0)
        # Check if half values are in powerful set
        # Create a set for O(1) lookup, or use the indicator
        same = np.zeros(max_N + 1, dtype=np.int64)
        even_indices = np.arange(2, max_N + 1, 2)
        half_vals = even_indices // 2
        same[even_indices] = indicator[half_vals]
        
        # Unordered count with a <= b:
        # For a < b: contributes 1, and in conv it's 2, so (conv[n] - same[n]) // 2 + same[n] = (conv[n] + same[n]) // 2
        pair_counts = (conv + same) // 2
        # Exclude n < 2
        pair_counts[:2] = 0
        
        # Find best N
        # Weight toward larger N? The task says maximize the count. Let's just take max count,
        # and among ties, take larger N.
        max_count = int(np.max(pair_counts))
        if max_count > best_count or (max_count == best_count and int(np.argmax(pair_counts)) > best_N):
            # Find all N with max count, take the largest
            candidates = np.where(pair_counts == max_count)[0]
            if len(candidates) > 0:
                best_N = int(candidates[-1])
                best_count = max_count
                # Generate pairs
                best_pairs = []
                for a in powerful:
                    b = best_N - a
                    if a > b:
                        break
                    if a + b == best_N and 1 <= a <= L and 1 <= b <= L:
                        # Check b is powerful (should be, but verify)
                        if b in set(powerful):
                            best_pairs.append([a, b])
        
        print(f"L={L}, P={P}, best_N={best_N}, best_count={best_count}, time={time.time()-start_time:.1f}s")
    
    # Safety: ensure we have valid output
    if best_count < 1:
        best_N = 2
        best_pairs = [[1, 1]]
    
    # Verify the result
    final_pairs = []
    pw_set = set()
    # Regenerate powerful numbers up to best_N for verification
    # Actually, we can use the fact that a and b must be powerful.
    # For efficiency, just trust the construction but do a spot check.
    for a, b in best_pairs:
        final_pairs.append([int(a), int(b)])
    
    # Ensure distinct pairs with a <= b
    final_pairs = sorted(set((min(a,b), max(a,b)) for a, b in final_pairs))
    final_pairs = [[a, b] for a, b in final_pairs if a + b == best_N]
    
    if not final_pairs:
        best_N = 2
        final_pairs = [[1, 1]]
    
    return {'N': int(best_N), 'pairs': [[int(a), int(b)] for a, b in final_pairs]}
