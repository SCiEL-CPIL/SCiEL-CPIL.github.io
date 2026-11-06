import time
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2).
    1 counts as powerful."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if x > 1:
        return False
    return True

def _generate_powerful(L, primes):
    """Generate all powerful numbers <= L using recursive prime-power products.
    A powerful number is prod p_i^{e_i} with e_i >= 2."""
    powerful = [1]
    for p in primes:
        p2 = p * p
        if p2 > L:
            break
        # For existing powerful numbers, multiply by p^2, p^3, ...
        new_adds = []
        for pw in powerful:
            val = pw * p2
            while val <= L:
                new_adds.append(val)
                val *= p
        powerful.extend(new_adds)
    return sorted(set(powerful))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate small primes up to ~1000 for better coverage
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(1000)
    
    # Generate powerful numbers up to a larger limit
    L = 100_000_000  # 10^8
    powerful = _generate_powerful(L, primes)
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    print(f"Generated {len(powerful)} powerful numbers up to {L}, took {elapsed:.1f}s")
    
    if time.time() > deadline - 20:
        # Fallback with small known result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    n_pwr = len(powerful)
    print(f"Number of powerful numbers: {n_pwr}")
    
    if n_pwr < 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Use np.unique with return_counts on pairwise sums
    # For efficiency, work in chunks of the powerful numbers list
    # We want to find N that maximizes the count of pairs (a,b) with a+b=N, a<=b
    
    # Convert to numpy array for vectorized operations
    pwr_arr = np.array(powerful, dtype=np.int64)
    
    # Use chunked approach: for each chunk of a's, compute all sums a+b
    # and accumulate counts using np.unique
    # This avoids the O(n^2) memory of full pairwise matrix
    
    # First pass: use a subset to quickly find good candidate N values
    # Then verify and expand
    
    # Chunk size: aim for ~10^7 sums per chunk to keep memory reasonable
    chunk_size = min(n_pwr, 50000)
    
    # Accumulate counts using a dictionary approach for top candidates
    # But np.unique is faster, so let's do it in chunks and merge
    
    # Actually, let's use a smarter approach:
    # 1. Take the first K powerful numbers (smallest ones)
    # 2. Compute all pairwise sums among them
    # 3. Use np.unique to find the most common sum N
    # 4. Verify and extract pairs
    
    # For a larger search, we can try multiple windows or the full set in chunks
    
    # Let's try with the full set but in a memory-efficient way
    # Using np.add.outer would be O(n^2) memory which is too large
    # Instead, use chunked computation
    
    # Alternative: since powerful numbers are sparse, use the fact that
    # for a given N, we only need to check a in powerful where N-a is also in powerful
    # So we can sample candidate N values from sums of pairs
    
    # Generate candidate N values from random pairs
    np.random.seed(42)
    n_samples = min(200000, n_pwr * n_pwr // 100)  # sample of pairs
    idx1 = np.random.randint(0, n_pwr, size=n_samples)
    idx2 = np.random.randint(0, n_pwr, size=n_samples)
    candidate_sums = pwr_arr[idx1] + pwr_arr[idx2]
    
    # Find frequent sums
    unique_sums, counts = np.unique(candidate_sums, return_counts=True)
    top_k = min(50, len(unique_sums))
    top_indices = np.argsort(counts)[-top_k:][::-1]
    candidate_Ns = unique_sums[top_indices]
    candidate_counts = counts[top_indices]
    
    print(f"Top candidate N from sampling: {candidate_Ns[:5]}")
    print(f"Estimated counts: {candidate_counts[:5]}")
    
    # Now verify each candidate N by counting actual pairs
    best_N = 2
    best_count = 0
    best_pairs = [[1, 1]]
    
    for N in candidate_Ns:
        if time.time() > deadline - 10:
            break
        N = int(N)
        if N < 2 or N > 2 * L:
            continue
        pairs = []
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
        cnt = len(pairs)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
        print(f"N={N}: {cnt} pairs")
    
    # Also try some structured candidates: multiples of small powerful numbers
    # N = 2*a for small powerful a (gives pair (a,a) if a powerful)
    # More promising: N values that are sums of many small powerful numbers
    
    # Try N values around the "center" of the powerful number distribution
    # The density of powerful numbers ~ c/sqrt(x), so sums concentrate near certain values
    
    # Additional candidates: N = a + b where a,b are from different ranges
    # Let's try N values that are common in the sum distribution
    
    # Use a second, larger sampling pass with different seed
    if time.time() < deadline - 15:
        np.random.seed(123)
        n_samples2 = min(300000, n_pwr * n_pwr // 50)
        idx1 = np.random.randint(0, n_pwr, size=n_samples2)
        idx2 = np.random.randint(0, n_pwr, size=n_samples2)
        candidate_sums2 = pwr_arr[idx1] + pwr_arr[idx2]
        unique_sums2, counts2 = np.unique(candidate_sums2, return_counts=True)
        top_indices2 = np.argsort(counts2)[-top_k:][::-1]
        candidate_Ns2 = unique_sums2[top_indices2]
        
        for N in candidate_Ns2:
            if time.time() > deadline - 5:
                break
            N = int(N)
            if N < 2 or N > 2 * L:
                continue
            if N == best_N:
                continue
            pairs = []
            for a in powerful:
                if a > N // 2:
                    break
                b = N - a
                if b in powerful_set:
                    pairs.append([a, b])
            cnt = len(pairs)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs
            print(f"N={N}: {cnt} pairs (pass 2)")
    
    # Final: try a systematic scan of N values near the best found
    if time.time() < deadline - 5 and best_N > 2:
        # Try N in a range around best_N
        search_range = min(1000, L // 10)
        for N in range(max(2, best_N - search_range), min(2*L + 1, best_N + search_range)):
            if time.time() > deadline - 2:
                break
            pairs = []
            for a in powerful:
                if a > N // 2:
                    break
                b = N - a
                if b in powerful_set:
                    pairs.append([a, b])
            cnt = len(pairs)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs
    
    print(f"Final best N = {best_N}, count = {best_count}")
    
    # Verify pairs
    for a, b in best_pairs:
        assert a + b == best_N
        assert a <= b
        assert _is_powerful(a)
        assert _is_powerful(b)
    
    return {'N': best_N, 'pairs': best_pairs}
