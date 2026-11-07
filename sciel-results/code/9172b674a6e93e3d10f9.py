"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    # Use Python's built-in isqrt if available, otherwise implement
    try:
        return math.isqrt(n)
    except AttributeError:
        # Fallback implementation
        x = int(n ** 0.5)
        while (x + 1) * (x + 1) <= n:
            x += 1
        while x * x > n:
            x -= 1
        return x

def _prime_sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _factorize_with_primes(n, primes):
    """Return list of primes that divide n an odd number of times."""
    result = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count % 2 == 1:
            result.append(p)
    if temp > 1:
        result.append(temp)
    return result

def construct():
    n_base = 210
    max_size = 100
    max_value = 10210
    
    # Primes up to sqrt(10210) ~ 101
    primes = _prime_sieve(200)
    
    # Parity vector of n_base = 210 = 2 * 3 * 5 * 7
    # So target parity is {2:1, 3:1, 5:1, 7:1}
    target = frozenset([2, 3, 5, 7])
    
    # We want to find a subset S of integers in (210, M] such that:
    # 1. |S| <= 100
    # 2. All distinct
    # 3. Product(S) * 210 is a perfect square
    #    <=> XOR of parity vectors of elements in S = parity vector of 210
    # 4. Minimize M - 210 where M = max(S)
    
    # Binary search on M
    lo = 211
    hi = max_value
    
    best_result = None
    best_m = None
    
    def find_subset_for_m(m):
        """Find a subset of integers in (210, m] with XOR = target, size <= 100.
        Returns the subset or None."""
        if m <= 210:
            return None
        
        candidates = list(range(211, m + 1))
        if len(candidates) > max_size:
            # We can only pick at most 100, but we need to be careful
            # Actually, we should just work with all candidates and find a small subset
            pass
        
        # Compute parity vectors
        # Map each prime to an index for GF(2) representation
        # First, collect all primes that appear
        all_primes_set = set()
        parity_vectors = []
        for num in candidates:
            pf = _factorize_with_primes(num, primes)
            all_primes_set.update(pf)
            parity_vectors.append(frozenset(pf))
        
        # Include target primes in the set
        all_primes_set.update([2, 3, 5, 7])
        
        # Now we need to find a subset of parity_vectors whose XOR = target
        # This is a linear algebra problem over GF(2)
        # Represent each vector as a bitset (integer)
        
        prime_list = sorted(all_primes_set)
        prime_to_idx = {p: i for i, p in enumerate(prime_list)}
        num_primes = len(prime_list)
        
        def to_bitset(fset):
            bits = 0
            for p in fset:
                if p in prime_to_idx:
                    bits |= (1 << prime_to_idx[p])
            return bits
        
        target_bits = to_bitset(target)
        
        # Gaussian elimination to find a subset
        # We want to find indices i1, i2, ..., ik such that XOR of vectors = target
        # This is equivalent to: target is in the span of the vectors
        
        # Build augmented system: we want to solve A*x = b where b = target
        # But we want the solution with minimum number of 1s (sparse solution)
        # That's NP-hard in general, but for small dimensions it's feasible
        
        # Alternative: since we just need ANY subset (size <= 100), and typically
        # the dimension is small, we can use Gaussian elimination to find one solution
        
        # Let's do standard Gaussian elimination on the matrix with target as last column
        # Rows: parity vectors + target
        # We'll perform row reduction
        
        rows = [to_bitset(v) for v in parity_vectors]
        rows.append(target_bits)  # Last row is the target
        
        n_rows = len(rows)
        
        # Gaussian elimination over GF(2)
        # We want to find if target is in the span of the other rows
        # Equivalently, after row reduction, the target row should reduce to 0
        
        # Let's do elimination:
        mat = rows[:]
        rank = 0
        for col in range(num_primes):
            # Find pivot row
            pivot = -1
            for r in range(rank, n_rows):
                if (mat[r] >> col) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate
            for r in range(n_rows):
                if r != rank and ((mat[r] >> col) & 1):
                    mat[r] ^= mat[rank]
            rank += 1
        
        # After elimination, check if the target row (last row) is all zeros
        # If so, target is in the span
        if mat[-1] != 0:
            return None
        
        # Now we need to find the actual subset
        # Re-do elimination while tracking the linear combinations
        
        # Let's use a different approach: track which original vectors combine to give each row
        # This is more complex. Let's use a simpler method:
        
        # Since the dimension is at most ~100 (primes up to 200), and we have up to 10000 candidates,
        # we can use a meet-in-the-middle or just find any solution via backtracking on the reduced system
        
        # Actually, let's just find the solution by back-substitution from the row-reduced form
        # But tracking the linear combinations is tricky.
        
        # Alternative simpler approach: since we just need ANY valid subset, and the problem size is manageable,
        # let's try a greedy/heuristic approach or use the fact that for small M, the solution is small
        
        # Let me try a different tactic: for each M, just check if there's a solution by solving
        # the linear system and if yes, extract the solution
        
        # Let's redo with tracking
        # We'll maintain for each row in the reduced matrix, which original vectors (by index) are XORed to produce it
        
        mat2 = [to_bitset(v) for v in parity_vectors]
        mat2.append(target_bits)
        
        # Track the combination: comb[r] is a bitset of original indices
        comb = []
        for i in range(len(parity_vectors)):
            comb.append(1 << i)
        comb.append(0)  # Target starts with no combination
        
        n_rows2 = len(mat2)
        rank2 = 0
        for col in range(num_primes):
            pivot = -1
            for r in range(rank2, n_rows2):
                if (mat2[r] >> col) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            mat2[rank2], mat2[pivot] = mat2[pivot], mat2[rank2]
            comb[rank2], comb[pivot] = comb[pivot], comb[rank2]
            for r in range(n_rows2):
                if r != rank2 and ((mat2[r] >> col) & 1):
                    mat2[r] ^= mat2[rank2]
                    comb[r] ^= comb[rank2]
            rank2 += 1
        
        # Check if target is in span
        if mat2[-1] != 0:
            return None
        
        # The combination for the target row gives us the subset
        # But wait, the target row was added as the last row, and if it reduced to 0,
        # then comb[-1] tells us which original vectors XOR to give the target
        
        subset_indices = []
        c = comb[-1]
        for i in range(len(parity_vectors)):
            if (c >> i) & 1:
                subset_indices.append(i)
        
        if len(subset_indices) == 0:
            return None
        
        if len(subset_indices) > max_size:
            # Need a smaller subset; this is harder
            # For now, return None and try larger M
            return None
        
        subset = [candidates[i] for i in subset_indices]
        
        # Verify
        prod = 1
        for x in subset:
            prod *= x
        prod *= 210
        s = _isqrt(prod)
        if s * s != prod:
            return None
        
        return subset
    
    # Binary search on M
    for m in range(211, max_value + 1):
        result = find_subset_for_m(m)
        if result is not None:
            best_result = result
            best_m = m
            break
    
    if best_result is None:
        # Fallback: use the seed solution
        return [4 * 210]
    
    # Ensure types are correct
    return list(best_result)
# EVOLVE-BLOCK-END
