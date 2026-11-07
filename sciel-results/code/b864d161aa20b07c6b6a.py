"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify_solution(selected):
    """Verify that product of selected * 210 is a perfect square."""
    prod = 210
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def _prime_parity_mask(n, prime_to_bit):
    """Return bitmask of primes with odd exponent in n."""
    mask = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1 and d in prime_to_bit:
                mask ^= (1 << prime_to_bit[d])
        d += 1
    if n > 1 and n in prime_to_bit:
        mask ^= (1 << prime_to_bit[n])
    return mask

def _sieve(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _gaussian_elimination_rows(rows, target_mask, num_cols):
    """
    Given a list of (mask, index) rows and a target_mask, find a subset of rows
    whose XOR equals target_mask. Returns the list of indices or None.
    Uses standard GF(2) Gaussian elimination with backtracking to extract the subset.
    """
    if not rows:
        return None
    
    # rows is a list of (mask, original_index)
    # We'll do elimination and track which rows combine to form each pivot row
    n = len(rows)
    
    # Build the augmented system: we want to find x such that A*x = target
    # where A is the matrix of row masks, and x is a 0/1 vector of length n
    # We'll use standard Gaussian elimination over GF(2)
    
    # Create list of [mask, index] pairs
    # We'll process rows one by one, maintaining a basis
    
    basis = {}  # pivot_col -> (mask, list_of_row_indices)
    
    for i, (mask, idx) in enumerate(rows):
        m = mask
        combo = [i]  # track which original rows combine to form this
        for col in range(num_cols - 1, -1, -1):
            if m & (1 << col):
                if col in basis:
                    # Eliminate: XOR with basis[col]
                    bm, bcombo = basis[col]
                    m ^= bm
                    # Update combo: symmetric difference
                    combo_set = set(combo)
                    bcombo_set = set(bcombo)
                    combo = list(combo_set.symmetric_difference(bcombo_set))
                else:
                    # New pivot
                    basis[col] = (m, combo)
                    break
        # If m becomes 0, the row is dependent - we could use it but skip for simplicity
    
    # Now check if target can be represented
    t = target_mask
    result_combo = []
    for col in range(num_cols - 1, -1, -1):
        if t & (1 << col):
            if col in basis:
                bm, bcombo = basis[col]
                t ^= bm
                result_combo_set = set(result_combo)
                bcombo_set = set(bcombo)
                result_combo = list(result_combo_set.symmetric_difference(bcombo_set))
            else:
                return None  # Cannot represent target
    
    if t != 0:
        return None
    
    if not result_combo:
        return None
    
    # Convert back to original indices
    indices = [rows[i][1] for i in result_combo]
    return indices

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Known good fallback
    best_raw = 30
    best_solution = [224, 240]
    
    # Generate primes up to a large bound
    max_search = 5000
    primes = _sieve(max_search)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Target: sf(210) = 2*3*5*7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_bit:
            target_mask |= (1 << prime_to_bit[p])
    
    # Precompute parity masks for numbers 211..max_search
    masks = {}
    for i in range(211, max_search + 1):
        masks[i] = _prime_parity_mask(i, prime_to_bit)
    
    # Phase 1: Binary search on M using GF(2) Gaussian elimination
    # For a given M, check if there exists a subset of {211, ..., M} whose XOR of masks = target_mask
    # The subset must be non-empty and have size <= 100
    
    def check_minimal_subset(M):
        """Check if there's a valid subset using numbers in [211, M]. Return (found, subset_indices) or (False, None)."""
        if M < 211:
            return False, None
        
        # Build rows: for each number in [211, M], (mask, number)
        rows = [(masks[i], i) for i in range(211, M + 1)]
        
        # Use Gaussian elimination to find a subset
        indices = _gaussian_elimination_rows(rows, target_mask, num_primes)
        if indices is None:
            return False, None
        
        # Verify the solution
        if len(indices) > 100:
            return False, None
        
        if _verify_solution(indices):
            return True, indices
        return False, None
    
    # Binary search for minimal M
    lo, hi = 211, max_search
    best_M = None
    best_subset = None
    
    while lo <= hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        found, subset = check_minimal_subset(mid)
        if found:
            best_M = mid
            best_subset = subset
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_subset is not None:
        raw = max(best_subset) - 210
        if raw < best_raw:
            best_raw = raw
            best_solution = list(best_subset)
    
    # Phase 2: Refine with direct search for smaller solutions
    # Try 2-element solutions up to a reasonable bound
    upper_bound_2 = min(210 + best_raw, max_search)
    mask_to_numbers = defaultdict(list)
    for i in range(211, upper_bound_2 + 1):
        mask_to_numbers[masks[i]].append(i)
    
    for a in range(211, upper_bound_2 + 1):
        if time.time() > deadline - 15:
            break
        needed_mask = masks[a] ^ target_mask
        if needed_mask in mask_to_numbers:
            for b in mask_to_numbers[needed_mask]:
                if b > a:
                    raw = b - 210
                    if raw < best_raw:
                        if _verify_solution([a, b]):
                            best_raw = raw
                            best_solution = [a, b]
    
    # Phase 3: Try 3-element solutions
    upper_bound_3 = min(210 + best_raw - 1, max_search)
    if upper_bound_3 > 211:
        mask_to_numbers3 = defaultdict(list)
        for i in range(211, upper_bound_3 + 1):
            mask_to_numbers3[masks[i]].append(i)
        
        for a in range(211, upper_bound_3 + 1):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, upper_bound_3 + 1):
                if time.time() > deadline - 10:
                    break
                needed_mask = masks[a] ^ masks[b] ^ target_mask
                if needed_mask in mask_to_numbers3:
                    for c in mask_to_numbers3[needed_mask]:
                        if c > b and c <= upper_bound_3:
                            raw = c - 210
                            if raw < best_raw:
                                if _verify_solution([a, b, c]):
                                    best_raw = raw
                                    best_solution = [a, b, c]
    
    # Phase 4: Try 4-element solutions
    upper_bound_4 = min(210 + best_raw - 1, max_search)
    if upper_bound_4 > 211 and (upper_bound_4 - 211) <= 150:
        mask_to_numbers4 = defaultdict(list)
        for i in range(211, upper_bound_4 + 1):
            mask_to_numbers4[masks[i]].append(i)
        
        for a in range(211, upper_bound_4 + 1):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, upper_bound_4 + 1):
                if time.time() > deadline - 5:
                    break
                for c in range(b + 1, upper_bound_4 + 1):
                    if time.time() > deadline - 5:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                    if needed_mask in mask_to_numbers4:
                        for d in mask_to_numbers4[needed_mask]:
                            if d > c and d <= upper_bound_4:
                                raw = d - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c, d]):
                                        best_raw = raw
                                        best_solution = [a, b, c, d]
    
    # Phase 5: Iterative refinement
    for iteration in range(30):
        if time.time() > deadline - 3:
            break
        upper_bound_iter = min(210 + best_raw - 1, max_search)
        if upper_bound_iter <= 211:
            break
        
        # Try 2-element with tighter bound
        mask_to_numbers_iter = defaultdict(list)
        for i in range(211, upper_bound_iter + 1):
            mask_to_numbers_iter[masks[i]].append(i)
        
        found_improvement = False
        for a in range(211, upper_bound_iter + 1):
            if time.time() > deadline - 3:
                break
            needed_mask = masks[a] ^ target_mask
            if needed_mask in mask_to_numbers_iter:
                for b in mask_to_numbers_iter[needed_mask]:
                    if b > a:
                        raw = b - 210
                        if raw < best_raw:
                            if _verify_solution([a, b]):
                                best_raw = raw
                                best_solution = [a, b]
                                found_improvement = True
                                break
            if found_improvement:
                break
        if found_improvement:
            continue
        
        # Try 3-element
        if (upper_bound_iter - 211) <= 150:
            for a in range(211, upper_bound_iter + 1):
                if time.time() > deadline - 3:
                    break
                for b in range(a + 1, upper_bound_iter + 1):
                    if time.time() > deadline - 3:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ target_mask
                    if needed_mask in mask_to_numbers_iter:
                        for c in mask_to_numbers_iter[needed_mask]:
                            if c > b and c <= upper_bound_iter:
                                raw = c - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c]):
                                        best_raw = raw
                                        best_solution = [a, b, c]
                                        found_improvement = True
                                        break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
        
        # Try 4-element
        if (upper_bound_iter - 211) <= 100:
            for a in range(211, upper_bound_iter + 1):
                if time.time() > deadline - 3:
                    break
                for b in range(a + 1, upper_bound_iter + 1):
                    if time.time() > deadline - 3:
                        break
                    for c in range(b + 1, upper_bound_iter + 1):
                        if time.time() > deadline - 3:
                            break
                        needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                        if needed_mask in mask_to_numbers_iter:
                            for d in mask_to_numbers_iter[needed_mask]:
                                if d > c and d <= upper_bound_iter:
                                    raw = d - 210
                                    if raw < best_raw:
                                        if _verify_solution([a, b, c, d]):
                                            best_raw = raw
                                            best_solution = [a, b, c, d]
                                            found_improvement = True
                                            break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
        
        break
    
    # Final verification
    if not _verify_solution(best_solution):
        best_solution = [224, 240]
        best_raw = 30
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result

# EVOLVE-BLOCK-END
