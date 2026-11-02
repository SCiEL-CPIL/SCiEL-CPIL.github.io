import time
import math
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
        U = sorted(U)
    n = len(U)
    if n <= 1:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0
    
    # Compute sumset and diffset directly for small sets
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(base)

def _score_set_fft(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
        U = sorted(U)
    n = len(U)
    if n <= 1:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0
    
    if n * n > 200000:
        # For very large sets, use FFT-based approach
        max_val = 2 * M
        size = 1
        while size < 2 * max_val + 1:
            size *= 2
        arr = np.zeros(size, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
        # Sumset via convolution
        conv = np.fft.fft(arr)
        conv_sq = conv * conv
        sumset_counts = np.real(np.fft.ifft(conv_sq))
        sumset_size = np.sum(sumset_counts > 0.5)
        
        # Difference set: shift and correlate
        # a - b where a,b in U. This is like cross-correlation
        # We can compute by: for each a in U, the set {a-b : b in U} = a + (-U)
        # So U - U = U + (-U)
        neg_U = [-x for x in U]
        min_neg = min(neg_U) if neg_U else 0
        offset = -min_neg if min_neg < 0 else 0
        arr2 = np.zeros(size, dtype=np.float64)
        for x in neg_U:
            if x + offset >= 0 and x + offset < size:
                arr2[x + offset] = 1.0
        conv2 = np.fft.fft(arr2)
        conv_diff = np.fft.fft(arr) * np.conj(conv2)
        diff_counts = np.real(np.fft.ifft(conv_diff))
        diffset_size = np.sum(diff_counts > 0.5)
    else:
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sumset_size = len(sumset)
        diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(base)

def _next_pow2(n):
    return 1 << (n - 1).bit_length()

def _prime_factors_distinct(n):
    """Return list of distinct prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _is_squarefree(n):
    """Check if n is squarefree."""
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 1
    return True

def _greedy_gap_multiplication(n_elements, max_search_val=2000):
    """
    Construct a set using gap-multiplication: each gap is a product of distinct primes.
    This ensures differences have unique factorizations, maximizing |U-U|.
    """
    # Start with [0, 1]
    U = [0, 1]
    
    # Precompute primes up to some limit
    def primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    primes = primes_up_to(100)
    
    for i in range(len(U), n_elements):
        if _elapsed() > 40:
            break
        
        best_gap = None
        best_score = -1
        current_max = U[-1]
        
        # Try gaps that are products of 1, 2, or 3 distinct primes
        # The key idea: if gaps are pairwise "independent" in terms of prime factors,
        # then differences will be unique
        
        used_primes = set()
        # Track which primes have been used in gaps so far
        for g in U[1:]:
            for p in _prime_factors_distinct(g):
                used_primes.add(p)
        
        # Try candidate gaps
        candidates = []
        
        # Single prime gaps
        for p in primes:
            if p not in used_primes and current_max + p <= max_search_val:
                candidates.append(p)
        
        # Product of two distinct primes
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                prod = primes[i] * primes[j]
                if prod > max_search_val - current_max:
                    break
                # Prefer if neither prime is heavily used
                score_weight = 0
                if primes[i] not in used_primes:
                    score_weight += 1
                if primes[j] not in used_primes:
                    score_weight += 1
                candidates.append((prod, score_weight))
        
        # Evaluate candidates
        for c in candidates:
            if isinstance(c, tuple):
                gap, weight = c
            else:
                gap, weight = c, 0
            
            if gap <= 0:
                continue
            new_val = current_max + gap
            if new_val > max_search_val:
                continue
            if new_val in U:
                continue
            
            trial = U + [new_val]
            s = _score_set(trial)
            
            # Prefer gaps with more unused prime factors
            adjusted_s = s + weight * 0.001
            
            if adjusted_s > best_score:
                best_score = adjusted_s
                best_gap = gap
        
        if best_gap is not None:
            U.append(U[-1] + best_gap)
        else:
            # Fallback: add a large prime
            for p in primes:
                if p not in used_primes:
                    U.append(U[-1] + p)
                    break
            else:
                U.append(U[-1] + 1)
    
    return U

def _local_search(U, max_iterations=500, timeout=100.0):
    """Local search to improve the set."""
    U = sorted(list(U))
    cur_score = _score_set(U)
    best_U = list(U)
    best_score = cur_score
    rng = np.random.RandomState(42)
    
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if _elapsed() > timeout:
            break
        
        improved = False
        n = len(U)
        max_val = max(U) if U else 1
        
        # Try adding an element
        for _ in range(20):
            if _elapsed() > timeout:
                break
            c = int(rng.randint(1, max_val * 2 + 20))
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try removing an element (keep 0)
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > timeout:
                    break
                trial = U[:idx] + U[idx + 1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try swapping an element
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > timeout:
                    break
                for _ in range(10):
                    if _elapsed() > timeout:
                        break
                    new_val = int(rng.randint(1, max_val * 2 + 20))
                    if new_val not in U:
                        trial = U.copy()
                        trial[idx] = new_val
                        trial = sorted(trial)
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = trial
                            best_U = U
                            best_score = cur_score
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 30:
                # Random restart with perturbation
                U = list(best_U)
                if len(U) > 2:
                    idx = int(rng.randint(1, len(U)))
                    U[idx] = int(rng.randint(1, max_val * 2 + 10))
                    U = sorted(set(U))
                    if 0 not in U:
                        U = [0] + U
                        U = sorted(U)
                    cur_score = _score_set(U)
                no_improve_count = 0
    
    return best_U, best_score

def _exhaustive_small(n, max_val):
    """Exhaustive search for small sets."""
    best_U = None
    best_score = -1
    
    for combo in combinations(range(1, max_val + 1), n - 1):
        if _elapsed() > 50:
            break
        U = sorted([0] + list(combo))
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U, best_score

def construct():
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Exhaustive search for small sets
    for n in range(3, 7):
        if _elapsed() > 20:
            break
        max_val = 15 if n <= 4 else 20
        U, s = _exhaustive_small(n, max_val)
        if U is not None and s > best_score:
            best_score = s
            best_U = U
    
    # Phase 2: Gap-multiplication construction
    for target_n in range(5, 12):
        if _elapsed() > 40:
            break
        U = _greedy_gap_multiplication(target_n, max_search_val=500)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 3: Local search from best
    U, s = _local_search(best_U, max_iterations=300, timeout=110.0)
    if s > best_score:
        best_score = s
        best_U = U
    
    # Phase 4: Try a few more gap-multiplication variants with different prime sets
    if _elapsed() < 120:
        for target_n in range(6, 10):
            if _elapsed() > 125:
                break
            # Use different starting points
            for start in [[0, 1], [0, 2], [0, 3], [0, 1, 2]]:
                if _elapsed() > 125:
                    break
                U = list(start)
                # Extend using gap multiplication
                def primes_up_to(limit):
                    sieve = [True] * (limit + 1)
                    sieve[0] = sieve[1] = False
                    for i in range(2, int(limit**0.5) + 1):
                        if sieve[i]:
                            for j in range(i*i, limit + 1, i):
                                sieve[j] = False
                    return [i for i, v in enumerate(sieve) if v]
                
                primes = primes_up_to(80)
                used_primes = set()
                for g in U[1:]:
                    for p in _prime_factors_distinct(g):
                        used_primes.add(p)
                
                for i in range(len(U), target_n):
                    if _elapsed() > 125:
                        break
                    best_gap = None
                    best_s = -1
                    current_max = U[-1]
                    
                    for p in primes:
                        if p not in used_primes and current_max + p <= 300:
                            trial = U + [current_max + p]
                            s_t = _score_set(trial)
                            if s_t > best_s:
                                best_s = s_t
                                best_gap = p
                    
                    if best_gap is None:
                        # Try product of two primes
                        for i2 in range(len(primes)):
                            for j2 in range(i2 + 1, len(primes)):
                                prod = primes[i2] * primes[j2]
                                if prod > 300 - current_max:
                                    break
                                trial = U + [current_max + prod]
                                s_t = _score_set(trial)
                                if s_t > best_s:
                                    best_s = s_t
                                    best_gap = prod
                    
                    if best_gap is not None:
                        U.append(U[-1] + best_gap)
                    else:
                        U.append(U[-1] + 7)
                
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
    
    return best_U
