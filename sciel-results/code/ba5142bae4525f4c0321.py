"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

_start_time = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_primes(arr):
    """Count distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _factorize(s)
    return len(primes), primes

def _generate_smooth_numbers(limit, max_exp=20):
    """Generate 2,3,5,7,11-smooth numbers up to limit."""
    smooths = set()
    # Generate all numbers of form 2^a * 3^b * 5^c * 7^d * 11^e <= limit
    p2 = [1]
    while p2[-1] * 2 <= limit:
        p2.append(p2[-1] * 2)
    p3 = [1]
    while p3[-1] * 3 <= limit:
        p3.append(p3[-1] * 3)
    p5 = [1]
    while p5[-1] * 5 <= limit:
        p5.append(p5[-1] * 5)
    p7 = [1]
    while p7[-1] * 7 <= limit:
        p7.append(p7[-1] * 7)
    p11 = [1]
    while p11[-1] * 11 <= limit:
        p11.append(p11[-1] * 11)
    
    for a in p2:
        for b in p3:
            ab = a * b
            if ab > limit:
                break
            for c in p5:
                abc = ab * c
                if abc > limit:
                    break
                for d in p7:
                    abcd = abc * d
                    if abcd > limit:
                        break
                    for e in p11:
                        val = abcd * e
                        if val > limit:
                            break
                        if val >= 1:
                            smooths.add(val)
    return sorted(smooths)

def _local_search(arr, smooths, max_iters=50000):
    """Local search via swaps to minimize distinct prime factors."""
    n = len(arr)
    best = list(arr)
    best_count, _ = _count_primes(best)
    
    # Precompute smooth set for O(1) lookup
    smooth_set = set(smooths)
    
    iters = 0
    while iters < max_iters and not _deadline_exceeded():
        iters += 1
        # Pick random pair to swap
        i, j = random.sample(range(n), 2)
        # Try swapping arr[i] with a random smooth number
        new_val = random.choice(smooths)
        if new_val in best:
            continue
        
        old_val = best[i]
        best[i] = new_val
        
        new_count, _ = _count_primes(best)
        if new_count < best_count:
            best_count = new_count
            # Occasionally try to improve further
        else:
            best[i] = old_val
            # With some probability, accept equal (for diversity)
            if new_count == best_count and random.random() < 0.1:
                best[i] = new_val
    
    return best, best_count

def _greedy_initial(smooths, n=12):
    """Greedy: start with smallest smooths, then add one that minimizes new primes."""
    # Start with a random subset of smooths
    random.shuffle(smooths)
    initial = smooths[:n]
    return initial

def construct():
    random.seed(42)
    
    smooths = _generate_smooth_numbers(10000)
    if len(smooths) < 12:
        # Fallback
        return list(range(1, 13))
    
    # Try multiple random restarts
    best_arr = None
    best_count = float('inf')
    
    num_restarts = 50
    for r in range(num_restarts):
        if _deadline_exceeded():
            break
        
        # Different initialization strategies
        if r % 3 == 0:
            # Smallest smooths
            arr = smooths[:12]
        elif r % 3 == 1:
            # Random smooths
            arr = random.sample(smooths, 12)
        else:
            # Mix of small and medium smooths
            small = smooths[:len(smooths)//3]
            medium = smooths[len(smooths)//3:2*len(smooths)//3]
            arr = random.sample(small, 6) + random.sample(medium, 6)
        
        arr_count, _ = _count_primes(arr)
        if arr_count < best_count:
            best_count = arr_count
            best_arr = list(arr)
        
        # Local search
        improved, imp_count = _local_search(arr, smooths, max_iters=2000)
        if imp_count < best_count:
            best_count = imp_count
            best_arr = list(improved)
        
        if best_count <= 3:
            break
    
    if best_arr is None:
        return list(range(1, 13))
    
    # Final polish: try single-element replacements
    if not _deadline_exceeded():
        smooth_set = set(smooths)
        for _ in range(10000):
            if _deadline_exceeded():
                break
            i = random.randint(0, 11)
            new_val = random.choice(smooths)
            if new_val in best_arr:
                continue
            old_val = best_arr[i]
            best_arr[i] = new_val
            c, _ = _count_primes(best_arr)
            if c < best_count:
                best_count = c
            else:
                best_arr[i] = old_val
    
    return best_arr
# EVOLVE-BLOCK-END
