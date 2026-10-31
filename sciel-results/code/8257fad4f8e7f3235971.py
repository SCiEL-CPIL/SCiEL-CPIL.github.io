import time
import math
import random

_START = time.time()
_DEADLINE = _START + 110.0

def _compute_set_sizes(u):
    """Compute |U+U| and |U-U| exactly for small sets."""
    if not u or len(u) < 2:
        return 0, 0
    u_set = set(u)
    sums = set()
    diffs = set()
    for a in u:
        for b in u:
            sums.add(a + b)
            diffs.add(a - b)
    return len(sums), len(diffs)

def _score(u):
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    n_sums, n_diffs = _compute_set_sizes(u)
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _booth_rotation(bits, k):
    """Rotate bits left by k positions."""
    k = k % len(bits)
    return bits[k:] + bits[:k]

def _construct_cyclotomic(n, p):
    """
    Construct a set U from a cyclotomic sequence of length n over Z_p.
    Uses a simple LCG to generate the sequence, then picks the rotation
    that minimizes the span (max value).
    """
    if n < 2:
        return [0]
    if n > p - 1:
        n = p - 1
    
    # Generate a random LCG sequence of length n
    # We'll try multiple seeds to find a good one
    best_u = None
    best_span = float('inf')
    
    for seed in range(min(50, max(10, n // 2))):
        random.seed(seed * 1000 + 42)
        a = random.randint(1, p - 1)
        c = random.randint(0, p - 1)
        x = random.randint(0, p - 1)
        
        seq = []
        for _ in range(n):
            x = (a * x + c) % p
            seq.append(x)
        
        # Try all rotations
        for rot in range(n):
            rotated = _booth_rotation(seq, rot)
            span = max(rotated)
            if span < best_span:
                best_span = span
                best_u = [0] + sorted(rotated)
    
    if best_u is None:
        # Fallback: just use [0, 1, 2, ..., n-1]
        return list(range(n))
    
    return best_u

def _local_search(u, timeout=2.0):
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
        
    current_score = _score(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 3.0:
        improved = False
        # Try each element
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 3.0:
                break
            old_val = u[i]
            # Try a range of new values
            search_range = max(3, max_val // 3)
            candidates = set()
            # Local neighborhood
            for delta in range(-search_range, search_range + 1, max(1, search_range // 10)):
                new_val = old_val + delta
                if new_val >= 0:
                    candidates.add(new_val)
            # Random jumps
            for _ in range(3):
                candidates.add(random.randint(0, max_val + search_range))
                
            for new_val in candidates:
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score(new_u)
                if s > current_score:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Random restart or mutation if stuck
            if random.random() < 0.2:
                idx = random.randint(0, n-1)
                u[idx] = random.randint(0, max_val + 10)
                u = sorted(set(u))
                if len(u) < n:
                    u = best_u[:]
                else:
                    current_score = _score(u)

    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Focus on small n where the score potential is highest
    for n in range(2, 14):
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Try cyclotomic construction with small primes
        # p should be close to n^2 to n^3 for good results
        primes_to_try = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
        for p in primes_to_try:
            if time.time() > _DEADLINE - 10.0:
                break
            u_init = _construct_cyclotomic(n, p)
            if len(u_init) < 2:
                continue
            u_opt, s_opt = _local_search(u_init, timeout=1.5)
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
        
        # Also try random initialization for local search
        for _ in range(2):
            if time.time() > _DEADLINE - 10.0:
                break
            max_span = n * n
            u_rand = [0]
            while len(u_rand) < n:
                v = random.randint(1, max_span)
                if v not in u_rand:
                    u_rand.append(v)
            u_rand = sorted(u_rand)
            u_opt, s_opt = _local_search(u_rand, timeout=1.0)
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]

    return best_u
