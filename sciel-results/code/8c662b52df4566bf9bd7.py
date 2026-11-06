"""Search for n in [1000,1000000] minimizing distinct small-prime factors across n+1..n+24."""
import time
import math
import random
from functools import lru_cache

# primes strictly below 24
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
P = len(PRIMES)
MAX_N = 1000000
MIN_N = 1000
WINDOW = 24
DEADLINE = 85.0  # safety margin under 90s kill

def _prime_factor_mask(x):
    """Return bitmask of primes strictly below 24 that divide x."""
    m = 0
    for i, p in enumerate(PRIMES):
        if x % p == 0:
            m |= (1 << i)
    return m

def _score(n):
    """Number of distinct primes <24 appearing in any of n+1..n+24."""
    union = 0
    for i in range(1, WINDOW + 1):
        union |= _prime_factor_mask(n + i)
    return bin(union).count('1')

def _score_bitmask(n):
    union = 0
    for i in range(1, WINDOW + 1):
        x = n + i
        for i2, p in enumerate(PRIMES):
            if x % p == 0:
                union |= (1 << i2)
    return union

def _scan_range(start, end, deadline):
    """Scan n in [start, end] for best score. Returns (best_n, best_score)."""
    best_n = None
    best_score = 99
    t0 = time.time()
    for n in range(start, end + 1):
        if time.time() - t0 > deadline:
            break
        union = 0
        # unroll a bit for speed
        for i in range(1, WINDOW + 1):
            x = n + i
            if x % 2 == 0: union |= 1
            if x % 3 == 0: union |= 2
            if x % 5 == 0: union |= 4
            if x % 7 == 0: union |= 8
            if x % 11 == 0: union |= 16
            if x % 13 == 0: union |= 32
            if x % 17 == 0: union |= 64
            if x % 19 == 0: union |= 128
            if x % 23 == 0: union |= 256
        s = bin(union).count('1')
        if s < best_score:
            best_score = s
            best_n = n
            if best_score <= 3:
                # can't do much better than ~3; keep scanning but early-stop if very low
                pass
    return best_n, best_score

def _local_search(n, best_score, deadline):
    """Random walk / hill climbing around n. Returns (best_n, best_score)."""
    t0 = time.time()
    cur_n = n
    cur_s = best_score
    steps = 0
    while time.time() - t0 < deadline:
        steps += 1
        # propose neighbor
        if random.random() < 0.5:
            cand = cur_n + random.randint(-50, 50)
        else:
            cand = cur_n + random.choice([-1, 1, -2, 2, -3, 3, -7, 7, -11, 11])
        if cand < MIN_N or cand > MAX_N:
            continue
        # compute score
        union = 0
        for i in range(1, WINDOW + 1):
            x = cand + i
            if x % 2 == 0: union |= 1
            if x % 3 == 0: union |= 2
            if x % 5 == 0: union |= 4
            if x % 7 == 0: union |= 8
            if x % 11 == 0: union |= 16
            if x % 13 == 0: union |= 32
            if x % 17 == 0: union |= 64
            if x % 19 == 0: union |= 128
            if x % 23 == 0: union |= 256
        s = bin(union).count('1')
        if s < cur_s:
            cur_s = s
            cur_n = cand
        elif s == cur_s and random.random() < 0.3:
            cur_n = cand
        if cur_s <= 3:
            break
    return cur_n, cur_s

def construct():
    start = time.time()
    deadline = DEADLINE - (time.time() - start) - 3.0  # reserve 3s for finalization
    if deadline < 5:
        deadline = 5.0

    best_n = MIN_N
    best_score = _score(best_n)

    # Phase 1: coarse random sampling to find promising regions
    t0 = time.time()
    deadline_phase1 = min(deadline * 0.4, 20.0)
    rng = random.Random(12345)
    samples = 20000
    for _ in range(samples):
        if time.time() - t0 > deadline_phase1:
            break
        n = rng.randint(MIN_N, MAX_N)
        union = 0
        for i in range(1, WINDOW + 1):
            x = n + i
            if x % 2 == 0: union |= 1
            if x % 3 == 0: union |= 2
            if x % 5 == 0: union |= 4
            if x % 7 == 0: union |= 8
            if x % 11 == 0: union |= 16
            if x % 13 == 0: union |= 32
            if x % 17 == 0: union |= 64
            if x % 19 == 0: union |= 128
            if x % 23 == 0: union |= 256
        s = bin(union).count('1')
        if s < best_score:
            best_score = s
            best_n = n
            if best_score <= 3:
                break

    # Phase 2: local search around best
    t0 = time.time()
    deadline_phase2 = min(deadline * 0.4, 25.0)
    cur_n, cur_s = _local_search(best_n, best_score, deadline_phase2)
    if cur_s < best_score:
        best_score = cur_s
        best_n = cur_n

    # Phase 3: targeted scan near best and a few random regions
    t0 = time.time()
    deadline_phase3 = deadline - (time.time() - start) - 3.0
    if deadline_phase3 > 2:
        # scan a window around best_n
        lo = max(MIN_N, best_n - 5000)
        hi = min(MAX_N, best_n + 5000)
        bn, bs = _scan_range(lo, hi, deadline_phase3 * 0.5)
        if bn is not None and bs < best_score:
            best_score = bs
            best_n = bn
        # scan another random region
        if time.time() - start < DEADLINE - 5:
            rn = rng.randint(MIN_N, MAX_N)
            lo2 = max(MIN_N, rn - 2000)
            hi2 = min(MAX_N, rn + 2000)
            bn2, bs2 = _scan_range(lo2, hi2, 5.0)
            if bn2 is not None and bs2 < best_score:
                best_score = bs2
                best_n = bn2

    # Final: ensure valid
    if best_n is None or best_n < MIN_N or best_n > MAX_N:
        best_n = MIN_N
    return best_n
