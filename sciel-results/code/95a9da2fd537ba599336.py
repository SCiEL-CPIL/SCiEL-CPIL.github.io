"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_MAX_SUM = 20000  # max possible sum of two numbers <= 10000

def _build_spf(n):
    """Build smallest prime factor table up to n."""
    spf = list(range(n + 1))
    if n >= 0:
        spf[0] = 0
    if n >= 1:
        spf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _build_spf(_MAX_SUM)

def _prime_factors_fast(x):
    """Return set of prime factors of x using SPF table."""
    if x <= 1:
        return frozenset()
    factors = set()
    while x > 1:
        p = _SPF[x]
        factors.add(p)
        while x % p == 0:
            x //= p
    return frozenset(factors)

def _score_set(arr, spf=None):
    """Count distinct prime factors across all pairwise sums."""
    if spf is None:
        spf = _SPF
    all_primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            # factor s using spf
            x = s
            while x > 1:
                p = spf[x]
                all_primes.add(p)
                while x % p == 0:
                    x //= p
    return len(all_primes)

def _score_set_incremental(arr, changed_idx, old_val, new_val, spf=None):
    """
    Fast rescore after changing one element.
    Returns (new_score, new_primes_set) for the full set with arr[changed_idx] = new_val.
    We recompute only the pairs involving changed_idx.
    """
    if spf is None:
        spf = _SPF
    n = len(arr)
    # Build primes from pairs NOT involving changed_idx
    all_primes = set()
    for i in range(n):
        if i == changed_idx:
            continue
        ai = arr[i]
        for j in range(i + 1, n):
            if j == changed_idx:
                continue
            s = ai + arr[j]
            x = s
            while x > 1:
                p = spf[x]
                all_primes.add(p)
                while x % p == 0:
                    x //= p
    # Now add pairs involving changed_idx
    nv = new_val
    for i in range(n):
        if i == changed_idx:
            continue
        s = nv + arr[i]
        x = s
        while x > 1:
            p = spf[x]
            all_primes.add(p)
            while x % p == 0:
                x //= p
    return len(all_primes)

def _generate_candidates(max_val=10000):
    """Generate a pool of candidate values."""
    # Focus on multiples of small numbers to encourage shared factors
    candidates = set()
    for base in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 128, 160, 192, 240, 256, 320, 384, 480, 512, 640, 768, 960, 1024, 1280, 1536, 1920, 2048, 2560, 3072, 3840, 4096, 5120, 6144, 7680, 8192, 10000]:
        k = 1
        while base * k <= max_val:
            candidates.add(base * k)
            k += 1
    # Also add some random values
    for _ in range(500):
        candidates.add(random.randint(1, max_val))
    return sorted(candidates)

def _local_search_sa(arr, candidates, spf, deadline, max_iter=50000):
    """Simulated annealing local search."""
    n = len(arr)
    current = list(arr)
    current_score = _score_set(current, spf)
    best = list(current)
    best_score = current_score
    
    # Precompute candidate set for fast membership
    cand_set = set(candidates)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        # Pick a new value from candidates (not already in set)
        # Try a few random candidates
        new_val = None
        for _ in range(10):
            cv = random.choice(candidates)
            if cv not in cand_set or cv == old_val:
                continue
            # Check if cv is already in current set
            if cv in current:
                continue
            new_val = cv
            break
        if new_val is None:
            # Generate a random value
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
        
        # Compute new score incrementally
        new_score = _score_set_incremental(current, idx, old_val, new_val, spf)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            # Accept
            current[idx] = new_val
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    deadline = time.time() + 80  # safety margin from 90s limit
    spf = _SPF
    
    candidates = _generate_candidates(10000)
    
    best_overall = None
    best_score_overall = float('inf')
    
    # Multiple diverse starting points
    random.seed(42)
    
    starting_sets = []
    
    # Start 1: multiples of 2
    s1 = sorted(random.sample([2*k for k in range(1, 5000)], 12))
    starting_sets.append(s1)
    
    # Start 2: multiples of 4
    s2 = sorted(random.sample([4*k for k in range(1, 2500)], 12))
    starting_sets.append(s2)
    
    # Start 3: multiples of 6
    s3 = sorted(random.sample([6*k for k in range(1, 1667)], 12))
    starting_sets.append(s3)
    
    # Start 4: multiples of 12
    s4 = sorted(random.sample([12*k for k in range(1, 834)], 12))
    starting_sets.append(s4)
    
    # Start 5: powers of 2 based
    s5 = sorted(random.sample([2**k for k in range(1, 14)] + [2**k * m for k in range(1, 10) for m in range(2, 6) if 2**k * m <= 10000], 12))
    starting_sets.append(s5)
    
    # Start 6: small consecutive multiples of 2
    s6 = [2*k for k in range(1, 13)]
    starting_sets.append(s6)
    
    # Start 7: random
    s7 = sorted(random.sample(range(1, 10001), 12))
    starting_sets.append(s7)
    
    # Start 8: multiples of 3
    s8 = sorted(random.sample([3*k for k in range(1, 3334)], 12))
    starting_sets.append(s8)
    
    # Start 9: multiples of 8
    s9 = sorted(random.sample([8*k for k in range(1, 1250)], 12))
    starting_sets.append(s9)
    
    # Start 10: mix of powers of 2
    s10 = sorted(random.sample([1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192], 12))
    starting_sets.append(s10)
    
    # Score each starting set and run SA on the best few
    scored_starts = []
    for ss in starting_sets:
        sc = _score_set(ss, spf)
        scored_starts.append((sc, ss))
    
    scored_starts.sort(key=lambda x: x[0])
    
    # Run SA on top 5 starting points
    num_sa_runs = min(5, len(scored_starts))
    time_per_run = (deadline - time.time()) / max(1, num_sa_runs)
    
    for rank in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        sc, ss = scored_starts[rank]
        run_deadline = min(deadline - 2, time.time() + time_per_run)
        result, result_score = _local_search_sa(ss, candidates, spf, run_deadline)
        if result_score < best_score_overall:
            best_score_overall = result_score
            best_overall = result
    
    if best_overall is None:
        best_overall = scored_starts[0][1]
    
    return best_overall
# EVOLVE-BLOCK-END
