import math
import random
import time
import numpy as np

def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    n = len(U)
    # For small sets, use brute force
    if n <= 30 and max_u <= 5000:
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        size_s = len(sums)
        size_d = len(diffs)
        if size_s == 0:
            return 0.0
        ratio = size_d / size_s
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        return 1.0 + log_ratio / log_denom
    
    # For larger sets, use FFT-based approach
    max_u = max(U)
    n = max_u + 1
    f = np.zeros(n, dtype=np.float64)
    for x in U:
        f[x] = 1.0
    
    # Sumset via convolution
    conv = np.convolve(f, f, mode='full')
    size_s = int(np.sum(conv > 0.5))
    
    # Difference set via convolution with flipped array
    f_flip = f[::-1]
    diff_conv = np.convolve(f, f_flip, mode='full')
    size_d = int(np.sum(diff_conv > 0.5))
    
    if size_s == 0:
        return 0.0
    ratio = size_d / size_s
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    return 1.0 + log_ratio / log_denom

def _greedy_sidon_prime(n, start_prime=2):
    """Generate a Sidon set using greedy prime-based construction."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    # Start with [0, 1]
    U = [0, 1]
    if n == 2:
        return U
    
    # Use primes to build Sidon set
    # A Sidon set has all pairwise sums distinct
    # Greedy: add smallest number that maintains Sidon property
    
    def is_sidon(candidate_set):
        sums = set()
        n = len(candidate_set)
        for i in range(n):
            for j in range(i, n):
                s = candidate_set[i] + candidate_set[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Generate primes
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]
    
    primes = gen_primes(100000)
    prime_idx = 0
    
    while len(U) < n:
        # Try next prime as candidate
        if prime_idx >= len(primes):
            # Fallback: try consecutive integers
            candidate = U[-1] + 1
            while True:
                test_set = U + [candidate]
                if is_sidon(test_set):
                    U.append(candidate)
                    break
                candidate += 1
            continue
        
        candidate = primes[prime_idx]
        prime_idx += 1
        
        if candidate <= U[-1]:
            continue
        
        test_set = U + [candidate]
        if is_sidon(test_set):
            U.append(candidate)
    
    return U[:n]

def _greedy_superincreasing(n):
    """Generate a superincreasing sequence (exponential growth)."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    U = [0]
    for i in range(1, n):
        # Each element is greater than sum of all previous
        next_val = 2 ** i
        U.append(next_val)
    
    return U

def _greedy_staircase(n, base=3):
    """Generate a staircase sequence with controlled growth."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    U = [0]
    for i in range(1, n):
        # Growth factor between 2 and 4
        growth = base - i * 0.1
        growth = max(2.0, growth)
        next_val = int(U[-1] * growth) + 1
        U.append(next_val)
    
    return U

def _local_search(U, max_iterations=50000, time_limit=60.0):
    """Local search to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    best_set = U[:]
    best_score = current_score
    
    n = len(U)
    max_u = max(U) if U else 0
    
    # Temperature for simulated annealing
    temperature = 0.1
    
    for iteration in range(max_iterations):
        if time.time() > start_time + time_limit:
            break
        
        # Random perturbation
        op = random.random()
        
        if op < 0.35 and len(U) > 2:
            # Remove a random element (keep 0)
            removable = [x for x in U if x > 0]
            if not removable:
                continue
            idx = random.choice(removable)
            new_U = [x for x in U if x != idx]
            
        elif op < 0.65:
            # Add a random element
            # Try different ranges
            range_choice = random.random()
            if range_choice < 0.3:
                max_range = max_u + random.randint(1, 50)
            elif range_choice < 0.6:
                max_range = max_u + random.randint(1, 200)
            else:
                max_range = max_u + random.randint(1, 1000)
            
            new_elem = random.randint(1, max_range)
            if new_elem not in U:
                new_U = U + [new_elem]
            else:
                continue
                
        elif op < 0.85:
            # Modify a random element
            modifiable = [x for x in U if x > 0]
            if not modifiable:
                continue
            idx = random.choice(modifiable)
            # Larger perturbations
            delta = random.randint(-10, 10)
            new_val = max(1, idx + delta)
            if new_val not in U:
                new_U = [new_val if x == idx else x for x in U]
            else:
                continue
                
        else:
            # Add a small element near the beginning
            if len(U) < 50:
                new_elem = random.randint(1, max(2, U[1] if len(U) > 1 else 2))
                if new_elem not in U:
                    new_U = U + [new_elem]
                else:
                    continue
            else:
                # Remove a large element
                if len(U) > 2:
                    removable = [x for x in U if x > 0]
                    idx = random.choice(removable)
                    new_U = [x for x in U if x != idx]
                else:
                    continue
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            continue
            
        new_score = _score_set(new_U)
        
        # Accept if better, or with probability based on temperature
        if new_score > best_score:
            best_score = new_score
            best_set = new_U[:]
            current_score = new_score
            U = new_U
            # Cool down
            temperature *= 0.999
        elif random.random() < math.exp((new_score - current_score) / max(temperature, 1e-10)):
            current_score = new_score
            U = new_U
    
    return best_set, best_score

def _prime_step_gaps(n, start=0):
    """Build set using prime gaps as step sizes."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]
    
    primes = gen_primes(50000)
    U = [0]
    
    for i in range(1, n):
        # Use prime as step
        if i < len(primes):
            step = primes[i]
        else:
            step = primes[-1] + i
        U.append(U[-1] + step)
    
    return U

def _random_construct(n, max_val, seed=None):
    """Generate a random set of size n."""
    if seed is not None:
        random.seed(seed)
    
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    candidates = list(range(1, max_val + 1))
    if len(candidates) < n - 1:
        return [0] + candidates
    
    sample = random.sample(candidates, n - 1)
    return sorted([0] + sample)

def _scale(U, factor):
    """Scale a set by a factor."""
    return sorted(set([int(x * factor) for x in U]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    random.seed(42)
    
    # Phase 1: Try various constructions
    constructions = [
        ("sidon_prime_5", _greedy_sidon_prime(5)),
        ("sidon_prime_7", _greedy_sidon_prime(7)),
        ("sidon_prime_10", _greedy_sidon_prime(10)),
        ("sidon_prime_15", _greedy_sidon_prime(15)),
        ("superinc_5", _greedy_superincreasing(5)),
        ("superinc_7", _greedy_superincreasing(7)),
        ("staircase_5", _greedy_staircase(5, 3)),
        ("staircase_7", _greedy_staircase(7, 3.5)),
        ("prime_gaps_5", _prime_step_gaps(5)),
        ("prime_gaps_7", _prime_step_gaps(7)),
        ("prime_gaps_10", _prime_step_gaps(10)),
    ]
    
    for name, U in constructions:
        if time.time() > deadline - 10:
            break
        if 0 not in U:
            U = [0] + U
        U = sorted(set(U))
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Phase 2: Random constructions with various sizes
    for n in [4, 5, 6, 7, 8, 10, 12, 15]:
        for max_val in [n*5, n*10, n*20, n*50, n*100]:
            for _ in range(100):
                if time.time() > deadline - 30:
                    break
                U = _random_construct(n, max_val)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Phase 3: Local search from best set
    if time.time() < deadline - 20:
        remaining = min(60, deadline - time.time() - 5)
        refined_set, refined_score = _local_search(best_set, max_iterations=100000, time_limit=remaining)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    # Phase 4: More local search if time permits
    if time.time() < deadline - 10:
        remaining = min(30, deadline - time.time() - 5)
        refined_set, refined_score = _local_search(best_set, max_iterations=50000, time_limit=remaining)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    return best_set
