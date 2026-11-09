import time
import random
import math
from collections import Counter

# Precompute small primes for faster factorization
_SMALL_PRIMES = []
def _init_primes():
    is_prime = [True] * 200
    is_prime[0] = is_prime[1] = False
    for i in range(2, 200):
        if is_prime[i]:
            _SMALL_PRIMES.append(i)
            for j in range(i*i, 200, i):
                is_prime[j] = False
_init_primes()

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            primes.add(p)
            n //= p
    if n > 1:
        primes.add(n)
    return primes

# Cache for prime factorization
_prime_cache = {}

def get_primes_cached(n):
    """Return the set of prime factors of n with caching."""
    if n in _prime_cache:
        return _prime_cache[n]
    result = get_primes(n)
    _prime_cache[n] = result
    return result

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes_cached(s))
    return all_primes

class IncrementalEvaluator:
    def __init__(self, set_a):
        self.n = len(set_a)
        self.a = set_a[:]
        self.pair_primes = [[set() for _ in range(self.n)] for _ in range(self.n)]
        self.prime_counts = Counter()
        self.update_all()

    def update_all(self):
        self.prime_counts.clear()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                s = self.a[i] + self.a[j]
                primes = get_primes_cached(s)
                self.pair_primes[i][j] = primes
                self.pair_primes[j][i] = primes
                for p in primes:
                    self.prime_counts[p] += 1

    def get_count(self):
        return sum(1 for c in self.prime_counts.values() if c > 0)

    def try_move(self, idx, new_val):
        """
        Attempt to change a[idx] to new_val.
        Returns (new_count, old_val) if successful, or None if invalid.
        Modifies internal state if successful.
        """
        if new_val < 1 or new_val > 10000:
            return None
        if new_val == self.a[idx]:
            return None
        
        # Check distinctness
        if new_val in self.a and self.a.index(new_val) != idx:
            return None
            
        old_val = self.a[idx]
        
        # Remove old contributions
        for j in range(self.n):
            if j == idx:
                continue
            s_old = old_val + self.a[j]
            primes_old = self.pair_primes[idx][j]
            for p in primes_old:
                self.prime_counts[p] -= 1
        
        # Set new value
        self.a[idx] = new_val
        
        # Add new contributions
        for j in range(self.n):
            if j == idx:
                continue
            s_new = new_val + self.a[j]
            primes_new = get_primes_cached(s_new)
            self.pair_primes[idx][j] = primes_new
            self.pair_primes[j][idx] = primes_new
            for p in primes_new:
                self.prime_counts[p] += 1
                
        return self.get_count()

    def revert(self, idx, new_val, old_val):
        """Revert a move."""
        # Remove new contributions
        for j in range(self.n):
            if j == idx:
                continue
            primes_new = self.pair_primes[idx][j]
            for p in primes_new:
                self.prime_counts[p] -= 1
        
        # Set old value
        self.a[idx] = old_val
        
        # Add old contributions
        for j in range(self.n):
            if j == idx:
                continue
            s_old = old_val + self.a[j]
            primes_old = get_primes_cached(s_old)
            self.pair_primes[idx][j] = primes_old
            self.pair_primes[j][idx] = primes_old
            for p in primes_old:
                self.prime_counts[p] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    def time_left():
        return deadline - time.time()

    # Initial candidates
    candidates = []
    
    # Strategy 1: Multiples of highly composite numbers
    # If a_i = k_i * M, then a_i + a_j = (k_i + k_j) * M.
    # Primes are factors of M and factors of (k_i + k_j).
    # Choosing M with many small factors helps.
    for base in [6, 12, 30, 60, 120, 210, 420, 840, 2520]:
        nums = []
        k = 1
        while len(nums) < 12:
            val = base * k
            if val > 10000:
                break
            nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)

    # Strategy 2: Residue classes modulo 2*p
    # If a_i = 2*p*k + p, then a_i + a_j = 2*p*(k_i + k_j + 1).
    # Factors are 2, p, and factors of (k_i + k_j + 1).
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        mod = 2 * p
        res = p
        nums = []
        k = 0
        while len(nums) < 12:
            val = mod * k + res
            if val < 1:
                k += 1
                continue
            if val > 10000:
                break
            if val not in nums:
                nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)

    # Strategy 3: Small integers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    candidates.append(list(range(1, 13, 2))) # 1,3,5...
    candidates.append(list(range(2, 14, 2))) # 2,4,6...

    # Strategy 4: Random
    for _ in range(5):
        candidates.append([random.randint(1, 10000) for _ in range(12)])

    best_set = None
    best_count = float('inf')
    
    # Evaluate all candidates
    for c in candidates:
        if time_left() < 5:
            break
        # Ensure distinctness and bounds
        c = [max(1, min(10000, x)) for x in c]
        if len(set(c)) < 12:
            existing = set(c)
            while len(c) < 12:
                v = random.randint(1, 10000)
                if v not in existing:
                    c.append(v)
                    existing.add(v)
        c = c[:12]
        
        primes = evaluate(c)
        if len(primes) < best_count:
            best_count = len(primes)
            best_set = c[:]
    
    if best_set is None:
        return list(range(1, 13))
    
    # Local Search with Incremental Evaluation
    current_set = best_set[:]
    evaluator = IncrementalEvaluator(current_set)
    current_count = evaluator.get_count()
    
    # Precompute a list of "good" deltas to try
    small_deltas = list(range(-50, 51))
    
    no_improve_count = 0
    max_no_improve = 2000
    
    # Precompute a pool of "smooth" candidates for replacement
    # Numbers with few prime factors
    smooth_pool = []
    for i in range(1, 10001):
        if len(get_primes_cached(i)) <= 2:
            smooth_pool.append(i)

    while time_left() > 5:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        r = random.random()
        if r < 0.5:
            # Small perturbation
            new_val = old_val + random.choice(small_deltas)
        elif r < 0.8:
            # Try a smooth number from the pool
            new_val = random.choice(smooth_pool)
        else:
            # Structured move: try to align with current prime factors
            # Just pick a random valid number
            new_val = random.randint(1, 10000)
            
        result = evaluator.try_move(idx, new_val)
        if result is None:
            continue
        
        new_count = result
        
        if new_count < current_count:
            current_count = new_count
            current_set = evaluator.a[:]
            no_improve_count = 0
            if current_count < best_count:
                best_count = current_count
                best_set = current_set[:]
        else:
            # Revert
            evaluator.revert(idx, new_val, old_val)
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart
                no_improve_count = 0
                restart_set = [random.randint(1, 10000) for _ in range(12)]
                # Ensure distinct
                while len(set(restart_set)) < 12:
                    restart_set.append(random.randint(1, 10000))
                restart_set = restart_set[:12]
                evaluator = IncrementalEvaluator(restart_set)
                current_count = evaluator.get_count()
                current_set = evaluator.a[:]
                if current_count < best_count:
                    best_count = current_count
                    best_set = current_set[:]

    return best_set
