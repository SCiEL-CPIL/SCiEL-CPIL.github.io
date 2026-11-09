import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128 and their exponent vectors
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Map each number to its prime factorization as a tuple of (prime, exponent)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        """Return exponent vector as tuple of length num_primes."""
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Precompute factorization for all numbers
    factorization = {n: factorize(n) for n in all_numbers}
    
    def add_vecs(a, b):
        """Add two exponent vectors."""
        return tuple(a[i] + b[i] for i in range(num_primes))
    
    def get_block_products(seq):
        """Return set of all consecutive block products as exponent vectors."""
        prods = set()
        n = len(seq)
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                fv = factorization[seq[j]]
                for k in range(num_primes):
                    vec[k] += fv[k]
                t = tuple(vec)
                prods.add(t)
        return prods
    
    def try_insertion_fast(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products.
        Uses incremental product computation for speed."""
        n = len(seq)
        
        # Find valid insertion positions
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            # Quick check: does inserting c at position i create any duplicate?
            # We need to check all blocks that include c
            # Blocks ending at i-1 (before c) and starting at i (after c)
            # New blocks: [i..j] for j >= i, [k..i-1] for k <= i-1, and combined [k..j]
            
            # Compute suffix products from position i-1 backwards
            suffix_prods = {}  # (start, end) -> product vector
            # Actually, let's just build the new sequence and check all products
            # But do it incrementally
            
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            prods = set()
            valid = True
            
            for a in range(m):
                vec = [0] * num_primes
                for b in range(a, m):
                    fv = factorization[new_seq[b]]
                    for k in range(num_primes):
                        vec[k] += fv[k]
                    t = tuple(vec)
                    if t in prods:
                        valid = False
                        break
                    prods.add(t)
                if not valid:
                    break
            
            if valid:
                return new_seq
        
        return None
    
    def build_greedy(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        random.Random(seed_order).shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insertion_fast(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq
    
    def local_search(seq, time_limit=5.0):
        """Try to add more elements to seq via local search."""
        improved = True
        limit_time = time.time() + time_limit
        while improved and time.time() < limit_time and time.time() < deadline:
            improved = False
            used = set(seq)
            remaining = [x for x in composites if x not in used]
            random.shuffle(remaining)
            for c in remaining:
                new_seq = try_insertion_fast(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    improved = True
                    break
        return seq
    
    def backtrack_csp(seq, available, depth, max_depth, time_limit):
        """CSP backtracking: try to extend seq by adding elements from available."""
        if time.time() > time_limit or depth >= max_depth:
            return seq
        
        best = seq[:]
        for idx, c in enumerate(available):
            if time.time() > time_limit:
                break
            new_seq = try_insertion_fast(seq, c)
            if new_seq is not None:
                # Recurse
                remaining = available[:idx] + available[idx+1:]
                result = backtrack_csp(new_seq, remaining, depth + 1, max_depth, time_limit)
                if len(result) > len(best):
                    best = result
        return best
    
    # Phase 1: Multiple greedy builds with local search
    best_seq = build_greedy(42)
    best_len = len(best_seq)
    
    # Try multiple random seeds
    seeds_tried = 0
    while time.time() < deadline - 15:
        seed_val = random.randint(0, 2**31 - 1)
        seeds_tried += 1
        seq = build_greedy(seed_val)
        
        # Local search on this sequence
        seq = local_search(seq, time_limit=2.0)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # Phase 2: CSP backtracking from the best sequence
    if time.time() < deadline - 10:
        used = set(best_seq)
        available = [x for x in composites if x not in used]
        random.shuffle(available)
        
        # Limit available to keep search tractable
        available = available[:25]
        
        # Try different max depths
        for max_depth in [best_len + 1, best_len + 2, best_len + 3, best_len + 4]:
            if time.time() > deadline - 5:
                break
            result = backtrack_csp(best_seq, available, 0, max_depth, deadline - 3)
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]
                # Update available
                used = set(best_seq)
                available = [x for x in composites if x not in used]
                random.shuffle(available)
                available = available[:25]
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2:
        best_seq = local_search(best_seq, time_limit=3.0)
        best_len = len(best_seq)
    
    return best_seq
