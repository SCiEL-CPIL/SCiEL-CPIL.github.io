import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Primes in [2, 128]
    is_prime = [True] * 129
    is_prime[0] = is_prime[1] = False
    for i in range(2, 11):
        if is_prime[i]:
            for j in range(i*i, 129, i):
                is_prime[j] = False
    primes = [i for i in range(2, 129) if is_prime[i]]
    composites = [i for i in range(2, 129) if not is_prime[i]]
    
    # Precompute prime factorization as frozenset of (prime, exponent) pairs
    def factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                count = 0
                while n % d == 0:
                    count += 1
                    n //= d
                factors.append((d, count))
            d += 1
        if n > 1:
            factors.append((n, 1))
        return frozenset(factors)
    
    # Precompute factorizations for all numbers 2-128
    factor_cache = {}
    for n in range(2, 129):
        factor_cache[n] = factorize(n)
    
    # Product of a consecutive block as a factorization
    # Represent as frozenset of (prime, total_exponent)
    def block_factors(seq, i, j):
        # Product of seq[i..j]
        counts = defaultdict(int)
        for k in range(i, j+1):
            for p, e in factor_cache[seq[k]]:
                counts[p] += e
        return frozenset(counts.items())
    
    def is_valid_sequence(seq):
        """Check if all consecutive block products are distinct."""
        products = set()
        n = len(seq)
        for i in range(n):
            for j in range(i, n):
                bf = block_factors(seq, i, j)
                if bf in products:
                    return False
                products.add(bf)
        return True
    
    # Better validity check using incremental product tracking
    def check_validity(seq):
        """Check validity, returns (is_valid, set_of_products) for incremental use."""
        products = set()
        n = len(seq)
        for i in range(n):
            # Build products ending at each position starting from i
            counts = defaultdict(int)
            for j in range(i, n):
                for p, e in factor_cache[seq[j]]:
                    counts[p] += e
                bf = frozenset(counts.items())
                if bf in products:
                    return False, None
                products.add(bf)
        return True, products
    
    # Greedy construction with backtracking
    # Start with all primes, try to insert composites
    best_seq = list(primes)
    best_len = len(primes)
    
    # Sort composites by a heuristic: prefer those with fewer/smaller prime factors
    # to reduce collision risk
    def composite_key(x):
        f = factor_cache[x]
        total_exp = sum(e for _, e in f)
        max_p = max(p for p, _ in f)
        return (total_exp, max_p, x)
    
    composites_sorted = sorted(composites, key=composite_key)
    
    # Multiple restart strategies
    rng = random.Random(123)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Strategy: start with primes, insert composites greedily
        current = list(primes)
        current_products = set()
        
        # Compute initial products for prime sequence
        n = len(current)
        for i in range(n):
            counts = defaultdict(int)
            for j in range(i, n):
                for p, e in factor_cache[current[j]]:
                    counts[p] += e
                current_products.add(frozenset(counts.items()))
        
        # Try inserting each composite
        for c in composites_sorted:
            if time.time() > deadline:
                break
            
            # Find valid insertion positions
            valid_positions = []
            n = len(current)
            for pos in range(n + 1):
                # Check strictly increasing
                if pos > 0 and current[pos-1] >= c:
                    continue
                if pos < n and current[pos] <= c:
                    continue
                
                # Check product collisions
                # New blocks: those that include c
                # Old blocks remain unchanged
                # We need to check:
                # 1. New blocks (containing c) don't collide with old blocks
                # 2. New blocks don't collide with each other
                
                # Efficient check: compute all block products in new sequence
                # that involve the new element, and check against existing + each other
                new_products = set()
                valid = True
                n_new = n + 1
                
                # Blocks ending at positions >= pos in new sequence
                # Actually, let's just do a focused check
                # For each starting position i in [0, pos] and ending position j in [pos, n-1]
                # (using new indexing where c is at pos)
                
                # In new sequence, c is at index pos
                # A block [i, j] in new sequence contains c if i <= pos <= j
                # i ranges from 0 to pos, j ranges from pos to n-1 (new indexing)
                # But j in new indexing: if j >= pos, the actual element is:
                #   new_idx < pos: same as old_idx
                #   new_idx == pos: c
                #   new_idx > pos: old_idx - 1
                
                # Let's compute products of blocks containing c
                # Block from i (0<=i<=pos) to j (pos<=j<=n-1 in new indexing)
                # Elements: current[0..i-1] (if i<pos), c, current[pos..j-1] (if j>pos)
                # Wait, in old indexing:
                #   new index k < pos -> old index k
                #   new index k == pos -> c
                #   new index k > pos -> old index k-1
                
                # Block [i, j] in new indexing where i <= pos <= j:
                # Old elements involved: current[i..pos-1] (if i < pos) and current[pos..j-1] (if j > pos)
                # Plus c
                
                for i in range(0, pos + 1):
                    # Product from i to pos-1 in old indexing (elements before c)
                    # Times c
                    # Times product from pos to j-1 in old indexing (elements after c)
                    
                    # Precompute prefix products of factorizations?
                    # Let's just compute directly for this small case
                    
                    # Product of current[i..pos-1] * c * current[pos..j-1]
                    # For each j from pos to n-1:
                    
                    # First compute product of current[i..pos-1] * c
                    left_counts = defaultdict(int)
                    for k in range(i, pos):  # current[i..pos-1]
                        for p, e in factor_cache[current[k]]:
                            left_counts[p] += e
                    for p, e in factor_cache[c]:
                        left_counts[p] += e
                    
                    # Now extend to the right
                    right_counts = dict(left_counts)
                    for j in range(pos, n):
                        if j > pos:
                            # Add current[j-1] (old indexing)
                            for p, e in factor_cache[current[j-1]]:
                                right_counts[p] += e
                        
                        bf = frozenset(right_counts.items())
                        if bf in current_products or bf in new_products:
                            valid = False
                            break
                        new_products.add(bf)
                        # right_counts is modified in place, so we need to be careful
                        # Actually we're building up, so this is fine
                    
                    if not valid:
                        break
                
                if valid:
                    valid_positions.append(pos)
            
            if valid_positions:
                # Pick a position (prefer middle to spread out?)
                pos = rng.choice(valid_positions)
                # Insert c
                current = current[:pos] + [c] + current[pos:]
                
                # Update current_products
                # Recompute all products (expensive but correct)
                current_products = set()
                n = len(current)
                for i in range(n):
                    counts = defaultdict(int)
                    for j in range(i, n):
                        for p, e in factor_cache[current[j]]:
                            counts[p] += e
                        current_products.add(frozenset(counts.items()))
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = list(current)
    
    # Final verification
    if not is_valid_sequence(best_seq):
        # Fallback to primes
        return primes
    
    return best_seq

if __name__ == "__main__":
    seq = construct()
    print(f"Length: {len(seq)}")
    print(f"Sequence: {seq}")
