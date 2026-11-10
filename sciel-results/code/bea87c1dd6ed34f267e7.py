"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _prime_factorization(n):
    """Return prime factorization as a dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute factorizations for all numbers in [2, 128]
_FACTS = {}
for _x in range(2, 129):
    _FACTS[_x] = _prime_factorization(_x)

def _vec_add(a, b):
    """Add two factorization dicts."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result

def _compute_all_product_vecs(seq):
    """Compute all consecutive block product factorization vectors."""
    vecs = set()
    n = len(seq)
    for i in range(n):
        prod_vec = {}
        for j in range(i, n):
            prod_vec = _vec_add(prod_vec, _FACTS[seq[j]])
            vecs.add(frozenset(prod_vec.items()))
    return vecs

def _is_valid_sequence(seq):
    """Check if sequence is strictly increasing and all block products are distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    n = len(seq)
    expected = n * (n + 1) // 2
    vecs = _compute_all_product_vecs(seq)
    return len(vecs) == expected

def _try_insert(seq, x, existing_vecs=None):
    """Try to insert x into seq at the correct position. Return (new_seq, new_vecs) or (None, None)."""
    n = len(seq)
    # Find position where x would go (maintaining strictly increasing)
    pos = 0
    while pos < n and seq[pos] < x:
        pos += 1
    if pos < n and seq[pos] == x:
        return None, None
    # Check bounds
    if pos > 0 and seq[pos-1] >= x:
        return None, None
    if pos < n and seq[pos] <= x:
        return None, None
    
    new_seq = seq[:pos] + [x] + seq[pos:]
    new_vecs = _compute_all_product_vecs(new_seq)
    expected = len(new_seq) * (len(new_seq) + 1) // 2
    if len(new_vecs) == expected:
        return new_seq, new_vecs
    return None, None

def _local_search(seq, deadline, rng):
    """Apply local search with insertions and swaps to improve sequence length."""
    best = list(seq)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        n = len(best)
        available = [x for x in range(2, 129) if x not in best]
        
        # 1. Try to insert any available number
        # Sort available by number of prime factors (fewer = less collision risk)
        available.sort(key=lambda x: sum(_FACTS[x].values()))
        
        for x in available:
            if time.time() >= deadline:
                break
            new_seq, _ = _try_insert(best, x, None)
            if new_seq is not None:
                best = new_seq
                improved = True
                break
        
        if time.time() >= deadline:
            break
            
        # 2. If no insertion worked, try 1-out/1-in swaps
        if not improved:
            found_swap = False
            for i in range(len(best)):
                if time.time() >= deadline:
                    break
                removed = best[i]
                base = best[:i] + best[i+1:]
                for x in available:
                    if time.time() >= deadline:
                        break
                    if x == removed:
                        continue
                    new_seq, _ = _try_insert(base, x, None)
                    if new_seq is not None:
                        best = new_seq
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
            # If a swap was made, try to insert again in next iteration
            if not found_swap:
                # Try 2-out/2-in swaps
                if len(best) >= 4 and time.time() < deadline - 5:
                    for attempt in range(10):
                        if time.time() >= deadline:
                            break
                        i, j = rng.sample(range(len(best)), 2)
                        removed1, removed2 = best[i], best[j]
                        base = [x for k, x in enumerate(best) if k != i and k != j]
                        candidates = [x for x in available if x != removed1 and x != removed2]
                        if len(candidates) < 2:
                            continue
                        c1, c2 = rng.sample(candidates, 2)
                        s1, _ = _try_insert(base, c1, None)
                        if s1 is None:
                            continue
                        s2, _ = _try_insert(s1, c2, None)
                        if s2 is not None:
                            best = s2
                            available.remove(c1)
                            available.remove(c2)
                            available.extend([removed1, removed2])
                            improved = True
                            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    rng = random.Random(42)
    
    # Greedy construction with heuristic: prefer numbers with fewer prime factors
    values = []
    remaining = list(range(2, 129))
    remaining.sort(key=lambda x: sum(_FACTS[x].values()))
    
    for x in remaining:
        if time.time() >= deadline - 10.0:
            break
        new_seq, _ = _try_insert(values, x, None)
        if new_seq is not None:
            values = new_seq
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 2.0:
        improved = _local_search(values, deadline, rng)
        if len(improved) > len(values):
            values = improved
    
    # Random restarts: if stuck, try different initial orderings
    if time.time() < deadline - 5.0:
        for restart in range(5):
            if time.time() >= deadline - 2.0:
                break
            rem = [x for x in range(2, 129) if x not in values]
            rng.shuffle(rem)
            rem.sort(key=lambda x: sum(_FACTS[x].values()))
            trial = list(values)
            for x in rem:
                if time.time() >= deadline - 2.0:
                    break
                new_seq, _ = _try_insert(trial, x, None)
                if new_seq is not None:
                    trial = new_seq
            if len(trial) > len(values):
                values = trial
                if time.time() < deadline - 2.0:
                    improved = _local_search(values, deadline, rng)
                    if len(improved) > len(values):
                        values = improved
    
    return values
# EVOLVE-BLOCK-END
