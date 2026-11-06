"""Stronger algorithm for dense increasing sequences with unique block products."""
import time
from functools import lru_cache
from itertools import product as iproduct

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _all_block_products(seq):
    """Return the set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _validate(seq):
    """Check if seq is strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return True
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    prods = _all_block_products(seq)
    return len(prods) == len(seq) * (len(seq) + 1) // 2


def _greedy_forward():
    """Greedy: try each x in [2,128], add if all new tail products are fresh."""
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _greedy_prime_first():
    """Greedy but prefer primes first, then composites."""
    primes = [x for x in range(2, 129) if _is_prime(x)]
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    values, products = [], set()
    for x in primes + composites:
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _insert_composites(seq):
    """Try to insert composites into a prime-based sequence at valid positions."""
    seq = list(seq)
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    for c in composites:
        if c in seq:
            continue
        # Find valid insertion positions where seq[i-1] < c < seq[i]
        inserted = False
        for i in range(len(seq) + 1):
            if i > 0 and seq[i - 1] >= c:
                continue
            if i < len(seq) and seq[i] <= c:
                continue
            new_seq = seq[:i] + [c] + seq[i:]
            if _validate(new_seq):
                seq = new_seq
                inserted = True
                break
    return seq


def _dfs_max(start_seq, deadline, max_depth=6):
    """Bounded DFS to extend a sequence by trying insertions at various positions."""
    best = list(start_seq)
    best_len = len(best)
    
    def _search(seq, depth):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if len(seq) > best_len:
            best = list(seq)
            best_len = len(seq)
        if depth >= max_depth:
            return
        
        # Try inserting each unused number at each valid position
        used = set(seq)
        candidates = [x for x in range(2, 129) if x not in used]
        
        for c in candidates:
            if time.time() > deadline:
                return
            for i in range(len(seq) + 1):
                if i > 0 and seq[i - 1] >= c:
                    continue
                if i < len(seq) and seq[i] <= c:
                    continue
                new_seq = seq[:i] + [c] + seq[i:]
                if _validate(new_seq):
                    _search(new_seq, depth + 1)
    
    _search(start_seq, 0)
    return best


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Try multiple strategies and keep the best
    candidates = []
    
    # Strategy 1: Simple greedy
    g1 = _greedy_forward()
    candidates.append(g1)
    
    # Strategy 2: Prime-first greedy
    g2 = _greedy_prime_first()
    candidates.append(g2)
    
    # Strategy 3: Prime-based + composite insertion
    primes = [x for x in range(2, 129) if _is_prime(x)]
    seq3 = _insert_composites(primes)
    candidates.append(seq3)
    
    # Strategy 4: DFS extension from best greedy
    if time.time() < deadline - 10:
        best_start = max(candidates, key=len)
        dfs_result = _dfs_max(best_start, deadline, max_depth=5)
        candidates.append(dfs_result)
    
    # Strategy 5: DFS from prime+composite base
    if time.time() < deadline - 10:
        dfs_result2 = _dfs_max(seq3, deadline, max_depth=4)
        candidates.append(dfs_result2)
    
    # Pick the longest valid result
    best = max(candidates, key=len)
    
    # Final validation
    if not _validate(best):
        # Fallback to simple greedy
        best = _greedy_forward()
    
    return best

# EVOLVE-BLOCK-END
