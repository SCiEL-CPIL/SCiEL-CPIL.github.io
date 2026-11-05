import time
import random
import math
from functools import lru_cache

# EVOLVE-BLOCK-START

def _is_valid(seq):
    """Check all consecutive block products are distinct."""
    seen = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in seen:
                return False
            seen.add(prod)
    return True

def _greedy(order):
    """Greedy construction respecting the given order of candidates."""
    values, products = [], set()
    for x in order:
        # Compute all new block products ending at x
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if products.isdisjoint(tails):
            values.append(x)
            products.update(tails)
    return values

def _score(seq):
    return len(seq)

def _try_improve(seq, time_budget):
    """Local search: try swapping adjacent or inserting to extend."""
    best = list(seq)
    deadline = time.time() + time_budget
    n = len(best)
    
    # Try single removal + re-insertion at best position
    for _ in range(50):
        if time.time() > deadline:
            break
        # Pick a random element to remove
        idx = random.randint(0, n - 1)
        elem = best.pop(idx)
        # Try re-inserting at various positions
        best_len = n - 1
        best_pos = idx
        for pos in range(n - 1, -1, -1):
            trial = best[:pos] + [elem] + best[pos:]
            if _is_valid(trial):
                if len(trial) > best_len:
                    best_len = len(trial)
                    best_pos = pos
        # Also try positions before idx
        for pos in range(idx):
            trial = best[:pos] + [elem] + best[pos:]
            if _is_valid(trial):
                if len(trial) > best_len:
                    best_len = len(trial)
                    best_pos = pos
        if best_len > n - 1:
            best = best[:best_pos] + [elem] + best[best_pos:]
            n = len(best)
        else:
            # Restore
            best.insert(idx, elem)
    
    return best

def _smart_order():
    """Generate a candidate order that favors numbers with unique prime factors."""
    def prime_factor_score(x):
        """Score: prefer numbers with large unique prime factors."""
        s = x
        score = 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]:
            if s % p == 0:
                cnt = 0
                while s % p == 0:
                    s //= p
                    cnt += 1
                # Larger primes contribute more to uniqueness
                score += p * cnt
        return score
    
    candidates = list(range(2, 129))
    # Sort by descending prime factor score (numbers with large primes first)
    candidates.sort(key=lambda x: -prime_factor_score(x))
    return candidates

def _try_multiple_orders(deadline):
    """Try several orderings and keep the best."""
    best = []
    
    # Order 1: natural ascending (baseline)
    seq1 = _greedy(list(range(2, 129)))
    best = seq1
    
    # Order 2: descending
    seq2 = _greedy(list(range(128, 1, -1)))
    if len(seq2) > len(best):
        best = seq2
    
    # Order 3: smart order (large prime factors first)
    smart = _smart_order()
    seq3 = _greedy(smart)
    if len(seq3) > len(best):
        best = seq3
    
    # Order 4: primes first, then composites
    primes = [x for x in range(2, 129) if all(x % p != 0 for p in range(2, int(math.isqrt(x)) + 1))]
    composites = [x for x in range(2, 129) if x not in primes]
    seq4 = _greedy(primes + composites)
    if len(seq4) > len(best):
        best = seq4
    
    # Order 5: descending primes then descending composites
    seq5 = _greedy(sorted(primes, reverse=True) + sorted(composites, reverse=True))
    if len(seq5) > len(best):
        best = seq5
    
    # Order 6: random permutations (limited)
    candidates = list(range(2, 129))
    for trial in range(20):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        seq = _greedy(candidates)
        if len(seq) > len(best):
            best = seq
    
    return best

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin for 90s limit
    
    # Phase 1: Try multiple greedy orderings
    best = _try_multiple_orders(deadline - 20.0)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 5.0:
        improved = _try_improve(best, deadline - 5.0 - time.time())
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        values, products = [], set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if products.isdisjoint(tails):
                values.append(x)
                products.update(tails)
        best = values
    
    return best

# EVOLVE-BLOCK-END
