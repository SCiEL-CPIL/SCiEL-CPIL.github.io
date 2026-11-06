"""Few prime factors across pairwise sums.

Strategy: pick a small candidate prime set S, build a compatibility graph over
integers in [1, 10000] where an edge (a,b) exists iff a+b has no prime factor
outside S, and find a 12-vertex clique. A clique guarantees the union of
pairwise-sum prime factors is a subset of S, i.e. at most |S| distinct primes.
"""
import time
import random
from math import gcd

MAXN = 10000
NEED = 12
DEADLINE = 75.0  # safety margin under the 90s wall-clock kill

# ---------- factorization ----------
_sieve = None

def _get_sieve(limit):
    global _sieve
    if _sieve is None:
        _sieve = bytearray(limit + 1)  # 1 => prime
        _sieve[0:2] = b'\x00\x00'
        for i in range(2, int(limit ** 0.5) + 1):
            if _sieve[i]:
                step = i
                start = i * i
                _sieve[start:limit + 1:step] = b'\x00' * (((limit - start) // step) + 1)
    return _sieve

def _is_smooth(n, primes):
    """Return True if every prime factor of n is in primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

# ---------- candidate prime sets ----------
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def _candidate_prime_sets():
    """Yield prime sets in increasing size (to minimize distinct primes)."""
    # size 2
    for i in range(len(_SMALL_PRIMES)):
        for j in range(i + 1, len(_SMALL_PRIMES)):
            yield (_SMALL_PRIMES[i], _SMALL_PRIMES[j])
    # size 3
    for i in range(len(_SMALL_PRIMES)):
        for j in range(i + 1, len(_SMALL_PRIMES)):
            for k in range(j + 1, len(_SMALL_PRIMES)):
                yield (_SMALL_PRIMES[i], _SMALL_PRIMES[j], _SMALL_PRIMES[k])
    # size 4 (limited combos for time)
    for i in range(len(_SMALL_PRIMES)):
        for j in range(i + 1, len(_SMALL_PRIMES)):
            for k in range(j + 1, len(_SMALL_PRIMES)):
                for l in range(k + 1, len(_SMALL_PRIMES)):
                    yield (_SMALL_PRIMES[i], _SMALL_PRIMES[j], _SMALL_PRIMES[k], _SMALL_PRIMES[l])

# ---------- clique search ----------
def _find_clique(primes, deadline):
    """Find a 12-clique among [1,10000] where edges = smooth sums over primes."""
    # Precompute smooth membership for each integer's contribution.
    # Edge condition: a+b smooth over primes.
    # We build adjacency on the fly using a bitset-ish approach with random sampling
    # and greedy + local search (branch & bound on a sampled subgraph).

    # For speed, restrict to a random sample of candidate integers, then expand.
    # Start with a pool.
    pool_size = 400
    pool = random.sample(range(1, MAXN + 1), pool_size)

    # Precompute smoothness of pair sums within pool
    # adj[i] = set of j > i such that pool[i]+pool[j] is smooth
    adj = [set() for _ in range(pool_size)]
    for i in range(pool_size):
        pi = pool[i]
        for j in range(i + 1, pool_size):
            if _is_smooth(pi + pool[j], primes):
                adj[i].add(j)
                adj[j].add(i)

    # Greedy: pick high-degree nodes, then branch.
    # We'll do a recursive backtracking with ordering by degree.
    best = []

    def backtrack(candidates, current):
        if len(current) == NEED:
            return list(current)
        if time.time() > deadline:
            return None
        if len(candidates) < NEED - len(current):
            return None
        # Order candidates by degree within candidates
        cands = sorted(candidates, key=lambda x: -len(adj[x] & set(candidates)))
        for v in cands:
            if time.time() > deadline:
                return None
            new_current = current + [v]
            # New candidates must be adjacent to v
            new_cands = [c for c in cands if c > v and v in adj[c]]
            # Actually we need all candidates adjacent to v that are after v in ordering
            # Simpler: new_cands = candidates after v that are adjacent to v
            new_cands = [c for c in cands[cands.index(v)+1:] if v in adj[c]]
            result = backtrack(new_cands, new_current)
            if result is not None:
                return result
        return None

    # Try multiple random pools
    for attempt in range(30):
        if time.time() > deadline:
            break
        if attempt == 0:
            # first attempt uses the initial pool
            pool = random.sample(range(1, MAXN + 1), pool_size)
        else:
            pool = random.sample(range(1, MAXN + 1), pool_size)

        adj = [set() for _ in range(pool_size)]
        for i in range(pool_size):
            pi = pool[i]
            for j in range(i + 1, pool_size):
                if _is_smooth(pi + pool[j], primes):
                    adj[i].add(j)
                    adj[j].add(i)

        # Greedy seed
        current = []
        candidates = list(range(pool_size))
        # Sort by degree
        candidates.sort(key=lambda x: -len(adj[x]))
        for v in candidates:
            if len(current) == NEED:
                break
            if all(v in adj[c] for c in current):
                current.append(v)

        if len(current) == NEED:
            return [pool[v] for v in current]

        # Backtrack from greedy seed
        def backtrack2(candidates, current):
            if len(current) == NEED:
                return list(current)
            if time.time() > deadline:
                return None
            if len(candidates) < NEED - len(current):
                return None
            cands = sorted(candidates, key=lambda x: -len(adj[x] & set(candidates)))
            for idx, v in enumerate(cands):
                if time.time() > deadline:
                    return None
                new_current = current + [v]
                new_cands = [c for c in cands[idx+1:] if v in adj[c]]
                result = backtrack2(new_cands, new_current)
                if result is not None:
                    return result
            return None

        result = backtrack2(list(range(pool_size)), [])
        if result is not None:
            return [pool[v] for v in result]

    return None

# ---------- evaluation ----------
def _distinct_prime_count(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            # factor s
            d = 2
            while d * d <= s:
                while s % d == 0:
                    prime_set.add(d)
                    s //= d
                d += 1 if d == 2 else 2
            if s > 1:
                prime_set.add(s)
    return len(prime_set)

# ---------- main ----------
def construct():
    start = time.time()
    deadline = start + DEADLINE
    random.seed(42)

    best_set = list(range(1, 13))
    best_score = _distinct_prime_count(best_set)

    for primes in _candidate_prime_sets():
        if time.time() > deadline:
            break
        result = _find_clique(primes, deadline)
        if result is not None:
            # Verify distinctness and bounds
            if len(set(result)) == NEED and all(1 <= x <= MAXN for x in result):
                score = _distinct_prime_count(result)
                if score < best_score:
                    best_score = score
                    best_set = result
                # If score equals len(primes), we can't do better with this prime set.
                # Try smaller prime sets next (already in order).
                if score <= len(primes):
                    # Good, but keep searching for even smaller prime sets
                    pass

    return best_set
