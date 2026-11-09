"""Construct maximum pairwise-non-coprime subset of [1,255] containing 255."""
import time
from math import gcd

# Precompute factor masks for each number 1..255
# Primes <= 255
_PRIMES = []
for _x in range(2, 256):
    _is_p = True
    _r = int(_x ** 0.5)
    for _d in range(2, _r + 1):
        if _x % _d == 0:
            _is_p = False
            break
    if _is_p:
        _PRIMES.append(_x)
_NUM_PRIMES = len(_PRIMES)

# Map primes to indices
_PIDX = {p: i for i, p in enumerate(_PRIMES)}

# Factor mask: 16384 bits (2^14) but we only have 54 primes <= 255
# Use a Python int as a bit mask
_FACT_MASK = [0] * 256
for _n in range(2, 256):
    _m = 0
    _x = _n
    for _i, _p in enumerate(_PRIMES):
        if _p > _x:
            break
        if _x % _p == 0:
            _m |= (1 << _i)
        while _x % _p == 0:
            _x //= _p
    _FACT_MASK[_n] = _m

def _validate(lst):
    """Return True if lst is a valid candidate: nonempty, subset of [1,255],
    contains 255, and gcd(a,b)>1 for all distinct pairs."""
    if not lst:
        return False
    s = set(lst)
    if len(s) != len(lst):
        return False  # duplicates
    if 255 not in s:
        return False
    if any(x < 1 or x > 255 for x in s):
        return False
    elems = list(s)
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if gcd(elems[i], elems[j]) == 1:
                return False
    return True

def construct():
    """Return the maximum-cardinality pairwise non-coprime subset of [1,255]
    containing 255.  Uses a greedy maximum-clique search on the compatibility
    graph, with a time-bounded deadline."""
    _t0 = time.time()
    _DEADLINE = _t0 + 75.0  # leave 15 s safety margin before 90 s kill

    # The trivially optimal answer we know: all multiples of 3 (size 85)
    base = list(range(3, 256, 3))
    # base contains 255, all pairs share factor 3, size 85.
    best = base

    # Build compatibility graph: node i (1-indexed, so node = number 1..255)
    # Two nodes are compatible iff gcd(num_i, num_j) > 1.
    # We need a clique containing node 255.
    # Represent neighborhood as bitmask (int).
    # 255 nodes → use Python int.
    compat = [0] * 256  # compat[n] has bit m set iff gcd(n, m) > 1
    for n in range(1, 256):
        for m in range(1, 256):
            if n == m:
                continue
            if gcd(n, m) > 1:
                compat[n] |= (1 << (m - 1))  # 0-indexed bit for number m

    # We search for a maximum clique in the subgraph induced by {255} ∪ N(255).
    # N(255) = all m in [1,255], m != 255, with gcd(255, m) > 1.
    # 255 = 3*5*17.  So m must be divisible by 3, 5, or 17.
    # Build the candidate pool: all numbers in [1,255] that share a factor with 255.
    pool = [n for n in range(1, 256) if n == 255 or _FACT_MASK[n] & _FACT_MASK[255] != 0]
    pool_set = set(pool)
    pool_idx = {n: i for i, n in enumerate(pool)}
    P = len(pool)
    
    # Build adjacency within pool
    adj = [0] * P
    for i, n in enumerate(pool):
        for j, m in enumerate(pool):
            if i != j and gcd(n, m) > 1:
                adj[i] |= (1 << j)
    
    # Ensure 255 is in the pool
    idx_255 = pool_idx[255]
    
    # Greedy + branch-and-bound maximum clique search
    # Start with a greedy seed, then try to grow it.
    # Time-bounded: check deadline periodically.
    
    import random
    
    def _greedy_clique(start_set):
        """Grow a clique greedily by adding the vertex with most connections."""
        cliq = set(start_set)
        improved = True
        while improved:
            improved = False
            # Candidates: pool nodes not in cliq, compatible with all in cliq
            cands = []
            for node in pool:
                if node in cliq:
                    continue
                ci = pool_idx[node]
                ok = True
                for cn in cliq:
                    cj = pool_idx[cn]
                    if not (adj[ci] >> cj) & 1:
                        ok = False
                        break
                if ok:
                    # Score: number of neighbors already in clique (or in pool compatible)
                    deg = 0
                    for cn in cliq:
                        cj = pool_idx[cn]
                        if (adj[ci] >> cj) & 1:
                            deg += 1
                    cands.append((deg, node))
            if not cands:
                break
            cands.sort(reverse=True)
            # Try adding the best candidate
            best_node = cands[0][1]
            cliq.add(best_node)
            improved = True
        return list(cliq)
    
    def _try_expand(start_list, max_iters=50):
        """Attempt to expand a clique by random perturbations + local search."""
        current = list(start_list)
        cur_set = set(current)
        for _it in range(max_iters):
            if time.time() > _DEADLINE:
                break
            # Find all nodes compatible with every node in cur_set
            cands = []
            for node in pool:
                if node in cur_set:
                    continue
                ci = pool_idx[node]
                ok = True
                for cn in cur_set:
                    cj = pool_idx[cn]
                    if not (adj[ci] >> cj) & 1:
                        ok = False
                        break
                if ok:
                    cands.append(node)
            if not cands:
                break
            # Try adding each candidate, pick one at random (or by degree)
            # Use random order for diversity
            random.shuffle(cands)
            added = False
            for node in cands:
                new_set = cur_set | {node}
                if len(new_set) > len(cur_set):
                    # Verify all pairs
                    valid = True
                    lst = list(new_set)
                    for a in range(len(lst)):
                        for b in range(a + 1, len(lst)):
                            if gcd(lst[a], lst[b]) == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cur_set = new_set
                        added = True
                        break
            if not added:
                # Try removing one random element and re-adding a different one
                if len(cur_set) > 1 and random.random() < 0.3:
                    # Find an element whose removal could help
                    # Just try: remove each element, see if we can add more
                    for rem in list(cur_set):
                        trial = cur_set - {rem}
                        # Find candidates for trial
                        tcands = []
                        for node in pool:
                            if node in trial:
                                continue
                            ci = pool_idx[node]
                            ok = True
                            for cn in trial:
                                cj = pool_idx[cn]
                                if not (adj[ci] >> cj) & 1:
                                    ok = False
                                    break
                            if ok:
                                tcands.append(node)
                        if tcands:
                            random.shuffle(tcands)
                            for node in tcands:
                                new_set = trial | {node}
                                lst = list(new_set)
                                valid = True
                                for a in range(len(lst)):
                                    for b in range(a + 1, len(lst)):
                                        if gcd(lst[a], lst[b]) == 1:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid and len(new_set) > len(cur_set):
                                    cur_set = new_set
                                    break
                        if len(cur_set) > len(trial):
                            break
                if not added:
                    break
        return list(cur_set)
    
    # Initial greedy from {255}
    current = _greedy_clique([255])
    if len(current) > len(best):
        best = current
    
    # Local search from current
    expanded = _try_expand(current, max_iters=200)
    if len(expanded) > len(best):
        best = expanded
    
    # Try a few random restarts
    for _restart in range(10):
        if time.time() > _DEADLINE:
            break
        # Random starting set: {255} plus a random compatible node
        compatible_with_255 = [n for n in pool if n != 255]
        if compatible_with_255:
            rand_node = random.choice(compatible_with_255)
            start = [255, rand_node]
            # Greedy
            cur = _greedy_clique(start)
            cur2 = _try_expand(cur, max_iters=100)
            if len(cur2) > len(best):
                best = cur2
    
    # Final validation
    if not _validate(best):
        # Fallback to multiples of 3
        best = list(range(3, 256, 3))
    
    return best
