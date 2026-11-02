"""Construct a ternary covering code K_3(7,1) via greedy set-cover with bit-parallel masks."""
import time
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 120s kill (evaluator says 150s but be safe)


def _all_words(q, n):
    """Generate all words as flat integers 0..q^n-1."""
    return np.arange(q ** n, dtype=np.int64)


def _digits_flat(w, q, n):
    """Convert flat integer to list of n digits (little-endian: index 0 is least significant)."""
    d = []
    x = w
    for _ in range(n):
        d.append(x % q)
        x //= q
    return d


def _sphere_masks(w, q, n):
    """
    For a word w (flat int), return the set of all words within Hamming distance <= 1.
    This includes w itself and all words differing in exactly one position.
    Returns a Python set of flat ints (for the center word) or we can precompute per-position.
    """
    d = _digits_flat(w, q, n)
    s = {w}
    for i in range(n):
        base = w
        # remove digit i
        mult = q ** i
        val = d[i]
        base_without = base - val * mult
        for v in range(q):
            if v != val:
                s.add(base_without + v * mult)
    return s


def _greedy_cover(q, n, radius, time_limit):
    """
    Greedy set-cover: repeatedly pick the codeword that covers the most uncovered words.
    Uses bit-parallel via numpy boolean arrays for speed.
    """
    N = q ** n
    # Precompute for each possible word w, the list of words it covers (within radius 1)
    # Since radius=1, each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    # Build coverage: for each center w, list of covered word indices
    # We'll use a numpy array of shape (N, 15) to store covered indices, but that's 2187*15 ~ 32k, fine.
    
    # Actually, let's precompute the sphere for each word as a list of flat ints.
    spheres = []
    for w in range(N):
        d = _digits_flat(w, q, n)
        s = [w]
        for i in range(n):
            mult = q ** i
            val = d[i]
            base_without = w - val * mult
            for v in range(q):
                if v != val:
                    s.append(base_without + v * mult)
        spheres.append(s)
    
    # uncovered: boolean array of length N, True if not yet covered
    uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    
    codewords = []
    
    while uncovered_count > 0:
        if time.time() > time_limit:
            break
        
        # For each candidate center w, count how many uncovered words it covers
        # We want to maximize this count.
        # Naive: for each w, sum(uncovered[spheres[w]]). That's N * 15 ops = ~32k, fine per iteration.
        # But we do this for up to ~186 iterations, so ~6M ops total. Fine.
        
        best_w = -1
        best_count = -1
        for w in range(N):
            # Quick check: if w is already covered, we can still pick it as center,
            # but it might be better to only consider uncovered centers? No, a covered center
            # could still cover many uncovered neighbors. But typically greedy picks uncovered ones.
            # Let's consider all w for correctness, but optimize: skip if w is covered AND
            # all its neighbors are covered (then it covers 0 new). Hard to check quickly.
            # Just compute for all.
            cnt = 0
            for idx in spheres[w]:
                if uncovered[idx]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_w = w
        
        if best_count <= 0:
            break  # shouldn't happen if uncovered_count > 0
        
        codewords.append(best_w)
        
        # Mark covered
        for idx in spheres[best_w]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
    
    return codewords


def _remove_redundant(codewords, q, n, radius):
    """Try to remove codewords that are redundant (their coverage is subset of others)."""
    N = q ** n
    # Precompute spheres
    spheres = []
    for w in range(N):
        d = _digits_flat(w, q, n)
        s = [w]
        for i in range(n):
            mult = q ** i
            val = d[i]
            base_without = w - val * mult
            for v in range(q):
                if v != val:
                    s.append(base_without + v * mult)
        spheres.append(s)
    
    m = len(codewords)
    # For each codeword i, check if removing it leaves all its covered words still covered by others.
    # i.e., for every word x in spheres[codewords[i]], there exists j != i such that x in spheres[codewords[j]].
    
    # Build: for each word x, list of codeword indices that cover it.
    coverers = [[] for _ in range(N)]
    for i, w in enumerate(codewords):
        for x in spheres[w]:
            coverers[x].append(i)
    
    removable = []
    for i in range(m):
        w = codewords[i]
        redundant = True
        for x in spheres[w]:
            # Check if any other codeword covers x
            others = [j for j in coverers[x] if j != i]
            if not others:
                redundant = False
                break
        if redundant:
            removable.append(i)
    
    if removable:
        keep = set(range(m)) - set(removable)
        codewords = [codewords[i] for i in sorted(keep)]
    
    return codewords


def _local_improve(codewords, q, n, radius, time_limit):
    """
    Try to improve by swapping a codeword with another that covers more uncovered-after-removal.
    Simple version: try removing each codeword; if still covered, keep removed.
    Then try adding back if needed.
    """
    N = q ** n
    spheres = []
    for w in range(N):
        d = _digits_flat(w, q, n)
        s = [w]
        for i in range(n):
            mult = q ** i
            val = d[i]
            base_without = w - val * mult
            for v in range(q):
                if v != val:
                    s.append(base_without + v * mult)
        spheres.append(s)
    
    # Repeatedly try to remove redundant codewords
    changed = True
    while changed and time.time() < time_limit:
        changed = False
        m = len(codewords)
        # Build coverers
        coverers = [[] for _ in range(N)]
        for i, w in enumerate(codewords):
            for x in spheres[w]:
                coverers[x].append(i)
        
        for i in range(m):
            if time.time() > time_limit:
                break
            w = codewords[i]
            redundant = True
            for x in spheres[w]:
                others = [j for j in coverers[x] if j != i]
                if not others:
                    redundant = False
                    break
            if redundant:
                codewords.pop(i)
                changed = True
                break  # rebuild coverers
    
    return codewords


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 110.0
    
    if q == 3 and n == 7 and radius == 1:
        # Specialized fast path
        N = 3 ** 7  # 2187
        
        # Precompute spheres for all 2187 words
        spheres = []
        for w in range(N):
            d = _digits_flat(w, q, n)
            s = [w]
            for i in range(n):
                mult = q ** i
                val = d[i]
                base_without = w - val * mult
                for v in range(q):
                    if v != val:
                        s.append(base_without + v * mult)
            spheres.append(s)
        
        # Greedy set cover
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        codewords = []
        
        while uncovered_count > 0:
            if time.time() > _DEADLINE:
                break
            
            best_w = -1
            best_count = -1
            for w in range(N):
                cnt = 0
                for idx in spheres[w]:
                    if uncovered[idx]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            
            if best_count <= 0:
                break
            
            codewords.append(best_w)
            for idx in spheres[best_w]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        # Local improvement: remove redundant codewords
        codewords = _local_improve(codewords, q, n, radius, _DEADLINE)
        
        # Convert to lists of digits
        result = []
        for w in codewords:
            d = _digits_flat(w, q, n)
            result.append(d)
        
        # Ensure distinctness
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        
        return unique_result
    else:
        # Generic fallback: direct product construction
        if n <= 4:
            # Brute force or simple
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
        else:
            # Use tetracode-based construction as fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
