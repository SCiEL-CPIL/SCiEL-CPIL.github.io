import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

# ---------------- Bitset helpers ----------------

def _has_clique4(n, bits):
    """Return True if graph (bits) contains a K4."""
    for a in range(n):
        ba = bits[a]
        for b in range(a + 1, n):
            if not ((ba >> b) & 1):
                continue
            common = ba & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not ((common >> c) & 1):
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                if common2:
                    return True
    return False

def _find_clique4(n, bits):
    """Return a K4 (a,b,c,d) or None."""
    for a in range(n):
        ba = bits[a]
        for b in range(a + 1, n):
            if not ((ba >> b) & 1):
                continue
            common = ba & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not ((common >> c) & 1):
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if (common2 >> d) & 1:
                        return (a, b, c, d)
    return None

def _has_clique6(n, bits):
    """Return True if graph (bits) contains a K6."""
    for a in range(n):
        ba = bits[a]
        for b in range(a + 1, n):
            if not ((ba >> b) & 1):
                continue
            common = ba & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not ((common >> c) & 1):
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not ((common2 >> d) & 1):
                        continue
                    common3 = common2 & bits[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not ((common3 >> e) & 1):
                            continue
                        common4 = common3 & bits[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        if common4:
                            return True
    return False

def _find_clique6(n, bits):
    """Return a K6 (a,b,c,d,e,f) or None."""
    for a in range(n):
        ba = bits[a]
        for b in range(a + 1, n):
            if not ((ba >> b) & 1):
                continue
            common = ba & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not ((common >> c) & 1):
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not ((common2 >> d) & 1):
                        continue
                    common3 = common2 & bits[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not ((common3 >> e) & 1):
                            continue
                        common4 = common3 & bits[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        for f in range(e + 1, n):
                            if (common4 >> f) & 1:
                                return (a, b, c, d, e, f)
    return None

def _is_valid(n, g_bits):
    """g_bits is adjacency of the graph. Valid iff no K4 in g and no K6 in complement."""
    if n < 1:
        return False
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ g_bits[i]) & ~(1 << i) for i in range(n)]
    if _has_clique4(n, g_bits):
        return False
    if _has_clique6(n, comp):
        return False
    return True

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Baseline: Paley(17) ----------------

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

_BASELINE = _paley_17()

# ---------------- Complement-graph local search ----------------

def _random_start(n, p, rng):
    """Random graph with edge probability p."""
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _degree_constrained_start(n, target_deg, rng):
    """Start with each vertex near target_deg by random edge selection with degree balancing."""
    bits = [0] * n
    deg = [0] * n
    # First pass: random edges
    p = target_deg / max(1, n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
                deg[i] += 1
                deg[j] += 1
    # Second pass: balance degrees by adding/removing edges
    for _ in range(n * 2):
        # Find vertex with too high degree
        hi = max(range(n), key=lambda i: deg[i])
        lo = min(range(n), key=lambda i: deg[i])
        if deg[hi] - deg[lo] < 2:
            break
        if deg[hi] > target_deg and deg[lo] < target_deg:
            # Remove a random edge from hi
            hi_adj = [j for j in range(n) if j != hi and (bits[hi] >> j) & 1]
            if hi_adj:
                v = rng.choice(hi_adj)
                bits[hi] &= ~(1 << v)
                bits[v] &= ~(1 << hi)
                deg[hi] -= 1
                deg[v] -= 1
            # Add a random edge to lo
            lo_non = [j for j in range(n) if j != lo and not ((bits[lo] >> j) & 1)]
            if lo_non:
                v = rng.choice(lo_non)
                bits[lo] |= (1 << v)
                bits[v] |= (1 << lo)
                deg[lo] += 1
                deg[v] += 1
    return bits

def _local_search_complement(n, bits, rng, max_iters=2000):
    """
    Search in the complement graph.
    Violation types:
      - K4 in G (original) -> flip an edge in that K4 (remove edge from G)
      - K6 in complement -> flip an edge in that K6 (add edge to G)
    We work with g_bits (original graph adjacency).
    """
    all_mask = (1 << n) - 1
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Check K4 in G
        k4 = _find_clique4(n, bits)
        if k4 is not None:
            a, b, c, d = k4
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            u, v = pairs[rng.randint(0, 5)]
            if (bits[u] >> v) & 1:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            continue
        
        # Check K6 in complement
        comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
        k6 = _find_clique6(n, comp)
        if k6 is not None:
            a, b, c, d, e, f = k6
            pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
            u, v = pairs[rng.randint(0, 14)]
            if not ((bits[u] >> v) & 1):
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            continue
        
        # No violations: valid!
        return bits
    
    return bits

def _try_n(n, rng, max_restarts=50, max_iters_per_restart=3000):
    """Try to find a valid graph on n vertices via complement local search."""
    target_deg = n // 2  # aim for ~50% edge density
    
    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 5:
            break
        
        # Alternate between random and degree-constrained starts
        if restart % 3 == 0:
            bits = _random_start(n, 0.5, rng)
        elif restart % 3 == 1:
            bits = _degree_constrained_start(n, target_deg, rng)
        else:
            # Slightly biased random
            p = 0.45 + 0.1 * rng.random()
            bits = _random_start(n, p, rng)
        
        result = _local_search_complement(n, bits, rng, max_iters=max_iters_per_restart)
        if _is_valid(n, result):
            return result
    
    return None

def _complement_search():
    """Main search: try increasing n from 18 upward."""
    best = _BASELINE
    rng = random.Random(12345)
    
    # Try n from 18 up to 36
    for n in range(18, 37):
        if time.time() > _DEADLINE - 10:
            break
        
        # Adaptive restarts: more for smaller n, less for larger
        if n <= 20:
            max_restarts = 30
            max_iters = 2000
        elif n <= 25:
            max_restarts = 20
            max_iters = 3000
        elif n <= 30:
            max_restarts = 15
            max_iters = 4000
        else:
            max_restarts = 10
            max_iters = 5000
        
        result = _try_n(n, rng, max_restarts, max_iters)
        if result is not None:
            edges = _edges_from_bits(n, result)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
            # Found valid at n, try to go higher
        else:
            # Didn't find at n, try a few more restarts with different params
            if n >= 25:
                # Give up on this n and try next
                pass
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    best = _BASELINE
    
    try:
        improved = _complement_search()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity
    if not _is_valid(best['n'], best['edges']):
        best = _BASELINE
    
    return best
