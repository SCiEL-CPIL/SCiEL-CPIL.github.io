import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 138.0  # safety margin before 150s kill

# ---------------- Bitset-based verification helpers (exact, fast) ----------------

def _adj_to_bitsets(n, adj):
    """Convert list of sets to list of integers (bitsets)."""
    return [0 if not adj[i] else int(''.join(str(1 if j in adj[i] else 0) for j in range(n-1, -1, -1)), 2) for i in range(n)]

def _has_K4_bitset(n, bits):
    """Check for K4 using bitsets. bits[i] has bit j set iff edge (i,j)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            # common neighbors of a and b, greater than b
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)  # only vertices > b
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                # need d > c in common & bits[c]
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)  # only vertices > c
                if common2:
                    return True
    return False

def _has_indep_6_bitset(n, bits):
    """Check for independent set of size 6 (clique of size 6 in complement)."""
    # complement adjacency: comp[i] has bit j set iff NOT edge (i,j) and i != j
    comp = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp[i] = (all_mask ^ bits[i]) & ~(1 << i)
    
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        if common4:
                            return True
    return False

def _is_valid(n, edges):
    if n < 1:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    bits = _adj_to_bitsets(n, adj)
    if _has_K4_bitset(n, bits):
        return False
    if _has_indep_6_bitset(n, bits):
        return False
    return True

def _edges_from_adj(n, adj):
    out = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                out.append([i, j])
    return out

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Baseline: 17-vertex Paley graph (known valid) ----------------

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

_BASELINE = _paley_17()

# ---------------- Search: randomized constructive + local repair with bitsets ----------------

def _random_graph_bits(n, p, rng):
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _find_K4(n, bits):
    """Return a K4 as tuple (a,b,c,d) or None."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if (common2 >> d) & 1:
                        return (a, b, c, d)
    return None

def _find_indep_6(n, bits):
    """Return an independent 6-set as tuple or None."""
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        for f in range(e + 1, n):
                            if (common4 >> f) & 1:
                                return (a, b, c, d, e, f)
    return None

def _local_search_bits(n, bits, rng, max_iters=3000):
    """Local search: flip edges to eliminate K4s and independent 6-sets."""
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        k4 = _find_K4(n, bits)
        if k4 is None:
            indep = _find_indep_6(n, bits)
            if indep is None:
                return bits
            # Add a random edge within the independent set
            verts = list(indep)
            u, v = verts[rng.randint(0, 5)], verts[rng.randint(0, 5)]
            if u != v:
                if not (bits[u] >> v) & 1:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
        else:
            # Remove a random edge from the K4
            a, b, c, d = k4
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            u, v = pairs[rng.randint(0, 5)]
            if (bits[u] >> v) & 1:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
    
    return bits

def _search_for_larger():
    best = _BASELINE
    rng = random.Random(42)
    
    # Try to reach 36+ by random init + local repair
    for target in [36, 37, 38, 39, 40]:
        if time.time() > _DEADLINE - 8:
            break
        attempts = 0
        while time.time() < _DEADLINE - 5 and attempts < 60:
            attempts += 1
            # Start from moderately dense random graph
            p = rng.uniform(0.35, 0.55)
            bits = _random_graph_bits(target, p, rng)
            bits2 = _local_search_bits(target, bits, rng, max_iters=2000)
            if not _has_K4_bitset(target, bits2) and not _has_indep_6_bitset(target, bits2):
                edges = _edges_from_bits(target, bits2)
                if target > best['n']:
                    best = {'n': target, 'edges': edges}
                break
    
    # Also try improving from a sparser/denser starting point
    if time.time() < _DEADLINE - 10:
        for target in range(36, 41):
            if time.time() > _DEADLINE - 5:
                break
            for p in [0.3, 0.4, 0.5, 0.6]:
                if time.time() > _DEADLINE - 5:
                    break
                bits = _random_graph_bits(target, p, rng)
                bits2 = _local_search_bits(target, bits, rng, max_iters=1500)
                if not _has_K4_bitset(target, bits2) and not _has_indep_6_bitset(target, bits2):
                    edges = _edges_from_bits(target, bits2)
                    if target > best['n']:
                        best = {'n': target, 'edges': edges}
                    break
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    best = _BASELINE
    
    # Spend remaining budget trying to improve
    try:
        improved = _search_for_larger()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        best = _BASELINE
    
    return best
