import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before 150s kill

# ---------------- Bitset-based verification helpers (exact, fast) ----------------

def _has_K4_bitset(n, bits):
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
                if common2:
                    return True
    return False

def _has_indep_6_bitset(n, bits):
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
    bits = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        bits[i] = m
    if _has_K4_bitset(n, bits):
        return False
    if _has_indep_6_bitset(n, bits):
        return False
    return True

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Paley graph construction (deterministic, structured) ----------------

def _paley_graph(p):
    """Paley graph on p vertices where p is a prime congruent to 1 mod 4."""
    if p < 2 or p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

# ---------------- Greedy edge-addition heuristic ----------------

def _greedy_build(n, rng, time_limit=5.0):
    """
    Greedily add edges to a random sparse graph while maintaining:
    - No K4
    - No independent set of size 6
    """
    start_time = time.time()
    
    # Start with a random graph with moderate density
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    
    # Precompute all potential edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    # Try to add more edges greedily
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        if (bits[u] >> v) & 1:
            continue  # Already present
        
        # Check if adding this edge creates a K4
        # K4 would be formed if u and v have two common neighbors that are adjacent
        common = bits[u] & bits[v]
        k4_created = False
        for c in range(n):
            if not (common >> c) & 1:
                continue
            for d in range(c + 1, n):
                if not (common >> d) & 1:
                    continue
                if (bits[c] >> d) & 1:
                    k4_created = True
                    break
            if k4_created:
                break
        
        if k4_created:
            continue
        
        # Tentatively add edge
        bits[u] |= (1 << v)
        bits[v] |= (1 << u)
        
        # Check if this creates an independent set of size 6
        # This is expensive, so we do a quick check
        if _has_indep_6_bitset(n, bits):
            # Revert
            bits[u] &= ~(1 << v)
            bits[v] &= ~(1 << u)
            continue
    
    # Final verification
    if _has_K4_bitset(n, bits) or _has_indep_6_bitset(n, bits):
        return None
    
    return bits

# ---------------- Local search with targeted improvements ----------------

def _local_search_targeted(n, bits, rng, time_limit=3.0):
    """Local search that targets specific constraint violations."""
    start_time = time.time()
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Check current state
        has_k4 = _has_K4_bitset(n, bits)
        has_ind6 = _has_indep_6_bitset(n, bits)
        
        if not has_k4 and not has_ind6:
            return bits
        
        # Try random edge flips with bias toward fixing violations
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        had_edge = (bits[u] >> v) & 1
        if had_edge:
            bits[u] &= ~(1 << v)
            bits[v] &= ~(1 << u)
        else:
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
        
        new_has_k4 = _has_K4_bitset(n, bits)
        new_has_ind6 = _has_indep_6_bitset(n, bits)
        
        # Accept if we improved the situation
        if (not new_has_k4 and not new_has_ind6):
            continue  # Keep the change
        elif not new_has_k4 and new_has_ind6 and has_k4:
            continue  # Better: fixed K4, still has ind6
        elif new_has_k4 and not new_has_ind6 and has_ind6:
            continue  # Better: fixed ind6, still has K4
        else:
            # Revert
            if had_edge:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            else:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
    
    return bits

# ---------------- Main search strategy ----------------

def _search_structured():
    best = {'n': 0, 'edges': []}
    rng = random.Random(12345)
    
    # Try Paley graphs for various primes
    paley_primes = [5, 13, 17, 29, 37]
    for p in paley_primes:
        if time.time() > _DEADLINE - 10:
            break
        bits = _paley_graph(p)
        if bits is None:
            continue
        if not _has_K4_bitset(p, bits) and not _has_indep_6_bitset(p, bits):
            edges = _edges_from_bits(p, bits)
            if p > best['n']:
                best = {'n': p, 'edges': edges}
    
    # Try greedy construction for larger n
    for n in [33, 36, 37, 38, 39, 40]:
        if time.time() > _DEADLINE - 5:
            break
        for attempt in range(5):
            if time.time() > _DEADLINE - 3:
                break
            bits = _greedy_build(n, rng, time_limit=2.0)
            if bits is not None:
                edges = _edges_from_bits(n, bits)
                if n > best['n']:
                    best = {'n': n, 'edges': edges}
                    break
    
    # Try local search from random starts
    if time.time() < _DEADLINE - 15:
        for n in [36, 37, 38]:
            if time.time() > _DEADLINE - 10:
                break
            for attempt in range(10):
                if time.time() > _DEADLINE - 8:
                    break
                bits = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < 0.45:
                            bits[i] |= (1 << j)
                            bits[j] |= (1 << i)
                
                bits2 = _local_search_targeted(n, bits, rng, time_limit=1.5)
                if not _has_K4_bitset(n, bits2) and not _has_indep_6_bitset(n, bits2):
                    edges = _edges_from_bits(n, bits2)
                    if n > best['n']:
                        best = {'n': n, 'edges': edges}
                        break
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    
    # Start with a known valid baseline (Paley 17)
    best = {'n': 17, 'edges': []}
    bits17 = _paley_graph(17)
    if bits17 is not None and not _has_K4_bitset(17, bits17) and not _has_indep_6_bitset(17, bits17):
        best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
    
    # Try structured search
    try:
        improved = _search_structured()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        # Fallback to Paley 17
        bits17 = _paley_graph(17)
        if bits17 is not None:
            best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
        else:
            best = {'n': 1, 'edges': []}
    
    return best
