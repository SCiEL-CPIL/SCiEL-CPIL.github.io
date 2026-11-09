import time
import random

_START = time.time()
_DEADLINE = _START + 130.0

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

# ---------------- Algebraic Construction over GF(q) ----------------

def _gf_mul(x, y, q, poly=None):
    # Simple GF(2^k) multiplication for small q, but we mostly use prime fields or GF(2^k) for specific cases
    # For simplicity and speed in this context, we will use prime fields p or GF(2^k) if p is not suitable.
    # However, the prompt suggests GF(q) geometry. Let's stick to prime fields p for simplicity and robustness,
    # as Paley graphs are a specific case of this. We can generalize to GF(2^k) if needed.
    # Actually, for n up to 40, prime fields 5, 13, 17, 29, 37 are good.
    # Let's implement a generic GF(p) quadratic form graph.
    pass

def _algebraic_graph_prime(p, rng, seed_offset=0):
    """
    Construct a graph on p vertices using a quadratic form over GF(p).
    Vertices are elements of GF(p).
    Edge (x, y) exists if Q(x-y) is a quadratic residue (or non-residue).
    This is essentially the Paley graph, but we can vary the quadratic form or the subset of vertices.
    
    To get different graphs, we can:
    1. Use different primes.
    2. Use subfields or subsets (e.g., projective plane points).
    3. Use a random linear transformation of the vertex space (isomorphic to Paley).
    4. Use a different quadratic form: Q(x) = ax^2 + bx + c.
    """
    if p < 2 or p % 4 != 1:
        return None
    
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * p
    
    # Randomly permute the "quadratic form" by applying a random affine transformation to the vertex labels
    # This doesn't change the isomorphism class but might help if we were doing local search, 
    # but since we are verifying exactly, it's just a relabeling.
    # However, we can define the edge set based on a random subset of the multiplicative group? No, that breaks symmetry.
    
    # Let's try a different algebraic structure: 
    # Vertices are points in GF(p)^2. 
    # Edge if dot product is non-zero? 
    # This gives a graph on p^2 vertices. 
    # For p=5, n=25. For p=7, n=49 (too big). 
    # Let's try p=5, n=25.
    
    # Actually, the prompt says "mapping vertices to points in a 2D vector space over GF(q)".
    # Let's implement this for q=5, 7, 11, 13.
    # n = q^2.
    # Edge (u, v) if (u-v) is not in a specific subspace or if a quadratic form Q(u-v) != 0.
    
    # Let's stick to the prime field Paley-like graphs first as they are known to be good, 
    # but the current program already does Paley. 
    # The inspiration says "randomized algebraic construction".
    # Let's try graphs based on finite projective planes or other structures.
    
    # Alternative: Use a random graph generated by a random linear code's dual?
    # Or simply: Try many random graphs with specific density and use the fast verifier.
    # But the directive says "Strategy to try... finite field geometry".
    
    # Let's implement the GF(q)^2 construction.
    # Vertices: (x, y) in GF(q) x GF(q). Total q^2 vertices.
    # Edge condition: Q(x1-x2, y1-y2) is a square in GF(q) and non-zero.
    # Q(u, v) = u^2 + v^2 (if -1 is a square in GF(q), i.e., q=1 mod 4).
    
    if q % 4 != 1:
        return None
        
    n = q * q
    bits = [0] * n
    
    # Map (x, y) to index i = x * q + y
    def idx(x, y):
        return x * q + y
        
    squares = {x * x % q for x in range(1, q)}
    
    for i in range(n):
        x1 = i // q
        y1 = i % q
        for j in range(i + 1, n):
            x2 = j // q
            y2 = j % q
            dx = (x1 - x2) % q
            dy = (y1 - y2) % q
            # Quadratic form
            val = (dx * dx + dy * dy) % q
            if val != 0 and val in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
                
    return bits

def _search_algebraic():
    best = {'n': 0, 'edges': []}
    
    # Try GF(q)^2 constructions for q = 5, 13, 17 (q must be 1 mod 4)
    for q in [5, 13, 17, 29, 37]:
        if time.time() > _DEADLINE - 10:
            break
        n = q * q
        if n > 40:
            continue
        bits = _algebraic_graph_prime(q, None)
        if bits is None:
            continue
        if not _has_K4_bitset(n, bits) and not _has_indep_6_bitset(n, bits):
            edges = _edges_from_bits(n, bits)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
                
    # Also try the standard Paley graphs (1D) again, just in case
    for p in [5, 13, 17, 29, 37]:
        if time.time() > _DEADLINE - 5:
            break
        squares = {x * x % p for x in range(1, p)}
        bits = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        if not _has_K4_bitset(p, bits) and not _has_indep_6_bitset(p, bits):
            edges = _edges_from_bits(p, bits)
            if p > best['n']:
                best = {'n': p, 'edges': edges}
                
    return best

# ---------------- Randomized Search with Fast Verification ----------------

def _random_search(rng, time_limit):
    best = {'n': 0, 'edges': []}
    start_time = time.time()
    
    # Try to find valid graphs for n = 20 to 40
    # Use a density that is likely to avoid K4 and Ind6.
    # K4-free requires low density, Ind6-free requires high density.
    # For n=30, density ~0.5 is a good start.
    
    for n in range(20, 41):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random graphs for this n
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            bits = [0] * n
            # Density heuristic: p = 0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        bits[i] |= (1 << j)
                        bits[j] |= (1 << i)
                        
            if not _has_K4_bitset(n, bits) and not _has_indep_6_bitset(n, bits):
                edges = _edges_from_bits(n, bits)
                if n > best['n']:
                    best = {'n': n, 'edges': edges}
                    # If we found a valid graph for n, we can try to extend it or just move on
                    # Moving on to larger n is more valuable for the score.
                    break
                    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    
    # Start with a known valid baseline (Paley 17)
    best = {'n': 17, 'edges': []}
    p = 17
    squares = {x * x % p for x in range(1, p)}
    bits17 = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits17[i] |= (1 << j)
                bits17[j] |= (1 << i)
    if not _has_K4_bitset(p, bits17) and not _has_indep_6_bitset(p, bits17):
        best = {'n': 17, 'edges': _edges_from_bits(p, bits17)}
    
    # Try algebraic construction
    try:
        improved = _search_algebraic()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Try random search with remaining time
    remaining = _DEADLINE - time.time()
    if remaining > 5:
        try:
            rng = random.Random(42)
            improved = _random_search(rng, remaining - 5)
            if improved['n'] > best['n']:
                best = improved
        except Exception:
            pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        # Fallback to Paley 17
        best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
    
    return best
