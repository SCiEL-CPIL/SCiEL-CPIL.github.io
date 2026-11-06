"""Construct a K4-free graph with no independent 6-set, maximizing n.

Strategy:
1. Generate candidate edge sets from algebraic constructions (circulant difference
   sets in Z_n, affine geometry AG(2,6) parallel-line/cap rules).
2. Validate each candidate exactly (no K4, no independent 6-set) using bitsets.
3. Take the best valid base, then grow it with a fast bitset local search
   (add a vertex with a structured initial neighborhood, then repair violations
   by edge flips) up to n=40 or the time budget.
"""

import time
import random
import itertools

_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - (time.time() - _START)


def construct():
    global _START
    _START = time.time()

    best_n = 0
    best_adj = None

    # --- Phase 1: Try algebraic constructions for n up to 36 ---
    # Circulant graphs: edge set defined by a difference set D in Z_n.
    # A circulant graph C(n, D) has edge (i, i+d mod n) for d in D.
    # For K4-freedom and alpha<=5, we need careful D choices.
    #
    # Known: Paley graph on q=17 (q ≡ 1 mod 4) is K4-free with alpha=5? 
    # Actually Paley(17) has alpha=5 and is K4-free? Let's check via construction.
    #
    # More broadly: try all "nice" difference sets for n in 25..36.

    # Try circulant constructions
    for n in range(25, 37):
        if _time_left() < 15:
            break
        adj = _try_circulant(n)
        if adj is not None:
            if n > best_n:
                best_n = n
                best_adj = adj
                if best_n >= 36:
                    break

    # Try AG(2,6) based construction for n=36
    if _time_left() > 15:
        adj = _try_ag26()
        if adj is not None:
            if 36 > best_n:
                best_n = 36
                best_adj = adj

    # If no good base found, fall back to a smaller known-good construction
    if best_adj is None or best_n < 20:
        adj = _try_circulant(17)
        if adj is not None:
            best_n = 17
            best_adj = adj
        else:
            # Absolute fallback: cycle C_17 (K4-free, alpha=8 -- not valid, but we'll repair)
            # Use Paley(17)
            adj = _paley_graph(17)
            best_n = 17
            best_adj = adj

    # --- Phase 2: Grow via local search ---
    if _time_left() > 10 and best_adj is not None:
        best_adj, best_n = _grow_local_search(best_adj, best_n)

    return {'n': best_n, 'edges': _adj_to_edges(best_adj, best_n)}


def _paley_graph(q):
    """Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    adj = [set() for _ in range(q)]
    # Quadratic residues mod q
    qr = set()
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == 1:
            qr.add(x)
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in qr:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _try_circulant(n):
    """Try to find a valid circulant graph on n vertices.
    
    A circulant graph is defined by a connection set D ⊆ {1, ..., floor(n/2)}.
    Edge (i, j) exists iff (j-i) mod n ∈ D or (i-j) mod n ∈ D.
    
    We try several structured D's:
    - Consecutive intervals
    - Every k-th
    - Random D with size ~ n/4
    """
    rng = random.Random(n * 7 + 13)
    
    # Try a few candidate difference sets
    candidates = []
    
    # Consecutive: D = {1, 2, ..., k}
    for k in [3, 4, 5, 6, 7, 8]:
        if 2 * k < n:
            candidates.append(set(range(1, k + 1)))
    
    # Every k-th: D = {k, 2k, ..., mk} where mk <= n//2
    for k in [2, 3, 4, 5]:
        d = set()
        m = k
        while m <= n // 2:
            d.add(m)
            m += k
        if d:
            candidates.append(d)
    
    # Random sets of various sizes
    for size in [n // 4, n // 3, n // 2 - 2]:
        for _ in range(5):
            d = set(rng.sample(range(1, n // 2 + 1), min(size, n // 2)))
            candidates.append(d)
    
    # Symmetric: ensure if d in D then n-d in D (automatic for undirected)
    for D in candidates:
        # Normalize: keep only values in 1..n//2
        D = {d % n for d in D if d % n != 0}
        D = {min(d, n - d) for d in D}
        if not D:
            continue
        
        adj = [set() for _ in range(n)]
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j > i:
                    adj[i].add(j)
                    adj[j].add(i)
        
        if _is_valid(adj, n):
            return adj
    
    return None


def _try_ag26():
    """Try to construct a valid graph on 36 vertices using AG(2,6) structure.
    
    AG(2,6) has 36 points arranged in a 6x6 grid.
    Lines are sets of 6 collinear points (horizontal, vertical, or slope-1 lines).
    There are 6 parallel classes, each with 6 lines.
    
    Idea: Define edges based on non-collinearity with a twist, or use
    a cap (no 3 collinear) structure.
    
    Simpler approach: Use the 6x6 grid with edges connecting points that
    are NOT on the same row, column, or "diagonal" (toroidal).
    This gives a 36-vertex graph. Let's check if it's K4-free and alpha<=5.
    
    Actually, let's try: edge (r1,c1)-(r2,c2) iff (r1-r2 mod 6) != 0 AND 
    (c1-c2 mod 6) != 0 AND (r1-c1 - (r2-c2) mod 6) != 0.
    This excludes rows, columns, and one diagonal direction.
    Degree = 36 - 1 - 5 - 5 - 5 = 20? Let's compute: for a fixed point, 
    5 in same row, 5 in same col, 5 in same diagonal. Total excluded neighbors: 15.
    So degree = 36-1-15 = 20.
    """
    n = 36
    adj = [set() for _ in range(n)]
    
    for r1 in range(6):
        for c1 in range(6):
            v1 = r1 * 6 + c1
            for r2 in range(6):
                for c2 in range(6):
                    v2 = r2 * 6 + c2
                    if v1 >= v2:
                        continue
                    # Exclude if same row, same col, or same (r-c mod 6) diagonal
                    if r1 == r2:
                        continue
                    if c1 == c2:
                        continue
                    if (r1 - c1) % 6 == (r2 - c2) % 6:
                        continue
                    adj[v1].add(v2)
                    adj[v2].add(v1)
    
    if _is_valid(adj, n):
        return adj
    
    # Try a variant: exclude rows, cols, and BOTH diagonals
    adj = [set() for _ in range(n)]
    for r1 in range(6):
        for c1 in range(6):
            v1 = r1 * 6 + c1
            for r2 in range(6):
                for c2 in range(6):
                    v2 = r2 * 6 + c2
                    if v1 >= v2:
                        continue
                    if r1 == r2:
                        continue
                    if c1 == c2:
                        continue
                    if (r1 - c1) % 6 == (r2 - c2) % 6:
                        continue
                    if (r1 + c1) % 6 == (r2 + c2) % 6:
                        continue
                    adj[v1].add(v2)
                    adj[v2].add(v1)
    
    if _is_valid(adj, n):
        return adj
    
    # Try: edge iff NOT same row AND NOT same col (complete 6x6 grid complement of rows+cols)
    # This is K_{6,6} complement... actually it's the complement of the union of 6 cliques (rows) and 6 cliques (cols)
    # Each vertex has degree 25. Likely has K4.
    adj = [set() for _ in range(n)]
    for r1 in range(6):
        for c1 in range(6):
            v1 = r1 * 6 + c1
            for r2 in range(6):
                for c2 in range(6):
                    v2 = r2 * 6 + c2
                    if v1 >= v2:
                        continue
                    if r1 == r2 or c1 == c2:
                        continue
                    adj[v1].add(v2)
                    adj[v2].add(v1)
    
    if _is_valid(adj, n):
        return adj
    
    return None


def _is_valid(adj, n):
    """Check if graph is K4-free and has no independent set of size 6."""
    if _has_k4(adj, n):
        return False
    if _has_ind6(adj, n):
        return False
    return True


def _has_k4(adj, n):
    """Check for K4 using bitsets. Returns True if found."""
    # Convert to bitsets
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        bits[i] = b
    
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            # Common neighbors of a and b
            common = bits[a] & bits[b]
            # Remove a and b
            common &= ~(1 << a)
            common &= ~(1 << b)
            # For each pair of common neighbors, check if they're adjacent
            # Instead, iterate over common neighbors
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    # Check if c-d edge exists
                    if (bits[c] >> d) & 1:
                        return True
    return False


def _has_ind6(adj, n):
    """Check for independent set of size 6 using bitsets and backtracking."""
    # Non-adjacency bitsets
    non_adj = [0] * n
    all_bits = (1 << n) - 1
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        non_adj[i] = all_bits ^ b ^ (1 << i)  # exclude self and neighbors
    
    def backtrack(candidates, current_size):
        if current_size >= 6:
            return True
        if not candidates:
            return False
        # Pick a vertex
        low = candidates & (-candidates)
        v = low.bit_length() - 1
        rest = candidates & ~low
        
        # Try including v
        new_cands = rest & non_adj[v]
        if backtrack(new_cands, current_size + 1):
            return True
        
        # Try excluding v
        if backtrack(rest, current_size):
            return True
        
        return False
    
    return backtrack((1 << n) - 1, 0)


def _grow_local_search(adj, n):
    """Grow the graph by adding vertices and repairing violations."""
    rng = random.Random(12345)
    
    for target_n in range(n + 1, 41):
        if _time_left() < 5:
            break
        
        # Try multiple attempts to add one vertex
        success = False
        for attempt in range(50):
            if _time_left() < 3:
                break
            
            # Create new graph with one extra vertex
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            # Give new vertex a random-ish neighborhood
            # Start with ~40% connection probability
            for i in range(n):
                if rng.random() < 0.4:
                    new_adj[new_n - 1].add(i)
                    new_adj[i].add(new_n - 1)
            
            # Repair
            repaired = _repair_fast(new_adj, new_n, rng, max_iters=500)
            if repaired is not None:
                adj = repaired
                n = new_n
                success = True
                break
        
        if not success:
            # Try a different strategy: add vertex with structured neighborhood
            success = False
            for attempt in range(30):
                if _time_left() < 3:
                    break
                
                new_n = n + 1
                new_adj = [set(s) for s in adj]
                new_adj.append(set())
                
                # Structured: connect to every k-th vertex
                k = rng.choice([2, 3, 4, 5])
                for i in range(n):
                    if i % k == 0:
                        new_adj[new_n - 1].add(i)
                        new_adj[i].add(new_n - 1)
                
                repaired = _repair_fast(new_adj, new_n, rng, max_iters=500)
                if repaired is not None:
                    adj = repaired
                    n = new_n
                    success = True
                    break
            
            if not success:
                break  # Can't grow further
    
    return adj, n


def _repair_fast(adj, n, rng, max_iters=500):
    """Fast repair using bitsets. Returns repaired adj or None."""
    adj = [set(s) for s in adj]
    
    for it in range(max_iters):
        if _time_left() < 1:
            return None
        
        # Check K4
        k4 = _find_k4(adj, n)
        if k4:
            a, b, c, d = k4
            # Remove a random edge from the K4
            pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            u, v = pairs[rng.randrange(len(pairs))]
            adj[u].discard(v)
            adj[v].discard(u)
            continue
        
        # Check independent 6-set
        ind_set = _find_ind6(adj, n)
        if ind_set:
            # Add an edge within the independent set
            u, v = ind_set[rng.randrange(len(ind_set))]
            adj[u].add(v)
            adj[v].add(u)
            continue
        
        # Valid!
        return adj
    
    # Final check
    if not _find_k4(adj, n) and not _find_ind6(adj, n):
        return adj
    return None


def _find_k4(adj, n):
    """Find a K4. Returns (a,b,c,d) or None."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        bits[i] = b
    
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            common = bits[a] & bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if (bits[c] >> d) & 1:
                        return (a, b, c, d)
    return None


def _find_ind6(adj, n):
    """Find an independent set of size 6. Returns list of (u,v) pairs or None."""
    all_bits = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        non_adj[i] = all_bits ^ b ^ (1 << i)
    
    def backtrack(candidates, current):
        if len(current) >= 6:
            return current[:]
        if not candidates:
            return None
        
        low = candidates & (-candidates)
        v = low.bit_length() - 1
        rest = candidates & ~low
        
        # Try including v
        new_cands = rest & non_adj[v]
        result = backtrack(new_cands, current + [v])
        if result is not None:
            return result
        
        # Try excluding v
        return backtrack(rest, current)
    
    vertices = backtrack(all_bits, [])
    if vertices is None:
        return None
    
    return [(vertices[i], vertices[j]) for i in range(len(vertices)) for j in range(i + 1, len(vertices))]


def _adj_to_edges(adj, n):
    """Convert adjacency (list of sets) to edge list."""
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
