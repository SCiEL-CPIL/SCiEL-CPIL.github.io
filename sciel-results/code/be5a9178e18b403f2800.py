"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _triangle_free_circulant(n, S):
    """Return True iff circulant graph on Z_n with symmetric connection set S is triangle-free."""
    Sset = set(S)
    Sneg = {(-x) % n for x in S}
    for a in S:
        for b in S:
            if a == b:
                continue
            c = (a + b) % n
            if c in Sset or c in Sneg:
                return False
    return True

def _alpha_circulant(n, S):
    """Exact maximum independent set size of a circulant graph via branch-and-bound."""
    Sset = set(S)
    # Precompute neighbor masks
    adj = [0] * n
    for v in range(n):
        m = 0
        for s in Sset:
            m |= (1 << ((v + s) % n))
        adj[v] = m
    full = (1 << n) - 1
    # Independent-set candidates when adding v: those not adjacent to v and not v
    indep_after = [0] * n
    for v in range(n):
        indep_after[v] = full & ~(1 << v) & ~adj[v]

    best = 0
    # Iterative DFS with explicit stack to avoid recursion depth issues
    # Stack entries: (candidates_mask, current_size)
    # We use a list as stack
    stack = [(full, 0)]
    while stack:
        cand, sz = stack.pop()
        if cand == 0:
            if sz > best:
                best = sz
            continue
        # Upper bound
        if sz + bin(cand).count('1') <= best:
            continue
        # Pick vertex with max degree in induced subgraph of cand
        temp = cand
        best_v = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(cand & adj[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        v = best_v
        # Exclude v first (often prunes more)
        stack.append((cand & ~(1 << v), sz))
        # Include v
        stack.append((cand & indep_after[v], sz + 1))
    return best

def _build_edges(n, S):
    Sset = set(S)
    edges = []
    for i in range(n):
        for s in Sset:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _try_circulant(n, S):
    """Check if circulant(n, S) is triangle-free with alpha < 10. Return edges or None."""
    if not _triangle_free_circulant(n, S):
        return None
    alpha = _alpha_circulant(n, S)
    if alpha >= 10:
        return None
    return _build_edges(n, S)

def construct():
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill

    best_n = 0
    best_edges = []

    # Iterate n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 15:
            break

        found = False
        # Number of attempts: more for smaller n where search space is more tractable
        attempts = 800 if n <= 36 else 400

        for attempt in range(attempts):
            if time.time() > deadline:
                break

            # Random symmetric connection set S
            # For circulant graphs, S must be symmetric: if x in S then (-x)%n in S
            # For odd n, we pick from 1..(n-1)/2 and add both x and n-x
            # For even n, we pick from 1..n/2-1 (and optionally n/2) and add both

            half = n // 2
            # Target density: we want enough edges to keep alpha < 10
            # Triangle-free circulant graphs have max degree ~ n/3 or so
            # Let's try various target sizes
            target = random.randint(3, min(half, int(n * 0.35)))
            if target < 2:
                target = 2

            # Build candidate pool of "pairs" {x, -x mod n}
            if n % 2 == 1:
                # Odd n: pairs are {x, n-x} for x in 1..(n-1)/2
                pool = list(range(1, half + 1))
            else:
                # Even n: pairs {x, n-x} for x in 1..n/2-1, plus singleton {n/2}
                pool = list(range(1, half))
                # Optionally include n/2
                if random.random() < 0.3:
                    pool.append(half)

            random.shuffle(pool)

            S = set()
            for x in pool:
                if len(S) >= target * 2:  # Each pair adds 2 elements (except n/2)
                    break
                if x == n - x:  # n/2 case for even n
                    new_S = S | {x}
                else:
                    new_S = S | {x, n - x}
                if _triangle_free_circulant(n, new_S):
                    S = new_S

            if len(S) < 2:
                continue

            edges = _try_circulant(n, S)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break

        if found:
            break

    # Fallback: hardcoded known triangle-free graphs with alpha < 10
    if best_n < 20:
        # Use a known construction: C5 blow-up or Andrásfai-type
        # Simple fallback: complement of a 9-clique-free graph... 
        # Let's use a 20-vertex circulant: n=20, S={1,2,3,4,6,7,8,9,12,13,14,15,16,17,18,19}
        # Actually, let's just use a known small example.
        # n=10, S={1,2,3,4,6,7,8,9}: check triangle-free? 
        # 1+2=3 in S -> triangle! 
        # Let's use n=19, S={1,2,3,4,5,6,7,8,9} minus conflicts
        # Simpler: use a 5-partite Turan-like graph
        # Actually, the simplest valid fallback: empty graph on 9 vertices (alpha=9 < 10, no triangles)
        # But we want to maximize n. Let's use a known 15-vertex graph.
        # For now, use n=9 empty graph as absolute fallback
        n_fb = 9
        best_edges = []
        best_n = n_fb

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
