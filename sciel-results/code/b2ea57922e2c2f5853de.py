"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(vals, edges):
    """Compute max(distinct_sums, distinct_products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(vals[i] + vals[j])
        prods.add(vals[i] * vals[j])
    return max(len(sums), len(prods))

def construct():
    t0 = time.time()
    deadline = t0 + 78
    n = 14
    m = 35
    maxv = 10000
    rng = random.Random(777)
    
    all_pairs = list(combinations(range(n), 2))
    
    best = float('inf')
    best_vals = None
    best_edges = None
    
    # Good initial: powers of 2 (1,2,4,...,8192)
    # Products: 2^(a+b), a≠b, a,b in 0..13 → 25 distinct
    # Sums: all distinct → 35
    # So raw = 35. Need to reduce sums.
    
    # Try: replace some powers of 2 to create sum collisions
    # e.g., 1+8=9, 2+7? No, 7 not power of 2.
    # 1+14? Not power of 2.
    # Key insight: 2^a + 2^b = 2^c + 2^d only if {a,b}={c,d}
    # So pure powers of 2 give 35 distinct sums.
    
    # Strategy: use values with shared structure
    # Try: all even numbers 2,4,6,...,28
    # Products: many collisions expected
    # Sums: 2+28=30, 4+26=30, 6+24=30, ... many collisions!
    
    init_sets = [
        list(range(2, 30, 2)),           # 2,4,...,28
        list(range(3, 45, 3)),           # 3,6,...,42
        [2**i for i in range(14)],       # powers of 2
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
    ]
    
    for iv in init_sets:
        if len(set(iv)) != n or any(v < 1 or v > maxv for v in iv):
            continue
        edges = [list(p) for p in all_pairs[:m]]
        s = _score(iv, edges)
        if s < best:
            best = s
            best_vals = list(iv)
            best_edges = edges
    
    if best_vals is None:
        best_vals = list(range(1, n+1))
        best_edges = [list(p) for p in all_pairs[:m]]
        best = _score(best_vals, best_edges)
    
    cur_vals = list(best_vals)
    cur_edges = [list(e) for e in best_edges]
    cur_score = best
    
    stall = 0
    while time.time() < deadline and stall < 800:
        improved = False
        
        # Perturb one value
        idx = rng.randint(0, n-1)
        old = cur_vals[idx]
        tried = False
        for _ in range(30):
            nv = rng.randint(1, maxv)
            if nv == old or nv in cur_vals:
                continue
            cur_vals[idx] = nv
            s = _score(cur_vals, cur_edges)
            if s < cur_score:
                cur_score = s
                improved = True
                if s < best:
                    best = s
                    best_vals = list(cur_vals)
                    best_edges = [list(e) for e in cur_edges]
                break
            elif s == cur_score and rng.random() < 0.1:
                # Accept equal (plateau move)
                improved = True
                break
            cur_vals[idx] = old
            tried = True
        if not tried:
            cur_vals[idx] = old
        
        # Perturb one edge (40% chance)
        if rng.random() < 0.4:
            ei = rng.randint(0, m-1)
            oe = cur_edges[ei]
            existing = set(tuple(sorted(e)) for e in cur_edges)
            missing = [p for p in all_pairs if tuple(sorted(p)) not in existing]
            if missing:
                ne = rng.choice(missing)
                cur_edges[ei] = list(ne)
                s = _score(cur_vals, cur_edges)
                if s < cur_score:
                    cur_score = s
                    improved = True
                    if s < best:
                        best = s
                        best_vals = list(cur_vals)
                        best_edges = [list(e) for e in cur_edges]
                elif s == cur_score and rng.random() < 0.1:
                    improved = True
                else:
                    cur_edges[ei] = oe
        
        if improved:
            stall = 0
        else:
            stall += 1
        
        # Restart from best with perturbation if stuck
        if stall > 200:
            cur_vals = list(best_vals)
            cur_edges = [list(e) for e in best_edges]
            cur_score = best
            for _ in range(rng.randint(3, 5)):
                idx = rng.randint(0, n-1)
                nv = rng.randint(1, maxv)
                if nv not in cur_vals:
                    cur_vals[idx] = nv
            for _ in range(rng.randint(2, 4)):
                ei = rng.randint(0, m-1)
                existing = set(tuple(sorted(e)) for e in cur_edges)
                missing = [p for p in all_pairs if tuple(sorted(p)) not in existing]
                if missing:
                    cur_edges[ei] = list(rng.choice(missing))
            cur_score = _score(cur_vals, cur_edges)
            stall = 0
    
    rv = [int(v) for v in (best_vals if best_vals else cur_vals)]
    re = [[int(i), int(j)] for i, j in (best_edges if best_edges else cur_edges)]
    return {'values': rv, 'edges': re}

# EVOLVE-BLOCK-END
