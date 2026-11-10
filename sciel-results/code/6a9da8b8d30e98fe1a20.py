"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all edges and their opposite edges for each face
    # Edge (u,v) where v = u ^ (1<<i). For each such edge, the two opposite edges in faces
    # are determined by flipping another bit j.
    
    # For a C4-free check: adding edge e=(x,y) with y=x^(1<<i) creates a C4 iff there exists
    # j != i such that edges (x, x^(1<<j)), (y, y^(1<<j)), and (x^(1<<j), y^(1<<j)) are all present.
    
    # We'll use a randomized greedy approach with multiple restarts and random orderings,
    # keeping the best result found.
    
    best_out = None
    best_count = 0
    
    # First, do the deterministic greedy to get a baseline
    chosen = set()
    out = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                ok = True
                for j in range(d):
                    if j != i:
                        c = x ^ (1 << j)
                        e = y ^ (1 << j)
                        if ((min(x, c), max(x, c)) in chosen and
                            (min(y, e), max(y, e)) in chosen and
                            (min(c, e), max(c, e)) in chosen):
                            ok = False
                            break
                if ok:
                    chosen.add((x, y))
                    out.append([x, y])
    
    if len(out) > best_count:
        best_count = len(out)
        best_out = out[:]
    
    if time.time() > deadline:
        return best_out
    
    # Now try randomized greedy with different edge orderings
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # Shuffle and try
    rng = random.Random(42)
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        order = list(all_edges)
        rng.shuffle(order)
        
        chosen = set()
        out = []
        for x, y, i in order:
            ok = True
            for j in range(d):
                if j != i:
                    c = x ^ (1 << j)
                    e = y ^ (1 << j)
                    if ((min(x, c), max(x, c)) in chosen and
                        (min(y, e), max(y, e)) in chosen and
                        (min(c, e), max(c, e)) in chosen):
                        ok = False
                        break
            if ok:
                chosen.add((x, y))
                out.append([x, y])
        
        if len(out) > best_count:
            best_count = len(out)
            best_out = out[:]
    
    return best_out
# EVOLVE-BLOCK-END
