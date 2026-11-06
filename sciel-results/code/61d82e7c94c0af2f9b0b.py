"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 135  # safety margin before 150s limit
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def is_valid_fast(coloring, n=16):
        """Fast validation using a 2D array and precomputed K4 edge indices."""
        # Build edge color lookup as a 2D array
        ec = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                ec[i, j] = coloring[idx]
                ec[j, i] = coloring[idx]
                idx += 1
        # Precompute all K4s and their 6 edge positions
        quads = list(itertools.combinations(range(n), 4))
        for a, b, c, d in quads:
            colors = [ec[a,b], ec[a,c], ec[a,d], ec[b,c], ec[b,d], ec[c,d]]
            if len(set(colors)) < 5:
                return False
        return True
    
    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    # Try various constructions
    candidates = []
    
    # Linear forms: (a*i + b*j) mod m
    for mod in range(13, 20):
        for a in range(1, 8):
            for b in range(1, 8):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)
    
    # Non-linear: (a*i*i + b*i*j + c*j*j) mod m
    for mod in range(13, 18):
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 4):
                    if a == b == c:
                        continue
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i*i + b*i*j + c*j*j) % mod
                    )
                    candidates.append(cand)
    
    # XOR-based: (i XOR j) * k mod m
    for mod in range(13, 18):
        for k in range(1, 5):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i ^ j) * k) % mod
            )
            candidates.append(cand)
    
    # Product-based: (i * j) mod m
    for mod in range(13, 18):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j) % mod)
        candidates.append(cand)
    
    # Mixed: (a*i + b*j + c*i*j) mod m
    for mod in range(13, 17):
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 3):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i + b*j + c*i*j) % mod
                    )
                    candidates.append(cand)
    
    # Difference-based: (i - j)^2 mod m
    for mod in range(13, 18):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i - j)**2) % mod)
        candidates.append(cand)
    
    # Sum of squares: (i^2 + j^2) mod m
    for mod in range(13, 18):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + j*j) % mod)
        candidates.append(cand)
    
    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')
    
    # Sort candidates by number of colors used (ascending) to find minimum faster
    candidates.sort(key=lambda c: count_colors(c))
    
    for cand in candidates:
        if time.time() > deadline:
            break
        nc = count_colors(cand)
        if nc >= best_num_colors:
            break  # No need to check candidates with more colors
        if is_valid_fast(cand):
            best_num_colors = nc
            best = cand
    
    if best is not None:
        return best
    
    # Fallback: return a known valid coloring
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
