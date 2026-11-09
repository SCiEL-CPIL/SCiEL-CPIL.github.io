"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85
    total_edges = N * (N - 1) // 2
    
    edge_idx = {}
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    # Precompute all 3-cycles per 4-vertex subset
    all_cycles = []
    edge_to_cycles = [[] for _ in range(total_edges)]
    
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        cycles = [
            (edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]),
            (edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]),
            (edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]),
        ]
        for cyc in cycles:
            idx = len(all_cycles)
            all_cycles.append(cyc)
            for ei in cyc:
                edge_to_cycles[ei].append(idx)
    
    colors = [0] * total_edges
    best_colors = colors[:]
    best_count = 1
    
    rng = random.Random(42)
    
    def is_valid_change(ei, new_color, cl):
        old = cl[ei]
        if old == new_color:
            return False
        for ci in edge_to_cycles[ei]:
            cyc = all_cycles[ci]
            cs = set()
            for e2 in cyc:
                cs.add(new_color if e2 == ei else cl[e2])
            if len(cs) == 4:
                return False
        return True
    
    def count_distinct(cl):
        return len(set(cl))
    
    current_count = 1
    
    while time.time() < deadline - 2:
        improved = False
        edge_order = list(range(total_edges))
        rng.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() >= deadline - 2:
                break
            old = colors[ei]
            used = set(colors)
            # Prefer unused colors
            unused = [c for c in range(120) if c not in used]
            if unused:
                rng.shuffle(unused)
                candidates = unused[:3]
            else:
                candidates = [rng.randint(0, 119)]
            
            for nc in candidates:
                if nc == old:
                    continue
                if is_valid_change(ei, nc, colors):
                    colors[ei] = nc
                    cc = count_distinct(colors)
                    if cc > best_count:
                        best_count = cc
                        best_colors = colors[:]
                    current_count = cc
                    improved = True
                    break
        
        if not improved:
            # Perturb: randomly change a few edges
            for _ in range(20):
                if time.time() >= deadline - 2:
                    break
                ei = rng.randint(0, total_edges - 1)
                nc = rng.randint(0, 119)
                if is_valid_change(ei, nc, colors):
                    colors[ei] = nc
                    cc = count_distinct(colors)
                    if cc > best_count:
                        best_count = cc
                        best_colors = colors[:]
                    current_count = cc
    
    return best_colors
# EVOLVE-BLOCK-END
