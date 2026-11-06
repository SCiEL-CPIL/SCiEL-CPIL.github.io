"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n = 16
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def build_edge_array(coloring, n=16):
        """Build a 16x16 symmetric matrix where edge_matrix[i,j] = color of edge (i,j)"""
        edge_matrix = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_matrix[i, j] = coloring[idx]
                edge_matrix[j, i] = coloring[idx]
                idx += 1
        return edge_matrix
    
    def is_valid_fast(edge_matrix, n=16):
        """Check if every K4 has at least 5 distinct colors using numpy"""
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            colors = np.array([
                edge_matrix[a, b], edge_matrix[a, c], edge_matrix[a, d],
                edge_matrix[b, c], edge_matrix[b, d], edge_matrix[c, d]
            ], dtype=np.int32)
            if len(np.unique(colors)) < 5:
                return False
        return True
    
    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    candidates = []
    
    # Linear formulas: (a*i + b*j) mod m
    for mod in range(13, 20):
        for a in range(1, 8):
            for b in range(1, 8):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)
    
    # XOR-based formulas
    for mod in range(13, 20):
        for shift in range(4):
            cand = make_coloring_formula(lambda i, j, shift=shift, mod=mod: ((i << shift) ^ j) % mod)
            candidates.append(cand)
    
    # Product-based: (i*j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j) % mod)
        candidates.append(cand)
    
    # Mixed: (i*j + i + j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j + i + j) % mod)
        candidates.append(cand)
    
    # Quadratic: (i^2 + j^2) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + j*j) % mod)
        candidates.append(cand)
    
    # Difference-based: (i-j)^2 mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i-j)*(i-j)) % mod)
        candidates.append(cand)
    
    # (i^2 + a*i*j + b*j^2) mod m
    for mod in range(13, 20):
        for a in range(1, 5):
            for b in range(1, 5):
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (i*i + a*i*j + b*j*j) % mod)
                candidates.append(cand)
    
    # Bit manipulation: (i ^ j) * k mod m
    for mod in range(13, 20):
        for k in range(1, 5):
            cand = make_coloring_formula(lambda i, j, k=k, mod=mod: ((i ^ j) * k) % mod)
            candidates.append(cand)
    
    # Sum of bits: popcount(i) + popcount(j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (bin(i).count('1') + bin(j).count('1')) % mod)
        candidates.append(cand)
    
    # (i+j)^2 mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i+j)*(i+j)) % mod)
        candidates.append(cand)
    
    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')
    best_edge_matrix = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        edge_matrix = build_edge_array(cand)
        if is_valid_fast(edge_matrix):
            nc = count_colors(cand)
            if nc < best_num_colors:
                best_num_colors = nc
                best = cand
                best_edge_matrix = edge_matrix
    
    if best is not None:
        return best
    
    # Fallback: return the seed
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
