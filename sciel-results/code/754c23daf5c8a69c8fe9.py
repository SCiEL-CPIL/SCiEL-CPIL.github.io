"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _build_color_matrix(colors, n=16):
    """Build adjacency color matrix from edge colors in lex order."""
    col = np.full((n, n), -1, dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i, j] = col[j, i] = colors[idx]
            idx += 1
    return col

def _check_validity_fast(col, n=16, min_colors=5):
    """Fast validity check using precomputed K4 edge indices."""
    # Precompute all K4s as flat arrays of 6 edge indices
    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Edge indices in lex order: (a,b),(a,c),(a,d),(b,c),(b,d),(c,d)
        idx_ab = a * (n - 1 - a) + (b - a - 1)  # This formula is wrong, let me use a different approach
        # Actually, let's just use the direct index calculation
        # For pair (i,j) with i<j, index = i*(n-1) - i*(i+1)//2 + (j-i-1)
        # Simpler: precompute a lookup
        pass
    
    # Simpler approach: just check each K4
    for a, b, c, d in itertools.combinations(range(n), 4):
        cset = {col[a,b], col[a,c], col[a,d], col[b,c], col[b,d], col[c,d]}
        if len(cset) < min_colors:
            return False
    return True

def _check_validity(colors, n=16, min_colors=5):
    """Check if the edge coloring satisfies the constraint."""
    col = _build_color_matrix(colors, n)
    return _check_validity_fast(col, n, min_colors)

def _search_solution():
    """Search for a valid coloring with minimal number of colors."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 16
    
    # Precompute all K4s for fast checking
    all_k4s = list(itertools.combinations(range(n), 4))
    
    best_colors = None
    best_num_colors = 17  # Default from seed
    
    def try_coloring(colors):
        nonlocal best_colors, best_num_colors
        col = _build_color_matrix(colors, n)
        # Quick check: count distinct colors
        num_used = len(set(colors))
        if num_used >= best_num_colors:
            return
        # Check all K4s
        for a, b, c, d in all_k4s:
            cset = {col[a,b], col[a,c], col[a,d], col[b,c], col[b,d], col[c,d]}
            if len(cset) < 5:
                return
        best_colors = colors
        best_num_colors = num_used
        if num_used <= 13:
            return True
        return False
    
    # First, try the known good solution
    default_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    try_coloring(default_colors)
    
    # Search for better solutions with smaller m
    # Try (ai + bj) % m
    for m in range(13, 17):
        if time.time() > deadline:
            break
        for a_coeff in range(0, m):
            for b_coeff in range(0, m):
                if time.time() > deadline:
                    break
                colors = [(a_coeff*i + b_coeff*j) % m for i in range(n) for j in range(i+1, n)]
                if try_coloring(colors):
                    return best_colors
    
    # Try (ai + bj + ci*j) % m
    if time.time() < deadline - 30 and best_num_colors > 13:
        for m in range(13, 17):
            if time.time() > deadline:
                break
            for a_coeff in range(0, m):
                for b_coeff in range(0, m):
                    for c_coeff in range(0, m):
                        if time.time() > deadline:
                            break
                        colors = [(a_coeff*i + b_coeff*j + c_coeff*i*j) % m 
                                 for i in range(n) for j in range(i+1, n)]
                        if try_coloring(colors):
                            return best_colors
    
    # Try (ai^2 + bj + ci*j) % m
    if time.time() < deadline - 20 and best_num_colors > 13:
        for m in range(13, 17):
            if time.time() > deadline:
                break
            for a_coeff in range(0, m):
                for b_coeff in range(0, m):
                    for c_coeff in range(0, m):
                        if time.time() > deadline:
                            break
                        colors = [(a_coeff*(i*i) + b_coeff*j + c_coeff*i*j) % m 
                                 for i in range(n) for j in range(i+1, n)]
                        if try_coloring(colors):
                            return best_colors
    
    # Try (ai + bj + ci*j + di*j*(i+j)) % m
    if time.time() < deadline - 10 and best_num_colors > 13:
        for m in range(13, 17):
            if time.time() > deadline:
                break
            for a_coeff in range(0, m):
                for b_coeff in range(0, m):
                    for c_coeff in range(0, m):
                        for d_coeff in range(0, m):
                            if time.time() > deadline:
                                break
                            colors = [(a_coeff*i + b_coeff*j + c_coeff*i*j + d_coeff*i*j*(i+j)) % m 
                                     for i in range(n) for j in range(i+1, n)]
                            if try_coloring(colors):
                                return best_colors
    
    return best_colors

def construct():
    result = _search_solution()
    if result is None:
        # Fallback to known valid solution
        result = [(i + j) % 17 for i in range(16) for j in range(i+1, 16)]
    return result
# EVOLVE-BLOCK-END
