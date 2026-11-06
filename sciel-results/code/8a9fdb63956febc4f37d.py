"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

# Precompute all K4 edge indices once
_N = 16
_EDGE_IDX = np.array([(i, j) for i in range(_N) for j in range(i+1, _N)], dtype=np.int32)
_K4_LIST = []
for a in range(_N):
    for b in range(a+1, _N):
        for c in range(b+1, _N):
            for d in range(c+1, _N):
                _K4_LIST.append((a,b,c,d))
_K4_LIST = np.array(_K4_LIST, dtype=np.int32)  # shape (1820, 4)

# For each K4, the 6 edges as (u,v) pairs -> map to linear index
def _edge_lin_idx(u, v):
    """Linear index for edge (u,v) with u < v in the 120-element list."""
    return u * 15 - (u*(u+1))//2 + (v - u - 1)

# Precompute: for each K4 (a,b,c,d), the 6 linear edge indices
_K4_EDGE_IDX = np.zeros((len(_K4_LIST), 6), dtype=np.int32)
for ki, (a, b, c, d) in enumerate(_K4_LIST):
    _K4_EDGE_IDX[ki, 0] = _edge_lin_idx(a, b)
    _K4_EDGE_IDX[ki, 1] = _edge_lin_idx(a, c)
    _K4_EDGE_IDX[ki, 2] = _edge_lin_idx(a, d)
    _K4_EDGE_IDX[ki, 3] = _edge_lin_idx(b, c)
    _K4_EDGE_IDX[ki, 4] = _edge_lin_idx(b, d)
    _K4_EDGE_IDX[ki, 5] = _edge_lin_idx(c, d)

# For fast validity checking: for each K4, check that the 6 edge colors have >= 5 distinct values
# Equivalently: NOT (all 6 same) AND NOT (5 same + 1 diff) AND NOT (4 same + 2 diff)
# i.e., the number of distinct values among the 6 must be >= 5
# Fast check: sort the 6, count distinct via diff

def _check_valid_vec(colors_arr):
    """Vectorized check: every K4 has >= 5 distinct colors. colors_arr is int array of length 120."""
    # Gather edge colors for all K4s: shape (1820, 6)
    k4_colors = colors_arr[_K4_EDGE_IDX]  # shape (1820, 6)
    # Sort along axis 1
    k4_sorted = np.sort(k4_colors, axis=1)  # shape (1820, 6)
    # Count distinct: 1 + number of positions where k4_sorted[:,k] != k4_sorted[:,k-1]
    diffs = (k4_sorted[:, 1:] != k4_sorted[:, :-1]).sum(axis=1)  # shape (1820,)
    distinct_counts = 1 + diffs
    return bool(np.all(distinct_counts >= 5))

def _num_colours_arr(colors_arr):
    """Count distinct colors in an array."""
    return len(np.unique(colors_arr))

def _num_colours(colors_list):
    return len(set(colors_list))

def _gen_base_colors():
    """Generate base (i+j)%17 coloring as numpy array."""
    colors = np.zeros(120, dtype=np.int32)
    idx = 0
    for i in range(_N):
        for j in range(i+1, _N):
            colors[idx] = (i + j) % 17
            idx += 1
    return colors

def _try_algebraic_forms(deadline):
    """Try algebraic forms, return best (colors_arr, count)."""
    best = None
    best_count = 999
    
    for p in [13, 17, 19, 23, 29]:
        if time.time() > deadline:
            break
        for a in range(1, min(p, 10)):
            for b in range(1, min(p, 10)):
                if time.time() > deadline:
                    break
                colors = np.zeros(120, dtype=np.int32)
                idx = 0
                for i in range(_N):
                    for j in range(i+1, _N):
                        colors[idx] = (a * i + b * j) % p
                        idx += 1
                cnt = _num_colours_arr(colors)
                if cnt < best_count:
                    if _check_valid_vec(colors):
                        best = colors.copy()
                        best_count = cnt
                        if best_count <= 13:
                            return best, best_count
    return best, best_count

def _simulated_annealing(start_colors, deadline):
    """Simulated annealing to minimize number of colors while maintaining validity."""
    current = start_colors.copy()
    current_count = _num_colours_arr(current)
    best = current.copy()
    best_count = current_count
    
    # Track which colors are in use
    max_color = int(np.max(current)) + 1
    
    # Precompute for each edge, which K4s it belongs to (for fast rejection)
    # Actually, let's just do full check since it's vectorized and fast
    
    T = 2.0  # initial temperature
    cooling = 0.9995
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > 0.01:
        # Pick a random edge
        edge_idx = int(rng.integers(0, 120))
        orig_color = current[edge_idx]
        
        # Choose new color: try to use an existing color to reduce palette,
        # or try a new color (but only if it helps)
        if current_count > 13:
            # Try to map to an existing color to reduce count
            # Pick a random existing color
            existing_colors = np.unique(current)
            new_color = int(existing_colors[rng.integers(0, len(existing_colors))])
        else:
            # Keep palette small: pick from 0..best_count-1
            new_color = int(rng.integers(0, best_count))
        
        if new_color == orig_color:
            T *= cooling
            continue
        
        current[edge_idx] = new_color
        
        # Check validity
        if _check_valid_vec(current):
            new_count = _num_colours_arr(current)
            delta = new_count - current_count  # negative = improvement
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best = current.copy()
                    if best_count <= 13:
                        break
            else:
                current[edge_idx] = orig_color
        else:
            current[edge_idx] = orig_color
        
        T *= cooling
    
    return best, best_count

def _local_search_refine(colors_arr, deadline):
    """Greedy local search: try to reduce color count by remapping edges."""
    current = colors_arr.copy()
    current_count = _num_colours_arr(current)
    best = current.copy()
    best_count = current_count
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for edge_idx in range(120):
            if time.time() > deadline:
                break
            orig = current[edge_idx]
            # Try each color from 0 to best_count-1
            for new_c in range(best_count):
                if new_c == orig:
                    continue
                current[edge_idx] = new_c
                if _check_valid_vec(current):
                    new_count = _num_colours_arr(current)
                    if new_count < current_count:
                        current_count = new_count
                        best_count = new_count
                        best = current.copy()
                        improved = True
                        if best_count <= 13:
                            return best, best_count
                        break
            current[edge_idx] = orig
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with base solution
    base = _gen_base_colors()
    if not _check_valid_vec(base):
        # Fallback: should not happen, but just in case
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_colors = base.copy()
    best_count = _num_colours_arr(base)
    
    # Phase 1: Try algebraic forms
    alg_colors, alg_count = _try_algebraic_forms(deadline)
    if alg_colors is not None and alg_count < best_count:
        best_colors = alg_colors
        best_count = alg_count
        if best_count <= 13:
            return best_colors.tolist()
    
    # Phase 2: Simulated annealing from best so far
    if time.time() < deadline - 20:
        sa_colors, sa_count = _simulated_annealing(best_colors, deadline - 20)
        if sa_count < best_count:
            best_colors = sa_colors
            best_count = sa_count
            if best_count <= 13:
                return best_colors.tolist()
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 5:
        ls_colors, ls_count = _local_search_refine(best_colors, deadline - 5)
        if ls_count < best_count:
            best_colors = ls_colors
            best_count = ls_count
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
