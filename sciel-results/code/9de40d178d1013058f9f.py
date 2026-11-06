"""
4-coloring of complete graphs without monochromatic C6 (R_4(C6))
Optimized with cached adjacency and faster C6 detection.
"""

import time
import random
import numpy as np

DEADLINE = 150.0
TIME_LIMIT = DEADLINE - 5.0  # 145s


def _c6_count_from_adj(A, n):
    """Count 6-cycles using adjacency matrix A (n x n bool/uint8).
    Uses: C6 = (1/12) * (sum_{i!=j} (A^6)[i,j] - 2*sum_{i!=j} (A^2)[i,j]*(A^4)[i,j] + ...)
    Actually, let's use the direct formula:
    Number of (not necessarily simple) closed walks of length 6 = trace(A^6)
    But we want simple 6-cycles.
    
    For small n, use the combinatorial formula:
    C6 = (1/12) * sum_{v0<v1<...<v5 distinct} [A(v0,v1)*A(v1,v2)*...*A(v5,v0) + A(v0,v5)*A(v5,v4)*...*A(v1,v0)]
    But that's the same as before.
    
    Let me use a different approach: for each pair of non-adjacent vertices (i,j),
    count paths of length 5 from i to j. Each C6 contributes to exactly 2 such
    counts (one for each "gap" in the cycle). Wait, no.
    
    Actually, for a C6 v0-v1-v2-v3-v4-v5-v0, the non-adjacent pairs among
    {v0,...,v5} are: (v0,v2),(v0,v3),(v0,v4),(v1,v3),(v1,v4),(v1,v5),(v2,v4),(v2,v5),(v3,v5)
    That's 9 non-adjacent pairs. For each, there may be multiple paths of length 5.
    This is getting complicated.
    
    For n <= 20, let's just use the direct enumeration but with numpy vectorization
    where possible.
    """
    A_bool = A.astype(bool)
    count = 0
    for v0 in range(n):
        v1_mask = A_bool[v0].copy()
        v1_mask[v0] = False
        v1s = np.where(v1_mask)[0]
        if len(v1s) == 0:
            continue
        for v1 in v1s:
            v2_mask = A_bool[v1].copy()
            v2_mask[v0] = False
            v2_mask[v1] = False
            v2s = np.where(v2_mask)[0]
            if len(v2s) == 0:
                continue
            for v2 in v2s:
                v3_mask = A_bool[v2].copy()
                v3_mask[v0] = False
                v3_mask[v1] = False
                v3_mask[v2] = False
                v3s = np.where(v3_mask)[0]
                if len(v3s) == 0:
                    continue
                for v3 in v3s:
                    v4_mask = A_bool[v3].copy()
                    v4_mask[v0] = False
                    v4_mask[v1] = False
                    v4_mask[v2] = False
                    v4_mask[v3] = False
                    v4s = np.where(v4_mask)[0]
                    if len(v4s) == 0:
                        continue
                    for v4 in v4s:
                        v5_mask = A_bool[v4].copy()
                        v5_mask[v0] = False
                        v5_mask[v1] = False
                        v5_mask[v2] = False
                        v5_mask[v3] = False
                        v5_mask[v4] = False
                        v5s = np.where(v5_mask)[0]
                        if len(v5s) == 0:
                            continue
                        # Check which v5s connect back to v0
                        back = A_bool[v5s, v0]
                        count += back.sum()
    return count // 12


def _c6_exists_from_adj(A, n):
    """Check if any 6-cycle exists."""
    return _c6_count_from_adj(A, n) > 0


def _cyclic_17():
    """Try cyclic constructions for K_17."""
    n = 17
    all_partitions = [
        [(0,3),(1,4),(2,5),(6,7)],
        [(0,2),(1,5),(3,6),(4,7)],
        [(0,1),(2,6),(3,5),(4,7)],
        [(0,4),(1,3),(2,6),(5,7)],
        [(0,5),(1,2),(3,4),(6,7)],
        [(0,6),(1,7),(2,3),(4,5)],
        [(0,7),(1,2),(3,5),(4,6)],
        [(0,3),(1,6),(2,4),(5,7)],
        [(0,4),(1,7),(2,5),(3,6)],
        [(0,5),(1,3),(2,7),(4,6)],
        [(0,6),(1,3),(2,7),(4,5)],
        [(0,7),(1,4),(2,5),(3,6)],
        [(0,2),(1,6),(3,4),(5,7)],
        [(0,1),(2,5),(3,7),(4,6)],
        [(0,4),(1,2),(3,6),(5,7)],
        [(0,5),(1,6),(2,3),(4,7)],
        [(0,6),(1,4),(2,7),(3,5)],
        [(0,7),(1,5),(2,4),(3,6)],
        [(0,3),(1,7),(2,4),(5,6)],
        [(0,2),(1,7),(3,5),(4,6)],
    ]
    for part in all_partitions:
        dist_color = {}
        for c, (d1, d2) in enumerate(part):
            dist_color[d1 + 1] = c
            dist_color[d2 + 1] = c
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d = min(d, n - d)
                colors.append(dist_color[d])
        # Check validity
        adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adjs[c][i, j] = 1
                adjs[c][j, i] = 1
                idx += 1
        valid = True
        for adj in adjs:
            if _c6_exists_from_adj(adj, n):
                valid = False
                break
        if valid:
            return colors
    return None


def _local_search(colors, n, time_budget, rng):
    """Local search to minimize total C6."""
    num_pairs = n * (n - 1) // 2
    
    # Build initial adjacencies
    adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    
    def compute_total_c6():
        return sum(_c6_count_from_adj(adj, n) for adj in adjs)
    
    best_colors = colors[:]
    best_c6 = compute_total_c6()
    current_temp = 30.0
    cooling = 0.9998
    start = time.time()
    
    while time.time() - start < time_budget:
        idx = rng.randint(0, num_pairs - 1)
        old = colors[idx]
        new = rng.randint(0, 3)
        if new == old:
            continue
        
        # Update adjacency
        # Find i,j from idx
        # idx = i*n - i*(i+1)//2 + (j - i - 1) ... let's precompute
        # Actually, let's just rebuild for simplicity (slow but correct)
        # Better: precompute pair indices
        pass
    
    # This approach of rebuilding is too slow. Let me precompute.
    # Precompute: for each idx, what are (i,j) and which color
    pair_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_map[idx] = (i, j)
            idx += 1
    
    best_colors = colors[:]
    adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    
    best_c6 = sum(_c6_count_from_adj(adj, n) for adj in adjs)
    current_temp = 30.0
    cooling = 0.9998
    start = time.time()
    
    while time.time() - start < time_budget:
        idx = rng.randint(0, num_pairs - 1)
        old = colors[idx]
        new = rng.randint(0, 3)
        if new == old:
            continue
        
        i, j = pair_map[idx]
        
        # Remove old edge
        adjs[old][i, j] = 0
        adjs[old][j, i] = 0
        # Add new edge
        adjs[new][i, j] = 1
        adjs[new][j, i] = 1
        colors[idx] = new
        
        new_c6 = sum(_c6_count_from_adj(adj, n) for adj in adjs)
        delta = new_c6 - best_c6
        
        if new_c6 < best_c6 or (current_temp > 0.01 and rng.random() < np.exp(-max(delta, 0) / current_temp)):
            best_colors = colors[:]
            best_c6 = new_c6
        else:
            # Revert
            adjs[new][i, j] = 0
            adjs[new][j, i] = 0
            adjs[old][i, j] = 1
            adjs[old][j, i] = 1
            colors[idx] = old
        
        current_temp *= cooling
    
    return best_colors, best_c6


def construct():
    start_time = time.time()
    rng = random.Random(999)
    
    # Get base 17-vertex coloring
    base = _cyclic_17()
    if base is None:
        # Fallback: random + local search for n=17
        n = 17
        colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        base, _ = _local_search(colors, n, 20.0, rng)
    
    best_n = 17
    best_colors = base
    
    # Try n=18
    if time.time() - start_time < TIME_LIMIT * 0.55:
        n18 = 18
        num_pairs_18 = n18 * (n18 - 1) // 2
        colors_18 = base + [rng.randint(0, 3) for _ in range(num_pairs_18 - len(base))]
        budget = min(TIME_LIMIT * 0.45, max(5.0, TIME_LIMIT - (time.time() - start_time) - 2.0))
        colors_18, c6_18 = _local_search(colors_18, n18, budget, rng)
        if c6_18 == 0:
            best_n = 18
            best_colors = colors_18
    
    # Try n=19
    if best_n == 17 and time.time() - start_time < TIME_LIMIT * 0.85:
        n19 = 19
        num_pairs_19 = n19 * (n19 - 1) // 2
        colors_19 = base + [rng.randint(0, 3) for _ in range(num_pairs_19 - len(base))]
        budget = min(TIME_LIMIT * 0.15, max(5.0, TIME_LIMIT - (time.time() - start_time) - 2.0))
        colors_19, c6_19 = _local_search(colors_19, n19, budget, rng)
        if c6_19 == 0:
            best_n = 19
            best_colors = colors_19
    
    # Final verification
    adjs = [np.zeros((best_n, best_n), dtype=np.uint8) for _ in range(4)]
    idx = 0
    for i in range(best_n):
        for j in range(i + 1, best_n):
            c = best_colors[idx]
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    valid = all(_c6_count_from_adj(adj, best_n) == 0 for adj in adjs)
    
    if not valid:
        n = 9
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
