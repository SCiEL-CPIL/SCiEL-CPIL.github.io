# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all K4s
_N = 16
_K4S = list(combinations(range(_N), 4))
_K4_EDGES = []
for quad in _K4S:
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            edges.append((i, j))
    _K4_EDGES.append(edges)

# Precompute for each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(_N * (_N - 1) // 2)]
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

for k4_idx, edges in enumerate(_K4_EDGES):
    for (i, j) in edges:
        _EDGE_TO_K4S[_edge_idx(i, j)].append(k4_idx)

# Precompute for each K4, the 6 edge indices
_K4_EDGE_IDXS = []
for k4_idx, edges in enumerate(_K4_EDGES):
    idxs = [_edge_idx(i, j) for (i, j) in edges]
    _K4_EDGE_IDXS.append(idxs)

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

# Vectorized distinct count for all K4s
# _K4_EDGE_IDXS is a list of lists of 6 indices. We can make it a 2D array.
_K4_EDGE_IDXS_ARR = np.array(_K4_EDGE_IDXS, dtype=np.int32)

def _count_bad_k4s_vectorized(colors):
    """
    Vectorized count of K4s with < 5 distinct colors.
    colors: numpy array of shape (120,)
    """
    # Get colors for all edges in all K4s
    # Shape: (NUM_K4S, 6)
    k4_colors = colors[_K4_EDGE_IDXS_ARR]
    
    # Count distinct colors in each row using a faster approach
    # Sort each row and count unique values
    sorted_colors = np.sort(k4_colors, axis=1)
    # Count changes in sorted array
    diffs = np.diff(sorted_colors, axis=1)
    unique_counts = 1 + np.sum(diffs != 0, axis=1)
    bad = np.sum(unique_counts < 5)
    return int(bad)

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _delta_bad_for_edge_change(c, edge_idx, new_color, bad_k4_flags):
    """Compute change in bad K4 count when changing edge_idx to new_color."""
    old_color = c[edge_idx]
    if old_color == new_color:
        return 0
    
    delta = 0
    for k4_idx in _EDGE_TO_K4S[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = c[eidxs[0]]
        c1 = c[eidxs[1]]
        c2 = c[eidxs[2]]
        c3 = c[eidxs[3]]
        c4 = c[eidxs[4]]
        c5 = c[eidxs[5]]
        
        was_bad = bad_k4_flags[k4_idx]
        
        # Find position of edge_idx in eidxs
        pos = _EDGE_POS_IN_K4.get((k4_idx, edge_idx), -1)
        if pos == -1: continue # Should not happen
        
        if pos == 0: nv = (new_color, c1, c2, c3, c4, c5)
        elif pos == 1: nv = (c0, new_color, c2, c3, c4, c5)
        elif pos == 2: nv = (c0, c1, new_color, c3, c4, c5)
        elif pos == 3: nv = (c0, c1, c2, new_color, c4, c5)
        elif pos == 4: nv = (c0, c1, c2, c3, new_color, c5)
        else: nv = (c0, c1, c2, c3, c4, new_color)
        
        # Count distinct
        c0n, c1n, c2n, c3n, c4n, c5n = nv
        d = 1
        if c1n != c0n: d += 1
        if c2n != c0n and c2n != c1n: d += 1
        if c3n != c0n and c3n != c1n and c3n != c2n: d += 1
        if c4n != c0n and c4n != c1n and c4n != c2n and c4n != c3n: d += 1
        if c5n != c0n and c5n != c1n and c5n != c2n and c5n != c3n and c5n != c4n: d += 1
        
        is_bad = d < 5
        
        if was_bad and not is_bad:
            delta -= 1
        elif not was_bad and is_bad:
            delta += 1
    
    return delta

# Precompute positions for faster lookup
_EDGE_POS_IN_K4 = {}
for k4_idx, edges in enumerate(_K4_EDGES):
    for pos, (i, j) in enumerate(edges):
        _EDGE_POS_IN_K4[(k4_idx, _edge_idx(i, j))] = pos

def _reduce_colors(c, max_colors, deadline, rng):
    """Try to reduce the number of colors used while maintaining validity."""
    current_colors = _count_colors_used(c)
    if current_colors <= max_colors:
        return c, current_colors
    
    # Get the set of colors currently used
    color_set = set(c.tolist())
    
    # Try to eliminate colors one by one, starting from the least frequently used
    color_counts = {}
    for col in c:
        color_counts[col] = color_counts.get(col, 0) + 1
    
    # Sort colors by frequency (ascending)
    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
    
    for color_to_eliminate in sorted_colors:
        if time.time() > deadline - 10:
            break
        if _count_colors_used(c) <= max_colors:
            break
        
        # Try to recolor all edges with this color
        edges_with_color = [i for i in range(_NUM_EDGES) if c[i] == color_to_eliminate]
        
        for edge_idx in edges_with_color:
            if time.time() > deadline - 10:
                break
            if _count_colors_used(c) <= max_colors:
                break
            
            # Try each other color
            for new_color in color_set:
                if new_color == color_to_eliminate:
                    continue
                
                old_color = c[edge_idx]
                c[edge_idx] = new_color
                
                if _count_bad_k4s_vectorized(c) == 0:
                    if _count_colors_used(c) < current_colors:
                        current_colors = _count_colors_used(c)
                else:
                    c[edge_idx] = old_color
                    break  # This new color doesn't work, try next
    
    return c, _count_colors_used(c)

def construct():
    deadline = time.time() + 130  # safety margin for 150s limit
    
    # 1. Try simple modular colorings
    for K in [15, 14, 13]:
        if time.time() > deadline - 15:
            break
        c_mod = np.array([(i + j) % K for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
        if _count_bad_k4s_vectorized(c_mod) == 0:
            # Try to reduce colors
            c_mod, _ = _reduce_colors(c_mod, K - 1, deadline, random.Random(42))
            return c_mod.tolist()
            
    # 2. Local Search with Simulated Annealing
    best_colors = None
    best_score = 100
    
    # Seeds
    seeds = []
    # Modular seed for 15
    seeds.append(np.array([(i + j) % 15 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32))
    # Random seeds
    rng = random.Random(42)
    for _ in range(5):
        seeds.append(np.array([rng.randint(0, 14) for _ in range(_NUM_EDGES)], dtype=np.int32))
        
    for target_K in [15, 14, 13]:
        if time.time() > deadline - 30:
            break
            
        for c_init in seeds:
            if time.time() > deadline - 10:
                break
                
            c = c_init.copy()
            bad = _count_bad_k4s_vectorized(c)
            bad_flags = np.zeros(_NUM_K4S, dtype=bool)
            # Compute initial flags
            for k4_idx in range(_NUM_K4S):
                eidxs = _K4_EDGE_IDXS[k4_idx]
                c0, c1, c2, c3, c4, c5 = (c[eidxs[p]] for p in range(6))
                d = 1
                if c1 != c0: d += 1
                if c2 != c0 and c2 != c1: d += 1
                if c3 != c0 and c3 != c1 and c3 != c2: d += 1
                if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
                if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
                bad_flags[k4_idx] = (d < 5)
                
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.9995
            
            max_iters = 150000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if bad == 0:
                    actual_colors = _count_colors_used(c)
                    if actual_colors < best_score:
                        best_score = actual_colors
                        best_colors = c.copy()
                    break
                    
                # Pick an edge
                # Bias towards edges in bad K4s
                if rng.random() < 0.8 and bad > 0:
                    bad_k4s = np.where(bad_flags)[0]
                    k4_idx = bad_k4s[rng.randint(0, len(bad_k4s) - 1)]
                    edges_in_k4 = _K4_EDGE_IDXS[k4_idx]
                    edge_idx = edges_in_k4[rng.randint(0, 5)]
                else:
                    edge_idx = rng.randint(0, _NUM_EDGES - 1)
                    
                new_color = rng.randint(0, target_K - 1)
                
                delta = _delta_bad_for_edge_change(c, edge_idx, new_color, bad_flags)
                
                # Acceptance
                if delta <= 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / T)
                    
                if accept:
                    old_color = c[edge_idx]
                    c[edge_idx] = new_color
                    bad += delta
                    # Update flags
                    for k4_idx in _EDGE_TO_K4S[edge_idx]:
                        eidxs = _K4_EDGE_IDXS[k4_idx]
                        c0, c1, c2, c3, c4, c5 = (c[eidxs[p]] for p in range(6))
                        d = 1
                        if c1 != c0: d += 1
                        if c2 != c0 and c2 != c1: d += 1
                        if c3 != c0 and c3 != c1 and c3 != c2: d += 1
                        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
                        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
                        bad_flags[k4_idx] = (d < 5)
                        
                T *= cooling
                if T < T_min:
                    T = T_min
                    
            if bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                break # Found valid for this target
                
    # Try to reduce colors in the best solution found
    if best_colors is not None and _count_bad_k4s_vectorized(best_colors) == 0:
        best_colors, final_score = _reduce_colors(best_colors, best_score - 1, deadline, rng)
        if final_score < best_score:
            best_score = final_score
        return best_colors.tolist()
        
    # Fallback
    return np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32).tolist()
# EVOLVE-BLOCK-END
