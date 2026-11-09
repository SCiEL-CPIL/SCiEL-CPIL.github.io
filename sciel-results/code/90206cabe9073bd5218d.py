"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

# For each K4, precompute the 6 edge indices
_K4_EDGE_IDXS = []
for edges in _K4_EDGES:
    _K4_EDGE_IDXS.append([_edge_idx(i, j) for (i, j) in edges])

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

# Precompute for each edge, the list of (k4_idx, pos_in_k4)
_EDGE_TO_K4_POS = [[] for _ in range(_NUM_EDGES)]
for k4_idx in range(_NUM_K4S):
    eidxs = _K4_EDGE_IDXS[k4_idx]
    for pos, eidx in enumerate(eidxs):
        _EDGE_TO_K4_POS[eidx].append((k4_idx, pos))

def _distinct_count(c0, c1, c2, c3, c4, c5):
    """Count distinct values among 6 colors."""
    distinct = 1
    if c1 != c0: distinct += 1
    if c2 != c0 and c2 != c1: distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
    return distinct

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors. colors is a numpy array of edge colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        c0 = colors[eidxs[0]]
        c1 = colors[eidxs[1]]
        c2 = colors[eidxs[2]]
        c3 = colors[eidxs[3]]
        c4 = colors[eidxs[4]]
        c5 = colors[eidxs[5]]
        if _distinct_count(c0, c1, c2, c3, c4, c5) < 5:
            bad += 1
    return bad

def _count_colors_used(colors):
    return len(set(colors.tolist()))

def _compute_delta_bad(c, edge_idx, new_color):
    """Compute change in bad K4 count if we set edge_idx to new_color."""
    old_color = c[edge_idx]
    if old_color == new_color:
        return 0
    
    delta_bad = 0
    for k4_idx, pos in _EDGE_TO_K4_POS[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        
        # Get colors of the 6 edges in this K4
        c0 = c[eidxs[0]]
        c1 = c[eidxs[1]]
        c2 = c[eidxs[2]]
        c3 = c[eidxs[3]]
        c4 = c[eidxs[4]]
        c5 = c[eidxs[5]]
        
        distinct_before = _distinct_count(c0, c1, c2, c3, c4, c5)
        bad_before = 1 if distinct_before < 5 else 0
        
        # Simulate new colors
        if pos == 0:
            distinct_after = _distinct_count(new_color, c1, c2, c3, c4, c5)
        elif pos == 1:
            distinct_after = _distinct_count(c0, new_color, c2, c3, c4, c5)
        elif pos == 2:
            distinct_after = _distinct_count(c0, c1, new_color, c3, c4, c5)
        elif pos == 3:
            distinct_after = _distinct_count(c0, c1, c2, new_color, c4, c5)
        elif pos == 4:
            distinct_after = _distinct_count(c0, c1, c2, c3, new_color, c5)
        else:
            distinct_after = _distinct_count(c0, c1, c2, c3, c4, new_color)
        
        bad_after = 1 if distinct_after < 5 else 0
        delta_bad += bad_after - bad_before
    
    return delta_bad

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with the known valid 17-coloring
    colors_17 = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    best_colors = colors_17.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    rng = random.Random(42)
    
    # Try to reduce colors from 16 down to 13
    for target_colors in [16, 15, 14, 13]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        for init_iter in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialization strategy:
            # If init_iter == 0, map 17-coloring to target_colors
            # Otherwise, random init
            if init_iter == 0:
                c = colors_17 % target_colors
            else:
                c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            
            # Local search
            temperature = 0.5
            cooling = 0.999
            max_iters = 50000
            no_improve_count = 0
            current_bad = bad
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if current_bad == 0:
                    break
                
                # Pick a random edge
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                old_color = c[edge_idx]
                
                # Try a new random color
                new_color = rng.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                delta_bad = _compute_delta_bad(c, edge_idx, new_color)
                
                if delta_bad < 0:
                    c[edge_idx] = new_color
                    current_bad += delta_bad
                    no_improve_count = 0
                elif delta_bad == 0:
                    # Accept neutral moves to explore
                    if rng.random() < 0.5:
                        c[edge_idx] = new_color
                else:
                    # Accept worse with probability based on temperature
                    if rng.random() < np.exp(-delta_bad / temperature):
                        c[edge_idx] = new_color
                        current_bad += delta_bad
                    else:
                        no_improve_count += 1
                        if no_improve_count > 1000:
                            # Restart temperature or break
                            temperature *= 0.9
                            no_improve_count = 0
                
                temperature *= cooling
            
            if current_bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                # Found valid for this target, move to next
                break
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to the original 17-coloring
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    # Return the best coloring found
    return best_colors.tolist()
# EVOLVE-BLOCK-END
