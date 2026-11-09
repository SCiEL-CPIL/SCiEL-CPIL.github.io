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

# Precompute for each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(_N * (_N - 1) // 2)]
def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

for k4_idx, edges in enumerate(_K4_EDGES):
    for (i, j) in edges:
        _EDGE_TO_K4S[_edge_idx(i, j)].append(k4_idx)

# For each K4, precompute the 6 edge indices
_K4_EDGE_IDXS = []
for edges in _K4_EDGES:
    _K4_EDGE_IDXS.append([_edge_idx(i, j) for (i, j) in edges])

# Precompute position of each edge within each K4 that contains it
# _EDGE_POS[k4_idx] = list of (edge_idx, position) for the 6 edges in that K4
_EDGE_POS = []
for eidxs in _K4_EDGE_IDXS:
    _EDGE_POS.append(list(zip(eidxs, range(6))))

# For fast delta computation: for each edge, store the K4s it's in and its position
_EDGE_K4_INFO = []  # list of (k4_idx, position) for each edge
for e in range(_NUM_EDGES):
    info = []
    for k4_idx in _EDGE_TO_K4S[e]:
        for eidx, pos in _EDGE_POS[k4_idx]:
            if eidx == e:
                info.append((k4_idx, pos))
                break
    _EDGE_K4_INFO.append(info)

_NUM_K4S = len(_K4S)
_NUM_EDGES = _N * (_N - 1) // 2  # 120

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
    for k4_idx, pos in _EDGE_K4_INFO[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
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

def _try_reduce_colors(c, current_colors, rng, deadline):
    """Try to reduce the number of colors used while maintaining validity."""
    used_colors = set(c.tolist())
    current_count = len(used_colors)
    
    # Try to eliminate each color one at a time
    for color_to_eliminate in sorted(used_colors):
        if time.time() > deadline - 5:
            break
        if color_to_eliminate not in set(c.tolist()):
            continue
        
        # Find all edges with this color
        edges_with_color = [i for i in range(_NUM_EDGES) if c[i] == color_to_eliminate]
        if not edges_with_color:
            continue
        
        # Try to remap each edge to a different color
        success = True
        for edge_idx in edges_with_color:
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in range(current_count):
                if new_color == color_to_eliminate:
                    continue
                delta = _compute_delta_bad(c, edge_idx, new_color)
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_delta <= 0:
                c[edge_idx] = best_new_color
            else:
                # Try with simulated annealing: accept with small probability
                if rng.random() < 0.05:
                    c[edge_idx] = best_new_color
                else:
                    success = False
                    break
        
        if success:
            # Verify the result is still valid
            if _count_bad_k4s(c) == 0:
                current_count -= 1
                # Update used_colors
                used_colors = set(c.tolist())
            else:
                # Roll back - this is complex, so just stop trying to eliminate this color
                pass
    
    return c, current_count

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Start with the known valid 17-coloring
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    best_colors = colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    
    rng = random.Random(42)
    
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for init_iter in range(100):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with target_colors
            c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            
            # Local search with simulated annealing
            temperature = 1.0
            cooling = 0.9998
            max_iters = 200000
            
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                if bad == 0:
                    # Valid coloring!
                    actual_colors = _count_colors_used(c)
                    if actual_colors < best_score:
                        best_score = actual_colors
                        best_colors = c.copy()
                    break
                
                # Pick a random edge and try changing it
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                old_color = c[edge_idx]
                
                # Try a new random color
                new_color = rng.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                delta_bad = _compute_delta_bad(c, edge_idx, new_color)
                
                if delta_bad <= 0:
                    c[edge_idx] = new_color
                    bad += delta_bad
                else:
                    # Accept worse with probability based on temperature
                    if rng.random() < np.exp(-delta_bad / temperature):
                        c[edge_idx] = new_color
                        bad += delta_bad
                
                temperature *= cooling
            
            if bad == 0:
                actual_colors = _count_colors_used(c)
                if actual_colors < best_score:
                    best_score = actual_colors
                    best_colors = c.copy()
                
                # Try to reduce colors further
                c_reduced, reduced_count = _try_reduce_colors(c.copy(), actual_colors, rng, deadline)
                if reduced_count < best_score:
                    best_score = reduced_count
                    best_colors = c_reduced.copy()
                
                # Found valid for this target, move to next
                break
    
    # Final check: ensure best_colors is valid
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to the original 17-coloring
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    # Return the best coloring found
    return best_colors.tolist()
# EVOLVE-BLOCK-END
