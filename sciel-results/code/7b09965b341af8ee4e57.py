import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles contain it
_EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(_CYCLES):
    for e in cycle:
        _EDGE_TO_CYCLES[e].append(ci)

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing colors[edge_idx] to new_color maintains validity."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    for ci in _EDGE_TO_CYCLES[edge_idx]:
        e0, e1, e2, e3 = _CYCLES[ci]
        c0 = colors[e0] if e0 != edge_idx else new_color
        c1 = colors[e1] if e1 != edge_idx else new_color
        c2 = colors[e2] if e2 != edge_idx else new_color
        c3 = colors[e3] if e3 != edge_idx else new_color
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _build_hierarchical(vertices, start_color, splits, color_map):
    """
    Build a hierarchical Gallai coloring.
    """
    n = len(vertices)
    if n <= 1:
        return start_color
    if n == 2:
        u, v = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        return start_color + 1
    if n == 3:
        u, v, w = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        color_map[(min(v,w), max(v,w))] = start_color + 1
        color_map[(min(u,w), max(u,w))] = start_color + 2
        return start_color + 3
    
    k = splits.get(n, n // 2)
    v1 = vertices[:k]
    v2 = vertices[k:]
    
    # Cross edges get one shared color
    cross_color = start_color
    for u in v1:
        for v in v2:
            color_map[(min(u,v), max(u,v))] = cross_color
    
    next_color = start_color + 1
    next_color = _build_hierarchical(v1, next_color, splits, color_map)
    next_color = _build_hierarchical(v2, next_color, splits, color_map)
    return next_color

def _get_optimal_splits():
    """
    Compute the optimal split configuration to maximize colors in hierarchical coloring.
    f(m) = 1 + max(f(k) + f(m-k)) for m >= 4
    f(1)=0, f(2)=1, f(3)=3
    """
    f = [0] * (N + 1)
    f[1] = 0
    f[2] = 1
    f[3] = 3
    splits = {}
    
    for m in range(4, N + 1):
        max_val = -1
        best_k = m // 2
        for k in range(1, m):
            val = f[k] + f[m-k]
            if val > max_val:
                max_val = val
                best_k = k
        f[m] = 1 + max_val
        splits[m] = best_k
        
    return splits, f[N]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    random.seed(42)
    
    # Step 1: Generate best hierarchical coloring
    splits, max_hier_colors = _get_optimal_splits()
    
    color_map = {}
    _build_hierarchical(list(range(N)), 0, splits, color_map)
    
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[_edge_idx(i, j)] = color_map[(i, j)]
            
    current_score = _count_colors(colors)
    best_score = current_score
    best_colors = colors[:]
    
    # Maintain color counts for fast delta calculation
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
        
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    # Track no-improve to potentially restart
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new random initialization or break
            # Let's just break or reset T if stuck
            if no_improve > max_no_improve:
                # Try a random restart from a slightly perturbed hierarchical coloring
                # Or just stop if we've been stuck for a long time
                break
            T = min_T
            no_improve = 0
            
        edge_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[edge_idx]
        
        # Choose new color
        # Bias towards colors that are not currently used or are used rarely?
        # Actually, trying a completely new color is best for increasing count.
        # But we need to check validity.
        
        # Strategy: 
        # 1. Try a random unused color (if any)
        # 2. Try a random used color
        
        used_colors = set(color_counts.keys())
        unused_colors = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        
        if unused_colors and random.random() < 0.5:
            new_color = random.choice(unused_colors)
        else:
            new_color = random.randint(0, MAX_COLOR)
            
        if new_color == old_color:
            continue
            
        if _check_edge_change(colors, edge_idx, new_color):
            # Calculate delta
            # old_color count decreases by 1
            # new_color count increases by 1
            
            old_count = color_counts[old_color]
            new_count = color_counts.get(new_color, 0)
            
            # Determine change in distinct color count
            # Case 1: old_count becomes 0 -> lose 1 distinct color
            # Case 2: new_count was 0 -> gain 1 distinct color
            
            delta = 0
            if old_count == 1:
                delta -= 1
            if new_count == 0:
                delta += 1
                
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # delta is 0 or -1
                # If delta == 0, always accept to explore
                # If delta < 0, accept with prob exp(delta / T)
                if delta == 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta / T)
                    
            if accept:
                colors[edge_idx] = new_color
                # Update counts
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
                
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                T *= cooling_rate
            else:
                no_improve += 1
                T *= cooling_rate
        else:
            no_improve += 1
            T *= cooling_rate

    # Final polish: strict hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for edge_idx in range(NUM_EDGES):
            old_color = colors[edge_idx]
            # Try all colors? Too slow. Try a few promising ones.
            # Let's try all unused colors and the color of a neighbor edge?
            # For simplicity, just try a few random ones or skip if time is tight.
            # Given the time constraint, we'll do a quick pass.
            
            # Actually, let's just return the best found.
            # The SA should have done the heavy lifting.
            pass
            
    return best_colors
