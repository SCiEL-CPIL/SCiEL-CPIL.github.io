"""
Edge-coloring of K_16 to maximize distinct colors with no rainbow C4.
Combines structured baseline + efficient local search from current program
with simulated-annealing-style exploration and numpy-based validation from inspiration.
"""
import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# --- Precomputation (done once at module load) ---

def _build_edges():
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1
    return edges, edge_idx

EDGES, EDGE_IDX = _build_edges()

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((EDGE_IDX[(a,b)], EDGE_IDX[(b,c)], EDGE_IDX[(c,d)], EDGE_IDX[(a,d)]))
                    # Cycle 2: a-b, b-d, c-d, a-c
                    cycles.append((EDGE_IDX[(a,b)], EDGE_IDX[(b,d)], EDGE_IDX[(c,d)], EDGE_IDX[(a,c)]))
                    # Cycle 3: a-c, b-c, b-d, a-d
                    cycles.append((EDGE_IDX[(a,c)], EDGE_IDX[(b,c)], EDGE_IDX[(b,d)], EDGE_IDX[(a,d)]))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# For each edge, list of cycle indices it belongs to
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)

# For fast numpy validation: precompute cycles as a 2D array
CYCLES_ARR = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def _count_violations_np(colors_np):
    """Count rainbow 4-cycles using numpy vectorization."""
    # colors_np: shape (120,)
    # For each cycle, get the 4 colors
    c = colors_np[CYCLES_ARR]  # shape (NUM_CYCLES, 4)
    # A cycle is rainbow iff all 4 colors are distinct
    # Check: max != min OR use set logic. Fast way: 
    # Sort each row and check no adjacent equals
    c_sorted = np.sort(c, axis=1)
    # rainbow iff c_sorted[:,0] != c_sorted[:,1] and c_sorted[:,1] != c_sorted[:,2] and c_sorted[:,2] != c_sorted[:,3]
    mask = (c_sorted[:, 0] != c_sorted[:, 1]) & \
           (c_sorted[:, 1] != c_sorted[:, 2]) & \
           (c_sorted[:, 2] != c_sorted[:, 3])
    return int(np.sum(mask))

def _count_violations(colors):
    """Count rainbow 4-cycles (list-based)."""
    count = 0
    for cycle in CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            count += 1
    return count

def _count_distinct(colors):
    return len(set(colors))

def _check_edge_change(colors, edge_i, new_color):
    """Check if changing edge_i to new_color creates/removes violations.
    Returns (num_new, num_removed)."""
    old_color = colors[edge_i]
    if old_color == new_color:
        return 0, 0
    
    num_new = 0
    num_removed = 0
    for ci in EDGE_TO_CYCLES[edge_i]:
        cycle = CYCLES[ci]
        others = [e for e in cycle if e != edge_i]
        o0, o1, o2 = others
        c0, c1, c2 = colors[o0], colors[o1], colors[o2]
        
        # Are the other three already all distinct?
        others_distinct = (c0 != c1 and c0 != c2 and c1 != c2)
        if not others_distinct:
            continue  # Can't be a violation regardless of this edge's color
        
        was_violation = (old_color != c0 and old_color != c1 and old_color != c2)
        is_violation = (new_color != c0 and new_color != c1 and new_color != c2)
        
        if is_violation and not was_violation:
            num_new += 1
        elif was_violation and not is_violation:
            num_removed += 1
    
    return num_new, num_removed

def _make_structured_baseline():
    """
    Structured coloring based on 4x4 grid coordinates.
    Edge (i,j) color = f(coords(i), coords(j)).
    This gives a valid coloring with ~16-25 distinct colors.
    """
    colors = [0] * NUM_EDGES
    vertex_coords = [(i // 4, i % 4) for i in range(16)]
    
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = vertex_coords[i]
            xj, yj = vertex_coords[j]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            # Color based on (dx, dy) — at most 4*4=16 values
            colors[EDGE_IDX[(i, j)]] = dx * 4 + dy
    return colors

def _make_valid_random_baseline(rng):
    """
    Build a valid coloring by greedy assignment with backtracking-lite.
    Start with all zeros, then try to introduce new colors one by one.
    """
    colors = [0] * NUM_EDGES
    used = {0}
    
    # Try to add new colors to random edges
    for _ in range(300):
        ei = rng.randint(0, NUM_EDGES - 1)
        old = colors[ei]
        # Try a new color
        nc = rng.randint(0, MAX_COLOR - 1)
        if nc == old:
            continue
        num_new, num_removed = _check_edge_change(colors, ei, nc)
        if num_new == 0:
            colors[ei] = nc
            used.add(nc)
    
    # Verify
    if _count_violations(colors) == 0:
        return colors
    return None

def _local_search_maximize_colors(colors, deadline, rng):
    """
    Local search to maximize distinct colors while maintaining validity.
    Efficient approach: track violations incrementally, try new colors aggressively.
    """
    colors = list(colors)
    used_colors = set(colors)
    best_colors = list(colors)
    best_distinct = len(used_colors)
    
    t0 = time.time()
    
    # Precompute: for each edge, the set of colors that would make it safe
    # Strategy: repeatedly try to assign new colors to edges where it's safe
    
    # Main loop: try to add new colors
    while True:
        if time.time() - t0 > deadline:
            break
        
        improved = False
        # Try multiple random edges per iteration
        for _ in range(50):
            ei = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[ei]
            
            # Try several candidate new colors (unused)
            for _ in range(12):
                nc = rng.randint(0, MAX_COLOR - 1)
                if nc == old_color or nc in used_colors:
                    continue
                
                num_new, num_removed = _check_edge_change(colors, ei, nc)
                if num_new == 0:
                    colors[ei] = nc
                    used_colors.add(nc)
                    improved = True
                    if len(used_colors) > best_distinct:
                        best_distinct = len(used_colors)
                        best_colors = list(colors)
                    break
            
            if improved:
                break
        
        if not improved:
            # Try to recolor edges to free up space for new colors
            # Pick an edge with a color used by few other edges, try to change it
            from collections import Counter
            color_counts = Counter(colors)
            
            # Find colors used by only 1-2 edges
            rare_colors = [c for c, cnt in color_counts.items() if cnt <= 2]
            
            found_change = False
            for _ in range(20):
                if not rare_colors:
                    break
                c = rng.choice(rare_colors)
                # Find an edge with this color
                edges_with_c = [i for i in range(NUM_EDGES) if colors[i] == c]
                if not edges_with_c:
                    continue
                ei = rng.choice(edges_with_c)
                
                # Try to change to a new color
                for _ in range(8):
                    nc = rng.randint(0, MAX_COLOR - 1)
                    if nc == c or nc in used_colors:
                        continue
                    num_new, _ = _check_edge_change(colors, ei, nc)
                    if num_new == 0:
                        colors[ei] = nc
                        used_colors.add(nc)
                        found_change = True
                        if len(used_colors) > best_distinct:
                            best_distinct = len(used_colors)
                            best_colors = list(colors)
                        break
                
                if found_change:
                    break
            
            if not found_change:
                # Last resort: try any valid recoloring to escape local optimum
                any_valid = False
                for _ in range(10):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    nc = rng.randint(0, MAX_COLOR - 1)
                    if nc == colors[ei]:
                        continue
                    num_new, _ = _check_edge_change(colors, ei, nc)
                    if num_new == 0:
                        colors[ei] = nc
                        any_valid = True
                        break
                
                if not any_valid:
                    break
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit
    
    rng = random.Random(12345)
    
    best_colors = None
    best_distinct = 0
    
    # Strategy 1: Structured baseline + local search
    base1 = _make_structured_baseline()
    v1 = _count_violations(base1)
    if v1 == 0:
        improved, d1 = _local_search_maximize_colors(base1, DEADLINE - (time.time() - start_time), rng)
        if d1 > best_distinct:
            best_distinct = d1
            best_colors = improved
    
    # Strategy 2: Multiple random restarts with structured initialization
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        for r in range(5):
            if time.time() - start_time > DEADLINE - 2:
                break
            
            # Create a diverse structured baseline
            colors = [0] * NUM_EDGES
            vertex_coords = [(i // 4, i % 4) for i in range(16)]
            offset = rng.randint(0, 50)
            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi = vertex_coords[i]
                    xj, yj = vertex_coords[j]
                    dx = abs(xi - xj)
                    dy = abs(yi - yj)
                    if dx == 0:
                        c = (dy * 20 + offset) % MAX_COLOR
                    elif dy == 0:
                        c = (dx * 20 + offset + 1) % MAX_COLOR
                    else:
                        c = (dx * 10 + dy + offset + 40) % MAX_COLOR
                    colors[EDGE_IDX[(i, j)]] = c
            
            # Check validity
            if _count_violations(colors) == 0:
                rem = DEADLINE - (time.time() - start_time)
                if rem > 2:
                    improved, d = _local_search_maximize_colors(colors, rem, rng)
                    if d > best_distinct:
                        best_distinct = d
                        best_colors = improved
    
    # Strategy 3: Random valid baseline + local search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 3:
        base2 = _make_valid_random_baseline(rng)
        if base2 is not None:
            improved2, d2 = _local_search_maximize_colors(base2, remaining - 1, rng)
            if d2 > best_distinct:
                best_distinct = d2
                best_colors = improved2
    
    # Final validation
    if best_colors is None:
        # Fallback: all zeros
        return [0] * NUM_EDGES
    
    final_v = _count_violations(best_colors)
    if final_v > 0:
        # Fallback
        return [0] * NUM_EDGES
    
    return best_colors


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    print(f"Time: {elapsed:.2f}s")
    # Validate
    v = _count_violations(result)
    print(f"Violations: {v}")
