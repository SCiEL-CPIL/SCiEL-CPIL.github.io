import numpy as np
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _get_all_4cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append([
                        _edge_idx(a, b),
                        _edge_idx(b, c),
                        _edge_idx(c, d),
                        _edge_idx(d, a)
                    ])
                    cycles.append([
                        _edge_idx(a, c),
                        _edge_idx(c, b),
                        _edge_idx(b, d),
                        _edge_idx(d, a)
                    ])
                    cycles.append([
                        _edge_idx(a, b),
                        _edge_idx(b, d),
                        _edge_idx(d, c),
                        _edge_idx(c, a)
                    ])
    return np.array(cycles, dtype=np.int32)

CYCLES = _get_all_4cycles()

def _is_valid(colors):
    cycle_colors = colors[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    diffs = np.diff(sorted_colors, axis=1)
    is_rainbow = np.all(diffs != 0, axis=1)
    return not np.any(is_rainbow)

def _count_colors(colors):
    return len(np.unique(colors))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_score = 1
    
    # Start with a valid 2-coloring to ensure we start in the valid space
    current_colors = np.random.randint(0, 2, NUM_EDGES)
    if not _is_valid(current_colors):
        # Fallback, though 2-coloring of any graph is valid for 4-cycles?
        # A 2-coloring can have a rainbow 4-cycle? No, max 2 colors in a cycle.
        # So it is always valid.
        pass
        
    current_score = _count_colors(current_colors)
    best_colors = current_colors.copy()
    best_score = current_score
    
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        edge_idx = np.random.randint(0, NUM_EDGES)
        old_color = current_colors[edge_idx]
        
        # Try a new color
        new_color = np.random.randint(0, 120)
        
        if new_color == old_color:
            continue
            
        current_colors[edge_idx] = new_color
        
        if _is_valid(current_colors):
            new_score = _count_colors(current_colors)
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors.copy()
                
            if new_score >= current_score:
                current_score = new_score
            else:
                delta = new_score - current_score
                # Accept with probability exp(delta / T)
                if np.random.rand() < np.exp(delta / T):
                    current_score = new_score
                else:
                    current_colors[edge_idx] = old_color
        else:
            current_colors[edge_idx] = old_color
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
