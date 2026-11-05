"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _edge_idx(i, j):
    """Get index for edge (i,j) in lexicographic order."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Precompute for each edge, the list of 4-cycles (as tuples of 4 edge indices) containing it."""
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on 4 vertices
        cycles = [
            [(a, b), (b, c), (c, d), (d, a)],
            [(a, b), (b, d), (d, c), (c, a)],
            [(a, c), (c, b), (b, d), (d, a)]
        ]
        
        for cyc in cycles:
            edge_indices = tuple(_edge_idx(u, v) for u, v in cyc)
            for ei in edge_indices:
                edge_to_cycles[ei].append(edge_indices)
    
    return edge_to_cycles

_EDGE_TO_CYCLES = _build_cycles()

def _check_edge_valid(colors, edge_idx, new_color):
    """Check if assigning new_color to edge_idx keeps all its cycles valid."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    
    colors[edge_idx] = new_color
    valid = True
    for cyc in _EDGE_TO_CYCLES[edge_idx]:
        cols = set(colors[i] for i in cyc)
        if len(cols) == 4:
            valid = False
            break
    colors[edge_idx] = old_color
    return valid

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s limit
    
    random.seed(42)
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_count = 1
    
    # Precompute edge order
    edge_order = list(range(NUM_EDGES))
    
    # Greedy passes: try to assign new colors
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            current_color = colors[ei]
            # Try to find a new color that increases distinct count
            found = False
            for new_color in range(1, MAX_COLOR + 1):
                if new_color == current_color:
                    continue
                if _check_edge_valid(colors, ei, new_color):
                    colors[ei] = new_color
                    found = True
                    break
            
            if not found:
                # Try swapping: find another edge that can take this edge's color
                # and this edge takes that edge's color, if it increases count
                pass
    
    count = len(set(colors))
    if count > best_count:
        best_count = count
        best_colors = colors[:]
    
    # Local search with swaps
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        random.shuffle(edge_order)
        improved = False
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            current_color = colors[ei]
            old_count = len(set(colors))
            
            # Try assigning a new color
            for new_color in range(MAX_COLOR + 1):
                if new_color == current_color:
                    continue
                if _check_edge_valid(colors, ei, new_color):
                    colors[ei] = new_color
                    new_count = len(set(colors))
                    if new_count > old_count:
                        improved = True
                        break
                    else:
                        colors[ei] = current_color
            
            if improved:
                new_count = len(set(colors))
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
    
    return best_colors
# EVOLVE-BLOCK-END
