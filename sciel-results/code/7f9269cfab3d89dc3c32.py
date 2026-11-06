"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses a structured initial coloring based on K4 decomposition and
    local search to maximize distinct colors while maintaining the C4-rainbow-free property.
    """
    deadline = time.time() + 55  # Safety margin for 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_edge(i, j):
        if i > j:
            i, j = j, i
        return edge_idx(i, j)
    
    # Precompute all 4-cycles as numpy arrays for vectorized validation
    all_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    all_cycles.append([get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a)])
                    # Cycle 2: a-b-d-c-a
                    all_cycles.append([get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a)])
                    # Cycle 3: a-c-b-d-a
                    all_cycles.append([get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a)])
    
    cycles_arr = np.array(all_cycles, dtype=np.int32)
    
    def is_valid_numpy(colors_np):
        """Vectorized check: no cycle has 4 distinct colors."""
        cycle_colors = colors_np[cycles_arr]
        sorted_colors = np.sort(cycle_colors, axis=1)
        diffs = np.abs(np.diff(sorted_colors, axis=1))
        rainbow_mask = np.all(diffs > 0, axis=1)
        return not np.any(rainbow_mask)
    
    def count_distinct(colors_list):
        return len(set(colors_list))
    
    def initial_coloring_structured():
        """
        Create a valid starting coloring using K4 decomposition.
        Partitions vertices into 4 groups of 4.
        Internal edges: 3 colors per group (12 total).
        External edges: 2 colors per group pair (12 total).
        Total 24 colors.
        """
        col = [0] * 120
        # Group assignment: g(v) = v // 4
        # Local index: l(v) = v % 4
        
        # Precompute internal edge colors for a single K4
        # Vertices 0,1,2,3
        # (0,1): 1, (2,3): 1
        # (0,2): 2, (1,3): 2
        # (0,3): 3, (1,2): 3
        internal_map = {
            (0,1): 1, (1,0): 1,
            (2,3): 1, (3,2): 1,
            (0,2): 2, (2,0): 2,
            (1,3): 2, (3,1): 2,
            (0,3): 3, (3,0): 3,
            (1,2): 3, (2,1): 3
        }
        
        for i in range(N):
            for j in range(i+1, N):
                gi = i // 4
                gj = j // 4
                li = i % 4
                lj = j % 4
                
                if gi == gj:
                    # Internal edge
                    c = internal_map[(li, lj)]
                    col[get_edge(i, j)] = gi * 3 + c
                else:
                    # External edge between Gi and Gj
                    # Use 2 colors based on parity of local indices
                    # Color 0 if li==lj, Color 1 if li!=lj ?
                    # Let's check if this creates rainbow C4 in K4,4
                    # Cycle u1,v1,u2,v2
                    # c(u1,v1) = 0 if u1==v1 else 1
                    # c(v1,u2) = 0 if v1==u2 else 1
                    # c(u2,v2) = 0 if u2==v2 else 1
                    # c(v2,u1) = 0 if v2==u1 else 1
                    # If u1=0, v1=0, u2=1, v2=1:
                    # c(0,0)=0, c(0,1)=1, c(1,1)=0, c(1,0)=1 -> {0,1} Safe.
                    # If u1=0, v1=1, u2=2, v2=3:
                    # c(0,1)=1, c(1,2)=1, c(2,3)=1, c(3,0)=1 -> {1} Safe.
                    # If u1=0, v1=1, u2=1, v2=2:
                    # c(0,1)=1, c(1,1)=0, c(1,2)=1, c(2,0)=1 -> {0,1} Safe.
                    # Seems safe.
                    base_color = (gi * 4 + gj) * 2 # Unique offset for this pair
                    # Ensure gi < gj for consistent offset? 
                    # We just need distinct colors for different pairs to maximize count
                    # Max pairs = 6. 6*2 = 12 colors.
                    if li == lj:
                        c = base_color
                    else:
                        c = base_color + 1
                    col[get_edge(i, j)] = c
                    
        return col

    # Start with structured coloring
    colors = initial_coloring_structured()
    colors_np = np.array(colors, dtype=np.int32)
    
    if not is_valid_numpy(colors_np):
        # Fallback if structured is somehow invalid (shouldn't be)
        colors = [min(i, j) for i in range(N) for j in range(i+1, N)]
        colors_np = np.array(colors, dtype=np.int32)
        
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    random.seed(42)
    np.random.seed(42)
    
    # Local search to add more colors
    # Strategy: Try to change an edge to a NEW color (not in used set)
    # If valid, accept. This directly increases score.
    
    used_colors = set(colors)
    
    while time.time() < deadline - 2:
        progress = False
        # Shuffle edges to avoid bias
        edges_to_try = list(range(120))
        random.shuffle(edges_to_try)
        
        for ei in edges_to_try:
            if time.time() > deadline - 2:
                break
            
            old_color = colors[ei]
            
            # Try to find a new color
            # We prefer colors that are not yet used
            # But we also try random ones to break symmetries
            candidates = []
            # Add some random new colors
            for _ in range(5):
                c = random.randint(0, 119)
                if c not in used_colors:
                    candidates.append(c)
            
            # If we have no new color candidates, try existing ones to rearrange?
            # For now, focus on adding new colors.
            if not candidates:
                continue
                
            random.shuffle(candidates)
            
            for new_color in candidates:
                colors[ei] = new_color
                colors_np[ei] = new_color
                
                if is_valid_numpy(colors_np):
                    # Accept
                    used_colors.add(new_color)
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_colors = colors[:]
                        best_count = new_count
                    progress = True
                    break
                else:
                    colors[ei] = old_color
                    colors_np[ei] = old_color
        
        if not progress:
            # If no progress, try a random restart of the local search state?
            # Or just break and try a different strategy?
            # Let's try a few random moves that don't increase color count but might enable future ones
            # This is a heuristic, might not help, but better than stopping.
            for _ in range(10):
                ei = random.randint(0, 119)
                old_color = colors[ei]
                new_color = random.randint(0, 119)
                if new_color == old_color:
                    continue
                colors[ei] = new_color
                colors_np[ei] = new_color
                if is_valid_numpy(colors_np):
                    # Accept even if count doesn't increase, to explore state space
                    pass
                else:
                    colors[ei] = old_color
                    colors_np[ei] = old_color

    return best_colors
# EVOLVE-BLOCK-END
