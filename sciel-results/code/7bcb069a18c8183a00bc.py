import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(u, v):
    if u > v:
        u, v = v, u
    return u * N - u * (u + 1) // 2 + (v - u - 1)

def _precompute_c4s():
    c4s = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                for l in range(k + 1, N):
                    # Three C4s on vertices i,j,k,l
                    # 1. i-j-k-l-i
                    c4s.append((i, j, k, l))
                    # 2. i-j-l-k-i
                    c4s.append((i, j, l, k))
                    # 3. i-k-j-l-i
                    c4s.append((i, k, j, l))
    
    # Map edge index to list of C4 indices
    edge_to_c4s = [[] for _ in range(NUM_EDGES)]
    for idx, c4 in enumerate(c4s):
        e1 = _edge_idx(c4[0], c4[1])
        e2 = _edge_idx(c4[1], c4[2])
        e3 = _edge_idx(c4[2], c4[3])
        e4 = _edge_idx(c4[3], c4[0])
        edge_to_c4s[e1].append(idx)
        edge_to_c4s[e2].append(idx)
        edge_to_c4s[e3].append(idx)
        edge_to_c4s[e4].append(idx)
        
    return c4s, edge_to_c4s

C4S, EDGE_TO_C4S = _precompute_c4s()

def _is_rainbow(c4_idx, coloring):
    c4 = C4S[c4_idx]
    c1 = coloring[_edge_idx(c4[0], c4[1])]
    c2 = coloring[_edge_idx(c4[1], c4[2])]
    c3 = coloring[_edge_idx(c4[2], c4[3])]
    c4_c = coloring[_edge_idx(c4[3], c4[0])]
    return len({c1, c2, c3, c4_c}) == 4

def _try_change(coloring, edge_idx, new_color, used_colors):
    old_color = coloring[edge_idx]
    if old_color == new_color:
        return True
    
    coloring[edge_idx] = new_color
    
    # Check all C4s involving this edge
    for c4_idx in EDGE_TO_C4S[edge_idx]:
        if _is_rainbow(c4_idx, coloring):
            coloring[edge_idx] = old_color
            return False
            
    # Update used_colors set if necessary (handled outside for efficiency in some cases, 
    # but here we just return success and let caller update)
    return True

def construct():
    start_time = time.time()
    time_limit = 50.0 # Safety margin
    
    # Initial coloring: all 0
    coloring = [0] * NUM_EDGES
    used_colors = {0}
    best_coloring = coloring[:]
    best_count = 1
    
    # Greedy phase: try to add new colors
    # We will iterate until time runs out or no progress
    # To avoid getting stuck, we can randomize the order of edges and colors
    
    # Pre-generate a list of edges
    edges = list(range(NUM_EDGES))
    random.shuffle(edges)
    
    # We want to maximize the number of distinct colors.
    # Current strategy: 
    # 1. Try to assign a new unused color to a random edge.
    # 2. If successful, count increases.
    # 3. If we run out of easy improvements, we might need to swap colors or reset.
    
    # Since the space of valid colorings is complex, a simple greedy might get stuck at a low number.
    # Let's try multiple restarts.
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Reset to all 0s for fresh start? Or keep best?
        # Fresh start allows exploring different local optima.
        coloring = [0] * NUM_EDGES
        used_colors = {0}
        current_count = 1
        
        # Shuffle edges for this restart
        random.shuffle(edges)
        
        # We will try to add colors 1, 2, ... 119
        # But we can pick any unused color.
        # Let's iterate through potential new colors.
        
        # To speed up, we can try to add colors in a random order.
        available_colors = list(range(1, 120))
        random.shuffle(available_colors)
        
        for new_color in available_colors:
            if time.time() - start_time > time_limit:
                break
                
            # Try to place new_color on a random edge
            # If we try all edges, it's slow. Let's try a few random edges.
            success = False
            # Try up to 50 random edges for this new color
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                e = random.choice(edges)
                if _try_change(coloring, e, new_color, used_colors):
                    used_colors.add(new_color)
                    current_count += 1
                    success = True
                    break
            
            if not success:
                # If we can't place this color, maybe we should try a different order or stop?
                # For now, we just skip to next color.
                pass
        
        if current_count > best_count:
            best_count = current_count
            best_coloring = coloring[:]
            
    return best_coloring
