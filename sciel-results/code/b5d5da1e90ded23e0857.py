import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_COLORS = 120

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # Precompute edge indices
    edges = []
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            idx += 1
            
    # Precompute 4-cycles
    # A 4-cycle is defined by 4 vertices. There are 3 distinct 4-cycles per set of 4 vertices.
    # Cycle 1: v0-v1-v2-v3-v0
    # Cycle 2: v0-v1-v3-v2-v0
    # Cycle 3: v0-v2-v1-v3-v0
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v0 in range(N):
        for v1 in range(v0 + 1, N):
            for v2 in range(v1 + 1, N):
                for v3 in range(v2 + 1, N):
                    # Cycle 1: v0-v1-v2-v3-v0
                    e1 = edge_index[(v0, v1)]
                    e2 = edge_index[(v1, v2)]
                    e3 = edge_index[(v2, v3)]
                    e4 = edge_index[(v3, v0)]
                    cycles.append((e1, e2, e3, e4))
                    edge_to_cycles[e1].append(len(cycles) - 1)
                    edge_to_cycles[e2].append(len(cycles) - 1)
                    edge_to_cycles[e3].append(len(cycles) - 1)
                    edge_to_cycles[e4].append(len(cycles) - 1)
                    
                    # Cycle 2: v0-v1-v3-v2-v0
                    e1 = edge_index[(v0, v1)]
                    e2 = edge_index[(v1, v3)]
                    e3 = edge_index[(v3, v2)]
                    e4 = edge_index[(v2, v0)]
                    cycles.append((e1, e2, e3, e4))
                    edge_to_cycles[e1].append(len(cycles) - 1)
                    edge_to_cycles[e2].append(len(cycles) - 1)
                    edge_to_cycles[e3].append(len(cycles) - 1)
                    edge_to_cycles[e4].append(len(cycles) - 1)
                    
                    # Cycle 3: v0-v2-v1-v3-v0
                    e1 = edge_index[(v0, v2)]
                    e2 = edge_index[(v2, v1)]
                    e3 = edge_index[(v1, v3)]
                    e4 = edge_index[(v3, v0)]
                    cycles.append((e1, e2, e3, e4))
                    edge_to_cycles[e1].append(len(cycles) - 1)
                    edge_to_cycles[e2].append(len(cycles) - 1)
                    edge_to_cycles[e3].append(len(cycles) - 1)
                    edge_to_cycles[e4].append(len(cycles) - 1)

    def is_valid_assignment(edge_idx, new_color, colors):
        """
        Check if assigning new_color to edge_idx results in any rainbow 4-cycle.
        """
        for c_idx in edge_to_cycles[edge_idx]:
            e1, e2, e3, e4 = cycles[c_idx]
            # Find the other 3 edges
            other_edges = []
            if e1 == edge_idx:
                other_edges = [e2, e3, e4]
            elif e2 == edge_idx:
                other_edges = [e1, e3, e4]
            elif e3 == edge_idx:
                other_edges = [e1, e2, e4]
            else:
                other_edges = [e1, e2, e3]
            
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            
            # If any of the other edges has the new color, the cycle is not rainbow
            # because the new color is repeated.
            if c1 == new_color or c2 == new_color or c3 == new_color:
                continue
                
            # If the other 3 edges do not have the new color, they must not be all distinct.
            # If they are all distinct, the cycle will have colors {new_color, c1, c2, c3}
            # which are 4 distinct colors.
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
                
        return True

    best_result = [0] * NUM_EDGES
    best_score = 1
    
    # Greedy construction
    # We try to assign a unique color to each edge if possible.
    # We use a random order for edges to break symmetries.
    
    for attempt in range(50): # Try a few different random orders
        if time.time() - start_time > time_limit:
            break
            
        random.seed(attempt)
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        colors = [0] * NUM_EDGES
        used_colors = {0}
        available_colors = list(range(1, NUM_COLORS))
        random.shuffle(available_colors)
        
        for e_idx in edge_order:
            if time.time() - start_time > time_limit:
                break
                
            # Try to assign a new color from available_colors
            assigned = False
            # We want to maximize distinct colors, so we prefer new colors.
            # We iterate through available colors.
            # To make it faster, we can just try the first few or all.
            # Given the low constraint density, the first valid color is likely to be found quickly.
            
            # Optimization: If we just want to maximize count, we can try to assign
            # the smallest available color that works.
            
            for c in available_colors:
                if is_valid_assignment(e_idx, c, colors):
                    colors[e_idx] = c
                    used_colors.add(c)
                    available_colors.remove(c)
                    assigned = True
                    break
            
            # If no new color works, we can leave it as 0 (or try to recolor to an existing color? 
            # No, that doesn't increase the count. We might leave it as 0 to preserve 
            # the "safety" of 0 for other edges? Or we could assign a random existing color 
            # to potentially free up "rainbow potential"? 
            # Actually, keeping it 0 is safest as 0 is very common and thus 
            # "repeats" often, preventing rainbows.
            
        current_score = len(used_colors)
        if current_score > best_score:
            best_score = current_score
            best_result = colors[:]

    # Final check and return
    # Ensure we return a list of 120 integers
    return best_result
