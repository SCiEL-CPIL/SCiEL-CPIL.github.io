import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge indices
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

# Precompute K4s and the edges within them
k4s = []
k4_edges = []  # List of lists of edge indices
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                k4s.append((a, b, c, d))
                e_ab = edge_to_idx[(a, b)]
                e_ac = edge_to_idx[(a, c)]
                e_ad = edge_to_idx[(a, d)]
                e_bc = edge_to_idx[(b, c)]
                e_bd = edge_to_idx[(b, d)]
                e_cd = edge_to_idx[(c, d)]
                k4_edges.append([e_ab, e_ac, e_ad, e_bc, e_bd, e_cd])

# Map edge index to list of K4 indices it participates in
edge_to_k4s = [[] for _ in range(NUM_EDGES)]
for k4_idx, edge_list in enumerate(k4_edges):
    for e_idx in edge_list:
        edge_to_k4s[e_idx].append(k4_idx)

def count_distinct_colors(edge_indices, colors):
    """Count distinct colors in a K4 given its edge indices."""
    c0 = colors[edge_indices[0]]
    c1 = colors[edge_indices[1]]
    c2 = colors[edge_indices[2]]
    c3 = colors[edge_indices[3]]
    c4 = colors[edge_indices[4]]
    c5 = colors[edge_indices[5]]
    
    # Simple set construction is fast enough for 6 elements
    return len({c0, c1, c2, c3, c4, c5})

def total_penalty(colors):
    """Calculate total penalty for a given coloring."""
    pen = 0
    for k4_idx in range(len(k4s)):
        n_col = count_distinct_colors(k4_edges[k4_idx], colors)
        if n_col < 5:
            pen += (5 - n_col) ** 2
    return pen

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # We prioritize finding a solution with 13 colors.
    # If not found, try 14, then 15.
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
        
        best_colours = None
        best_penalty = float('inf')
        
        # Number of restarts based on time available
        # Let's do a fixed number of restarts, each with a time limit
        max_restarts = 20
        time_per_restart = 4.0  # seconds
        
        for r in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize random coloring
            colors = np.random.randint(0, K, size=NUM_EDGES, dtype=np.int32)
            current_penalty = total_penalty(colors)
            
            if current_penalty == 0:
                return colors.tolist()
                
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colours = colors.copy()
                
            # Simulated Annealing with incremental updates
            T = 50.0
            T_min = 1.0
            cooling = 0.95
            
            t_start = time.time()
            
            while time.time() - t_start < time_per_restart:
                # Pick a random edge to change
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_c = colors[e_idx]
                
                # Pick a new color
                new_c = random.randint(0, K - 1)
                while new_c == old_c:
                    new_c = random.randint(0, K - 1)
                    
                # Calculate penalty change
                delta_penalty = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    # Get the edge indices for this K4
                    k4_e = k4_edges[k4_idx]
                    
                    # Calculate old distinct count
                    # We need to know the colors of the other 5 edges
                    # Since e_idx is one of them, we can construct the set
                    # To be fast, we just re-count using the current colors, 
                    # but we need to simulate the change.
                    
                    # Old state:
                    old_colors = set()
                    for e in k4_e:
                        if e == e_idx:
                            old_colors.add(old_c)
                        else:
                            old_colors.add(colors[e])
                    old_distinct = len(old_colors)
                    old_pen = max(0, 5 - old_distinct) ** 2
                    
                    # New state:
                    new_colors = set()
                    for e in k4_e:
                        if e == e_idx:
                            new_colors.add(new_c)
                        else:
                            new_colors.add(colors[e])
                    new_distinct = len(new_colors)
                    new_pen = max(0, 5 - new_distinct) ** 2
                    
                    delta_penalty += new_pen - old_pen
                
                # Acceptance criterion
                if delta_penalty < 0 or random.random() < np.exp(-delta_penalty / T):
                    colors[e_idx] = new_c
                    current_penalty += delta_penalty
                    
                    if current_penalty == 0:
                        return colors.tolist()
                        
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colours = colors.copy()
                else:
                    # Reject: colors[e_idx] remains old_c
                    pass
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    
        if best_penalty == 0:
            return best_colours.tolist()
            
    # Fallback: Return a valid coloring with 15 colors if found, or a seed
    # The seed (i+j)%17 uses 17 colors? No, 0..16 is 17 colors.
    # We need to ensure we return something valid if possible.
    # If we didn't find a 0 penalty, we return the best we had.
    # But the task requires a valid coloring? "A valid colouring certifies..."
    # If we can't find a valid one, we should probably return the best attempt
    # or a known valid one.
    # The seed program returns [(i + j) % 17 ...] which uses 17 colors and is valid.
    # But we want to minimize colors.
    # If we didn't find a 0 penalty, best_colours might be invalid.
    # We should check if best_colours is valid. If not, use seed.
    
    if best_colours is not None and total_penalty(best_colours) == 0:
        return best_colours.tolist()
        
    # Fallback to seed (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
