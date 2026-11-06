import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Edge indices: (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute K4s
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        # Get indices of the 6 edges in the quad
        e_indices = []
        for a, b in itertools.combinations(quad, 2):
            # Find index of edge (min(a,b), max(a,b))
            # Since edges are generated in lex order, we can map (i,j) to index
            # i*(n-1) - i*(i+1)//2 + (j-i-1) ... or just use a dictionary
            e_indices.append((min(a,b), max(a,b)))
        k4s.append(e_indices)
    
    # Map edge tuple to index for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Convert K4s to lists of edge indices
    k4_indices = []
    for e_indices in k4s:
        k4_indices.append([edge_to_idx[e] for e in e_indices])

    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for k4 in k4_indices:
            # Extract colors
            c0 = colors[k4[0]]
            c1 = colors[k4[1]]
            c2 = colors[k4[2]]
            c3 = colors[k4[3]]
            c4 = colors[k4[4]]
            c5 = colors[k4[5]]
            # Count distinct
            distinct = {c0, c1, c2, c3, c4, c5}
            if len(distinct) < 5:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Initial coloring: Seed
    # (i + j) % 17
    best_colors = [(i + j) % 17 for i, j in edges]
    best_count = count_colors(best_colors)
    
    if not is_valid(best_colors):
        # Fallback if seed is invalid (shouldn't happen)
        # Use a trivial valid coloring: all edges unique
        best_colors = list(range(num_edges))
        best_count = num_edges

    # Local Search: Merging colors
    # We will try to merge color A into color B.
    
    current_colors = best_colors[:]
    current_count = best_count
    
    # Randomize edge order or color order? 
    # We'll iterate through pairs of colors present in the current coloring.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current_colors)
        # Sort colors to have deterministic order, or randomize for diversity?
        # Let's try randomizing the order of pairs to escape symmetries
        color_list = list(current_set)
        random.shuffle(color_list)
        
        for c1 in color_list:
            for c2 in color_list:
                if c1 == c2:
                    continue
                # Try merging c1 into c2
                # Create a copy
                new_colors = current_colors[:]
                for i in range(num_edges):
                    if new_colors[i] == c1:
                        new_colors[i] = c2
                
                # Check validity
                if is_valid(new_colors):
                    new_count = count_colors(new_colors)
                    if new_count < current_count:
                        current_colors = new_colors
                        current_count = new_count
                        if current_count < best_count:
                            best_colors = current_colors[:]
                            best_count = current_count
                        improved = True
                        # Restart loop with new color set
                        break
            if improved:
                break
    
    # If time remains, try some random single-edge recolorings to break local optima?
    # Merging is powerful. If no merge works, we are at a local minimum for that metric.
    # We can try restarting the whole process with a slightly perturbed start if time allows.
    
    # For now, return the best found
    return best_colors

# EVOLVE-BLOCK-END
