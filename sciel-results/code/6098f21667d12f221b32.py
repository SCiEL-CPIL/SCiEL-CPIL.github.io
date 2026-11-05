"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Leave 40s safety margin
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx
    
    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    best_colours = None
    best_num_colors = 18  # Initialize with worse than current best (17)
    
    # Try a few random starts with different color counts
    # We prioritize lower color counts first
    for max_colors in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for this max_colors
        # Reduced trials to fit within time limit
        for trial in range(20):
            if time.time() > deadline:
                break
                
            # Random assignment using colors 0..max_colors-1
            colours = [random.randint(0, max_colors - 1) for _ in range(n_edges)]
            
            # Local search to fix K4 violations
            # We will iterate until no violations found or max iterations reached
            improved = True
            iterations = 0
            max_iterations = 2000  # Increased iterations since each step is faster
            
            while improved and iterations < max_iterations and time.time() < deadline:
                improved = False
                iterations += 1
                
                # Pick a random edge
                e_idx = random.randint(0, n_edges - 1)
                
                # Pick a random K4 containing this edge
                k4_indices = edge_to_k4s[e_idx]
                if not k4_indices:
                    continue
                k4_idx = random.choice(k4_indices)
                edges = k4_edges[k4_idx]
                
                # Check if this K4 is violated
                colors_in_k4 = set(colours[e] for e in edges)
                if len(colors_in_k4) < 5:
                    # It is violated. Try to fix it.
                    # We want to change one of the edges in this K4 to a color not in colors_in_k4.
                    # Let's try changing the edge e_idx we picked.
                    # Find a candidate color not in colors_in_k4
                    candidate_colors = [c for c in range(max_colors) if c not in colors_in_k4]
                    if candidate_colors:
                        new_color = random.choice(candidate_colors)
                        colours[e_idx] = new_color
                        improved = True
                    else:
                        # This shouldn't happen if max_colors >= 5 and len(colors_in_k4) < 5
                        # But if it does, we can't fix it with this edge. 
                        # We might need to change another edge in the K4.
                        # For simplicity, we just skip this step (improved remains False)
                        pass
            
            # After local search, check if we have a valid solution
            # Only do full check if we think we might be valid (e.g., if loop ended without 'improved' being False due to no violations? 
            # Actually, the loop ends when improved is False (no violation found for random edge) or max_iter.
            # It's possible there are still violations. We must verify.
            
            # Optimization: Only verify if we have spent some time, or just verify always?
            # Verification is O(1820 * 6) which is fast (~10k ops).
            valid = True
            for edges in k4_edges:
                if len(set(colours[e] for e in edges)) < 5:
                    valid = False
                    break
            
            if valid:
                num_colors = len(set(colours))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colours = colours[:]
                    # If we found 13, we can't do better, so we can return early?
                    # The lower bound is 13. So yes.
                    if num_colors == 13:
                        return best_colours
    
    # If we found a better solution, use it. Otherwise, fall back to the 17-color solution.
    if best_colours is not None:
        return best_colours
    else:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
