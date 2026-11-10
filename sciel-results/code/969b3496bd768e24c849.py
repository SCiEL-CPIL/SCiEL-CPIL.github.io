import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Precompute all K4s and their edge indices
    n = 16
    # Build edge index lookup: (i,j) -> index in the 120-element array
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    # Precompute all K4s as tuples of 6 edge indices
    k4_edges = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        edges = [
            edge_idx[(a, b)],
            edge_idx[(a, c)],
            edge_idx[(a, d)],
            edge_idx[(b, c)],
            edge_idx[(b, d)],
            edge_idx[(c, d)]
        ]
        k4_edges.append(edges)
    k4_edges = np.array(k4_edges)  # shape (1820, 6)
    
    def generate_seed():
        return np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    def count_violations(colours):
        """Count K4s that have fewer than 5 distinct colours."""
        # colours is a numpy array of shape (120,)
        # For each K4, extract the 6 edge colours
        edge_colors = colours[k4_edges]  # shape (1820, 6)
        
        # Count distinct colours per K4
        # A K4 violates if it has < 5 distinct colours
        # Equivalent: max frequency >= 2 AND number of distinct < 5
        # Simpler: check if len(set) < 5
        
        # Vectorized approach: for each K4, check if any colour appears >= 2 times
        # and total distinct < 5
        # Actually, let's just count distinct directly
        # For 1820 K4s with 6 edges each, we can do:
        distinct_counts = np.zeros(len(edge_colors), dtype=np.int32)
        for k in range(len(edge_colors)):
            distinct_counts[k] = len(set(edge_colors[k].tolist()))
        
        return np.sum(distinct_counts < 5)
    
    def is_valid(colours):
        return count_violations(colours) == 0
    
    current = generate_seed()
    
    # Try to reduce colours by merging
    def try_merge(colours, c1, c2):
        """Try to merge colour c1 into c2. Returns new colours if valid, else None."""
        new_colours = colours.copy()
        mask = new_colours == c1
        if not np.any(mask):
            return None
        new_colours[mask] = c2
        if is_valid(new_colours):
            return new_colours
        return None
    
    # Greedy merging: try to merge pairs of colours
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        current_colours = set(current.tolist())
        if len(current_colours) <= 13:
            break
            
        sorted_colours = sorted(current_colours)
        merged = False
        
        for i in range(len(sorted_colours)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(sorted_colours)):
                if time.time() > deadline:
                    break
                c1 = sorted_colours[i]
                c2 = sorted_colours[j]
                
                new_colours = try_merge(current, c1, c2)
                if new_colours is not None:
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            break
    
    # Local search: try changing individual edge colours to reduce count
    for iteration in range(500):
        if time.time() > deadline:
            break
            
        current_colours = set(current.tolist())
        if len(current_colours) <= 13:
            break
        
        # Try changing each edge to a different colour
        improved = False
        for edge_i in range(len(current)):
            if time.time() > deadline:
                break
            orig_colour = current[edge_i]
            
            # Try all colours from 0 to max_colour+1
            max_c = max(current_colours)
            for new_c in range(max_c + 2):
                if new_c == orig_colour:
                    continue
                
                new_colours = current.copy()
                new_colours[edge_i] = new_c
                
                if is_valid(new_colours):
                    # Check if this reduces the number of colours
                    new_set = set(new_colours.tolist())
                    if len(new_set) < len(current_colours):
                        current = new_colours
                        improved = True
                        break
                    elif len(new_set) == len(current_colours) and new_set != current_colours:
                        # Same number of colours but different set - could help future merges
                        # Only accept if it doesn't increase violations (it's valid, so no violations)
                        # But we need to be careful - only accept if it's "better" in some way
                        pass
            if improved:
                break
        
        if not improved:
            break
    
    return current.tolist()
