import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)  # 120
    
    # Edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s and their edge indices
    all_k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in all_k4s:
        e_list = []
        for a, b in combinations(k4, 2):
            e_list.append(edge_idx[(a, b)])
        k4_edges.append(e_list)
    
    # Precompute: for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    
    # Fast color count function
    def count_colors(colors):
        return len(set(colors.tolist()))
    
    # Fast violation count using precomputed structure
    def count_violations_fast(colors):
        violations = 0
        for e_list in k4_edges:
            c = colors[e_list]
            if len(set(c.tolist())) < 5:
                violations += 1
        return violations
    
    # Initial coloring: (i+j) % 17
    colors = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)
    
    init_colors = count_colors(colors)
    init_violations = count_violations_fast(colors)
    
    best_colors = colors.copy()
    best_num_colors = init_colors
    best_violations = init_violations
    
    # Simulated annealing to reduce colors
    random.seed(42)
    np.random.seed(42)
    
    current_temp = 1.0
    cooling_rate = 0.99995
    min_temp = 0.01
    
    # Track color usage
    def get_color_counts(colors):
        counts = np.bincount(colors, minlength=20)
        return counts
    
    # Try to eliminate a specific color by reassigning its edges
    def try_eliminate_color(colors, target_color):
        """Try to reassign all edges with target_color to other colors."""
        target_edges = np.where(colors == target_color)[0]
        if len(target_edges) == 0:
            return True
        
        original = colors.copy()
        
        for e_idx in target_edges:
            i, j = edges[e_idx]
            found = False
            
            # Try existing colors first (0 to max_color), then new ones
            max_c = int(colors.max())
            for new_color in range(max_c + 1):
                if new_color == target_color:
                    continue
                
                colors[e_idx] = new_color
                
                # Check only K4s containing this edge
                valid = True
                for k4_idx in edge_to_k4s[e_idx]:
                    e_list = k4_edges[k4_idx]
                    c_vals = colors[e_list]
                    if len(set(c_vals.tolist())) < 5:
                        valid = False
                        break
                
                if valid:
                    found = True
                    break
                else:
                    colors[e_idx] = original[e_idx]
            
            if not found:
                colors[:] = original
                return False
        
        return True
    
    # Main SA loop
    iterations = 0
    max_iterations = 500000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Get color counts
        color_counts = get_color_counts(colors)
        
        # Find colors with low frequency (candidates for elimination)
        used_colors = np.where(color_counts > 0)[0]
        if len(used_colors) <= 1:
            break
        
        # Sort by frequency ascending
        freq_sorted = sorted(used_colors, key=lambda c: color_counts[c])
        
        # Try to eliminate the least frequent color(s)
        progress_made = False
        for target_color in freq_sorted[:3]:
            if color_counts[target_color] > 2:
                continue
            
            old_colors = colors.copy()
            if try_eliminate_color(colors, target_color):
                # Check if valid and improved
                new_violations = count_violations_fast(colors)
                new_colors = count_colors(colors)
                
                if new_violations == 0 and new_colors < best_num_colors:
                    best_colors = colors.copy()
                    best_num_colors = new_colors
                    best_violations = 0
                    progress_made = True
                    break
                elif new_violations == 0:
                    # Same number of colors but valid, keep trying
                    progress_made = True
                else:
                    # Revert
                    colors[:] = old_colors
        
        # If no progress, do a random perturbation to escape local minima
        if not progress_made:
            # Pick a random edge and change its color
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            max_c = int(colors.max()) + 1
            new_c = random.randint(0, max_c)
            
            if new_c != old_c:
                colors[e_idx] = new_c
                new_violations = count_violations_fast(colors)
                
                # Accept if better or with probability based on temperature
                if new_violations < best_violations:
                    best_violations = new_violations
                    if new_violations == 0:
                        nc = count_colors(colors)
                        if nc < best_num_colors:
                            best_colors = colors.copy()
                            best_num_colors = nc
                            best_violations = 0
                else:
                    # Metropolis criterion
                    delta = new_violations - best_violations
                    if delta < 0 or random.random() < np.exp(-delta / max(current_temp, 1e-10)):
                        # Accept
                        pass
                    else:
                        # Reject
                        colors[e_idx] = old_c
            
            # Cool down
            current_temp *= cooling_rate
            if current_temp < min_temp:
                current_temp = min_temp
    
    # Final check and return
    if best_violations == 0:
        result = best_colors.tolist()
        # Ensure all colors are in range 0..119
        result = [c % 120 for c in result]
        return result
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
