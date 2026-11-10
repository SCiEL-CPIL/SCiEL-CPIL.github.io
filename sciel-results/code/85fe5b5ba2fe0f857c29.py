"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before the 150s limit
    
    n = 16
    # Build edge index mapping
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [edge_to_idx[(a,b)], edge_to_idx[(a,c)], edge_to_idx[(a,d)],
                      edge_to_idx[(b,c)], edge_to_idx[(b,d)], edge_to_idx[(c,d)]]
        k4s.append(e_indices)
    num_k4s = len(k4s)  # 1820
    
    # For each edge, track which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4s):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Initial coloring: (i+j) % 15
    coloring = np.array([(i + j) % 15 for i, j in edges], dtype=np.int32)
    
    def count_colors_used(coloring):
        return len(set(coloring.tolist()))
    
    def validate(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        for e_indices in k4s:
            if len(set(coloring[e_indices].tolist())) < 5:
                return False
        return True
    
    best_coloring = coloring.copy()
    best_num_colors = count_colors_used(best_coloring)
    
    # Check if initial coloring is valid
    if not validate(best_coloring):
        # Try (i+j) % 17
        coloring2 = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)
        if validate(coloring2):
            best_coloring = coloring2.copy()
            best_num_colors = count_colors_used(best_coloring)
        else:
            # Try a random coloring
            np.random.seed(42)
            for attempt in range(100):
                coloring3 = np.random.randint(0, 20, size=num_edges, dtype=np.int32)
                if validate(coloring3):
                    best_coloring = coloring3.copy()
                    best_num_colors = count_colors_used(best_coloring)
                    break
    
    # Try to reduce number of colors
    # Strategy: for each color from highest to lowest, try to reassign its edges
    # to lower-numbered colors while maintaining validity
    
    def can_reassign_edge(edge_idx, new_color, coloring, target_color):
        """Check if reassigning edge_idx to new_color maintains validity for all K4s containing it."""
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4s[k4_idx]
            # Count distinct colors in this K4 with the proposed change
            colors_set = set()
            for ei in e_indices:
                if ei == edge_idx:
                    colors_set.add(new_color)
                else:
                    colors_set.add(coloring[ei])
            if len(colors_set) < 5:
                return False
        return True
    
    def try_eliminate_color(coloring, target_color):
        """Try to eliminate target_color by reassigning all its edges to other colors."""
        edges_with_color = [i for i in range(num_edges) if coloring[i] == target_color]
        if not edges_with_color:
            return True, coloring
        
        new_coloring = coloring.copy()
        
        # Sort edges by how constrained they are (more K4s = more constrained)
        edges_with_color.sort(key=lambda e: len(edge_to_k4s[e]), reverse=True)
        
        for e_idx in edges_with_color:
            # Try to find a new color for this edge
            found = False
            # Try colors 0 to target_color-1 (lower colors)
            for new_c in range(target_color):
                if can_reassign_edge(e_idx, new_c, new_coloring, target_color):
                    new_coloring[e_idx] = new_c
                    found = True
                    break
            
            if not found:
                # Try colors target_color+1 to max_color
                max_c = max(new_coloring)
                for new_c in range(target_color + 1, max_c + 1):
                    if can_reassign_edge(e_idx, new_c, new_coloring, target_color):
                        new_coloring[e_idx] = new_c
                        found = True
                        break
            
            if not found:
                # Failed to reassign this edge
                return False, coloring
        
        return True, new_coloring
    
    # Iteratively try to eliminate colors from highest to lowest
    current_coloring = best_coloring.copy()
    current_num_colors = best_num_colors
    
    while time.time() < deadline - 5:
        # Find the highest color used
        max_color = current_num_colors - 1
        if max_color < 0:
            break
        
        success, new_coloring = try_eliminate_color(current_coloring, max_color)
        if success:
            new_num_colors = count_colors_used(new_coloring)
            if new_num_colors < best_num_colors:
                best_num_colors = new_num_colors
                best_coloring = new_coloring.copy()
                current_coloring = new_coloring.copy()
                current_num_colors = new_num_colors
            else:
                # No improvement, break
                break
        else:
            # Try a different approach: local search with random restarts
            break
    
    # If we're still above 13 colors, try a more aggressive local search
    if current_num_colors > 13 and time.time() < deadline - 10:
        # Try random perturbations and local optimization
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start from a random coloring with current_num_colors or fewer
            np.random.seed(1000 + restart)
            max_c = min(current_num_colors, 15)
            trial = np.random.randint(0, max_c, size=num_edges, dtype=np.int32)
            
            if not validate(trial):
                continue
            
            trial_colors = count_colors_used(trial)
            if trial_colors >= current_num_colors:
                continue
            
            # Try to reduce further
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                max_c = count_colors_used(trial) - 1
                if max_c < 0:
                    break
                success, new_trial = try_eliminate_color(trial, max_c)
                if success:
                    new_colors = count_colors_used(new_trial)
                    if new_colors < trial_colors:
                        trial = new_trial.copy()
                        trial_colors = new_colors
                    else:
                        break
                else:
                    break
            
            if trial_colors < best_num_colors:
                best_num_colors = trial_colors
                best_coloring = trial.copy()
                current_coloring = trial.copy()
                current_num_colors = trial_colors
    
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
