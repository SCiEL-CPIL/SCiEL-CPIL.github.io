"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles in the complete graph K20
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d with edges (a,b),(b,c),(c,d),(d,a)
    # We enumerate all unique 4-cycles (up to rotation and reflection)
    
    # For each set of 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # But we only need to consider cycles where all 4 edges exist in our graph
    
    # Precompute all 4-cycles as tuples of 4 edges (each edge is a frozenset of 2 vertices)
    all_4cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Three distinct 4-cycles on vertices {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    all_4cycles.append(((a,b), (b,c), (c,d), (d,a)))
                    # Cycle 2: a-b-d-c-a
                    all_4cycles.append(((a,b), (b,d), (d,c), (c,a)))
                    # Cycle 3: a-c-b-d-a
                    all_4cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    # Convert edges to canonical form (u,v) with u<v
    def canon_edge(e):
        return (min(e[0], e[1]), max(e[0], e[1]))
    
    # Normalize all 4-cycles
    normalized_cycles = []
    seen = set()
    for cyc in all_4cycles:
        norm = tuple(sorted(tuple(sorted(canon_edge(e))) for e in cyc))
        if norm not in seen:
            seen.add(norm)
            normalized_cycles.append(norm)
    
    # Build edge -> cycles mapping for fast lookup
    edge_to_cycles = defaultdict(list)
    for idx, cyc in enumerate(normalized_cycles):
        for e in cyc:
            edge_to_cycles[e].append(idx)
    
    num_cycles = len(normalized_cycles)
    
    # Convert to numpy arrays for vectorized operations
    # Each cycle has 4 edges, each edge is (u, v)
    cycle_edges_u = np.zeros((num_cycles, 4), dtype=np.int32)
    cycle_edges_v = np.zeros((num_cycles, 4), dtype=np.int32)
    for idx, cyc in enumerate(normalized_cycles):
        for j, (u, v) in enumerate(cyc):
            cycle_edges_u[idx, j] = u
            cycle_edges_v[idx, j] = v
    
    def count_bad_cycles_fast(color_mat, edge_present):
        """
        Count non-rainbow 4-cycles using vectorized numpy.
        color_mat: 20x20 matrix, color_mat[u,v] = color or -1 if no edge
        edge_present: 20x20 boolean matrix
        """
        # Get colors for all edges in all cycles
        # For each cycle idx and edge position j, get color_mat[u, v]
        u_idx = cycle_edges_u.ravel()
        v_idx = cycle_edges_v.ravel()
        colors_all = color_mat[u_idx, v_idx]  # shape (num_cycles * 4,)
        colors_all = colors_all.reshape(num_cycles, 4)
        
        # Check if all 4 edges are present
        present_all = edge_present[u_idx, v_idx].reshape(num_cycles, 4)
        all_present = present_all.all(axis=1)
        
        # Check if all 4 colors are distinct
        c0, c1, c2, c3 = colors_all[:, 0], colors_all[:, 1], colors_all[:, 2], colors_all[:, 3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        # Bad cycles: present but not distinct
        bad_mask = all_present & ~distinct
        return np.sum(bad_mask), bad_mask
    
    def get_edge_count(color_mat):
        """Count number of edges (non -1 entries) in color_mat."""
        # Only count upper triangle to avoid double counting
        return np.sum(np.triu(color_mat, k=1) != -1)
    
    def try_add_edge(color_mat, edge_present, u, v, color):
        """Try to add edge (u,v) with given color. Returns (success, new_bad_count)."""
        if u == v:
            return False, 0
        u, v = min(u, v), max(u, v)
        if color_mat[u, v] != -1:
            return False, 0
        
        old_color = color_mat[u, v]
        old_present = edge_present[u, v]
        
        color_mat[u, v] = color
        color_mat[v, u] = color
        edge_present[u, v] = True
        edge_present[v, u] = True
        
        new_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
        
        # Revert if not valid
        if new_bad > 0:
            color_mat[u, v] = old_color
            color_mat[v, u] = old_color
            edge_present[u, v] = old_present
            edge_present[v, u] = old_present
            return False, new_bad
        
        return True, 0
    
    def try_remove_edge(color_mat, edge_present, u, v):
        """Remove edge (u,v). Returns success."""
        if u == v:
            return False
        u, v = min(u, v), max(u, v)
        if color_mat[u, v] == -1:
            return False
        
        color_mat[u, v] = -1
        color_mat[v, u] = -1
        edge_present[u, v] = False
        edge_present[v, u] = False
        return True
    
    # Phase 1: Greedy construction with multiple random restarts
    best_color_mat = None
    best_edge_count = 0
    best_bad_count = float('inf')
    
    random.seed(42)
    np.random.seed(42)
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with no edges
        color_mat = np.full((n, n), -1, dtype=np.int32)
        edge_present = np.zeros((n, n), dtype=bool)
        
        # Create list of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        current_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 10.0:
                break
            
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for color in colors:
                success, _ = try_add_edge(color_mat, edge_present, u, v, color)
                if success:
                    added = True
                    break
            
            if not added:
                pass
        
        edge_count = get_edge_count(color_mat)
        bad_count, _ = count_bad_cycles_fast(color_mat, edge_present)
        
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_color_mat = color_mat.copy()
            best_bad_count = 0
    
    # Phase 2: Local search on the best solution
    if best_color_mat is not None:
        color_mat = best_color_mat.copy()
        edge_present = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(n):
                if color_mat[u, v] != -1:
                    edge_present[u, v] = True
        
        current_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
        current_edges = get_edge_count(color_mat)
        
        # Try to add more edges
        missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if color_mat[u, v] == -1]
        random.shuffle(missing_edges)
        
        for (u, v) in missing_edges:
            if time.time() > deadline - 5.0:
                break
            
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for color in colors:
                success, _ = try_add_edge(color_mat, edge_present, u, v, color)
                if success:
                    current_edges += 1
                    break
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_color_mat = color_mat.copy()
    
    # Phase 3: Simulated annealing-style local search
    if best_color_mat is not None:
        color_mat = best_color_mat.copy()
        edge_present = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(n):
                if color_mat[u, v] != -1:
                    edge_present[u, v] = True
        
        current_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
        current_edges = get_edge_count(color_mat)
        
        # Local search: try to flip colors or add/remove edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        for step in range(20000):
            if time.time() > deadline - 3.0:
                break
            
            # Random action: add edge, remove edge, or change color
            action = random.randint(0, 2)
            
            if action == 0:  # Add edge
                u, v = random.choice(all_edges)
                if color_mat[u, v] == -1:
                    color = random.randint(0, num_colors - 1)
                    success, _ = try_add_edge(color_mat, edge_present, u, v, color)
                    if success:
                        current_edges += 1
                        if current_edges > best_edge_count:
                            best_edge_count = current_edges
                            best_color_mat = color_mat.copy()
            
            elif action == 1:  # Remove edge
                present_edges = [(u, v) for u, v in all_edges if color_mat[u, v] != -1]
                if present_edges:
                    u, v = random.choice(present_edges)
                    try_remove_edge(color_mat, edge_present, u, v)
                    current_edges -= 1
            
            else:  # Change color
                present_edges = [(u, v) for u, v in all_edges if color_mat[u, v] != -1]
                if present_edges:
                    u, v = random.choice(present_edges)
                    old_color = color_mat[u, v]
                    new_color = random.randint(0, num_colors - 1)
                    if new_color != old_color:
                        # Check if changing color creates bad cycles
                        color_mat[u, v] = new_color
                        color_mat[v, u] = new_color
                        new_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
                        if new_bad > 0:
                            color_mat[u, v] = old_color
                            color_mat[v, u] = old_color
        
        final_bad, _ = count_bad_cycles_fast(color_mat, edge_present)
        if final_bad == 0:
            final_edges = get_edge_count(color_mat)
            if final_edges > best_edge_count:
                best_edge_count = final_edges
                best_color_mat = color_mat.copy()
    
    # If we still have no valid solution, return a star (fallback)
    if best_color_mat is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    # Convert color_mat to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_color_mat[u, v] != -1:
                c = int(best_color_mat[u, v])
                result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
