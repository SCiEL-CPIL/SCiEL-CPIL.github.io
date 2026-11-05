import time
import random
import numpy as np
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Fast check using bitmasks and recursive backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute edge masks
    edge_masks = np.zeros(m, dtype=np.int64)
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
    
    # Vertex to edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # State: color of each vertex (-1 = unassigned, 0, 1)
    colors = np.full(n, -1, dtype=np.int8)
    # Edge counts: [count_color0, count_color1, count_unknown]
    edge_counts = np.zeros((m, 3), dtype=np.int8)
    edge_counts[:, 2] = 5
    
    def propagate(queue):
        """
        Process queue of (vertex, color) assignments.
        Returns False if contradiction, True otherwise.
        """
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            # Assign v to c
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                # Update counts
                edge_counts[e_idx, 2] -= 1
                edge_counts[e_idx, c] += 1
                
                c0 = edge_counts[e_idx, 0]
                c1 = edge_counts[e_idx, 1]
                cU = edge_counts[e_idx, 2]
                
                # Check contradiction: monochromatic
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation
                if cU == 1:
                    # One unknown left
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        # Must be 1
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        # Must be 0
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Heuristic: Pick unassigned vertex with highest "pressure"
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx, 0]
                c1 = edge_counts[e_idx, 1]
                cU = edge_counts[e_idx, 2]
                if cU == 1:
                    if c0 == 4 or c1 == 4:
                        score += 10
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True  # All assigned
            
        v = best_v
        
        # Try 0 then 1
        for c in [0, 1]:
            old_colors = colors.copy()
            old_edge_counts = edge_counts.copy()
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_edge_count = 200
    
    # Try to find a 51-edge solution
    # The known construction has 51 edges
    
    # First, try a known small non-2-colorable hypergraph
    # Fano plane variant or other known constructions
    
    # Let's try systematic search with local improvement
    for n in [14, 15, 16, 17, 18]:
        if time.time() > deadline - 15.0:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_all_edges = len(all_edges)
        
        if num_all_edges < 51:
            continue
        
        # Try multiple random initial configurations
        for trial in range(20):
            if time.time() > deadline - 15.0:
                break
                
            # Start with a random subset of size ~60-70
            random.shuffle(all_edges)
            current_edges = set(all_edges[:65])
            
            # Ensure non-2-colorable by adding edges if needed
            max_adds = 30
            adds = 0
            while is_2_colorable(n, list(current_edges)) and adds < max_adds:
                if time.time() > deadline - 15.0:
                    break
                # Add a random edge not in current set
                candidate = random.choice(all_edges)
                if candidate not in current_edges:
                    current_edges.add(candidate)
                    adds += 1
                else:
                    break
            
            if is_2_colorable(n, list(current_edges)):
                continue  # Failed to make non-2-colorable
            
            current_edges_list = list(current_edges)
            
            # Local Search: try to remove edges while maintaining non-2-colorability
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                # Try removing each edge
                for i, edge in enumerate(current_edges_list):
                    if time.time() > deadline - 10.0:
                        break
                    
                    new_edges = current_edges_list[:i] + current_edges_list[i+1:]
                    
                    if not is_2_colorable(n, new_edges):
                        # Accept removal
                        current_edges_list = new_edges
                        improved = True
                        if len(current_edges_list) < best_edge_count:
                            best_edge_count = len(current_edges_list)
                            best_result = {'n': n, 'edges': [list(e) for e in current_edges_list]}
                        break  # Restart loop with new list
            
            # If we got to 51 or below, we can stop
            if best_edge_count <= 51:
                return best_result
            
            # Try swap moves to escape local minima
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                
                if len(current_edges_list) <= 30:
                    break
                
                idx = random.randint(0, len(current_edges_list) - 1)
                edge_to_remove = current_edges_list[idx]
                edge_to_add = random.choice(all_edges)
                
                if edge_to_add == edge_to_remove:
                    continue
                
                new_edges = [e for e in current_edges_list if e != edge_to_remove]
                new_edges.append(edge_to_add)
                
                if not is_2_colorable(n, new_edges):
                    current_edges_list = new_edges
                    # Try to remove edges again
                    improved = True
                    while improved and time.time() < deadline - 10.0:
                        improved = False
                        for i, edge in enumerate(current_edges_list):
                            if time.time() > deadline - 10.0:
                                break
                            new_edges2 = current_edges_list[:i] + current_edges_list[i+1:]
                            if not is_2_colorable(n, new_edges2):
                                current_edges_list = new_edges2
                                improved = True
                                if len(current_edges_list) < best_edge_count:
                                    best_edge_count = len(current_edges_list)
                                    best_result = {'n': n, 'edges': [list(e) for e in current_edges_list]}
                                break
    
    # Fallback: return a known non-2-colorable hypergraph
    # Complete 5-uniform on 9 vertices is definitely non-2-colorable
    if best_result is None:
        n_fb = 9
        all_edges_fb = list(combinations(range(n_fb), 5))
        best_result = {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
        best_edge_count = len(all_edges_fb)
    
    return best_result
