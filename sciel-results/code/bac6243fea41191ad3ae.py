import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Fast backtracking with unit propagation for 2-colorability of 5-uniform hypergraph.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    m = len(edges)
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    colors = [-1] * n
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                edge_counts[e_idx][2] -= 1
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if cU == 1:
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
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
            return True
            
        v = best_v
        colors_choice = [0, 1]
        random.shuffle(colors_choice)
        
        for c in colors_choice:
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
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
    best_edge_count = 200  # Maximum allowed
    
    # Strategy: 
    # 1. Try to find a non-2-colorable hypergraph with n=11 and m around 51 (known upper bound)
    # 2. Use local search to remove edges while maintaining non-2-colorability
    # 3. If that fails, fall back to n=9 with all edges (126)
    
    # Pre-compute all edges for n=11
    all_edges_11 = list(combinations(range(11), 5))
    random.shuffle(all_edges_11)
    
    # Try multiple random subsets of size 51-60
    for m in [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
        if time.time() > deadline - 10.0:
            break
            
        # Try multiple random subsets
        for trial in range(20):
            if time.time() > deadline - 10.0:
                break
                
            # Random subset
            random.shuffle(all_edges_11)
            current_edges = all_edges_11[:m]
            edge_set = set(map(tuple, current_edges))
            
            # Check if non-2-colorable
            if not is_2_colorable(11, [list(e) for e in current_edges]):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': 11, 'edges': [list(e) for e in current_edges]}
                
                # Local search: try to remove edges while maintaining non-2-colorability
                improved = True
                while improved and time.time() < deadline - 10.0:
                    improved = False
                    # Try removing each edge
                    for i in range(len(current_edges)):
                        if time.time() > deadline - 10.0:
                            break
                            
                        # Remove edge i
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        
                        # Check if still non-2-colorable
                        if not is_2_colorable(11, [list(e) for e in test_edges]):
                            # Accept the removal
                            current_edges = test_edges
                            if len(current_edges) < best_edge_count:
                                best_edge_count = len(current_edges)
                                best_result = {'n': 11, 'edges': [list(e) for e in current_edges]}
                            improved = True
                            break  # Restart from beginning after removal
    
    # If we found a good result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: n=9, all edges (126). This is guaranteed to be non-2-colorable.
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
