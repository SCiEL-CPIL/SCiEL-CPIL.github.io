import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    total_edges = n * (n - 1) // 2
    
    # Precompute edge index mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    
    # Build 1-factorization of K16
    factors = [[] for _ in range(15)]
    
    for r in range(15):
        factors[r].append((min(15, r), max(15, r)))
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            if u > v:
                u, v = v, u
            factors[r].append((u, v))
    
    # Assign colors
    colors = [0] * total_edges
    for r in range(15):
        for (u, v) in factors[r]:
            colors[edge_to_idx[(u, v)]] = r
    
    # Precompute K4 structures
    k4_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4_list.append((a, b, c, d))
    
    num_k4s = len(k4_list)
    
    # For each K4, store the 6 edge indices
    k4_edges = np.zeros((num_k4s, 6), dtype=np.int32)
    for k4_idx, (a, b, c, d) in enumerate(k4_list):
        k4_edges[k4_idx, 0] = edge_to_idx[(a, b)]
        k4_edges[k4_idx, 1] = edge_to_idx[(a, c)]
        k4_edges[k4_idx, 2] = edge_to_idx[(a, d)]
        k4_edges[k4_idx, 3] = edge_to_idx[(b, c)]
        k4_edges[k4_idx, 4] = edge_to_idx[(b, d)]
        k4_edges[k4_idx, 5] = edge_to_idx[(c, d)]
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(total_edges)]
    for k4_idx in range(num_k4s):
        for e in k4_edges[k4_idx]:
            edge_to_k4s[e].append(k4_idx)
    
    # Convert edge_to_k4s to numpy arrays for speed
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                      for lst in edge_to_k4s]
    
    # Penalty: for each K4, count how many distinct colors it has
    # We want all K4s to have >= 5 distinct colors
    # Penalty = sum over K4s of max(0, 5 - distinct_colors)
    
    def compute_penalty(color_array):
        # Get colors for all edges in all K4s
        edge_colors = color_array[k4_edges]  # (num_k4s, 6)
        penalty = 0
        for k4 in edge_colors:
            distinct = len(set(k4))
            if distinct < 5:
                penalty += (5 - distinct)
        return penalty
    
    # Local search with O(1) penalty updates
    def local_search(color_array, max_time=100.0):
        current = color_array.astype(np.int32).copy()
        num_colors = int(max(current)) + 1
        
        # Precompute initial penalty per K4
        k4_penalties = np.zeros(num_k4s, dtype=np.int32)
        for k4_idx in range(num_k4s):
            k4 = current[k4_edges[k4_idx]]
            distinct = len(set(k4))
            k4_penalties[k4_idx] = max(0, 5 - distinct)
        
        total_penalty = int(np.sum(k4_penalties))
        best = current.copy()
        best_penalty = total_penalty
        
        while time.time() < start_time + max_time:
            improved = False
            
            # Try to reduce penalty by recoloring edges
            # Focus on edges in K4s with high penalty
            if total_penalty > 0:
                # Find K4s with positive penalty
                bad_k4s = np.where(k4_penalties > 0)[0]
                
                for k4_idx in bad_k4s:
                    k4 = k4_edges[k4_idx]
                    cols = current[k4]
                    
                    # Try recoloring each edge in this K4
                    for pos in range(6):
                        edge_idx = k4[pos]
                        old_color = current[edge_idx]
                        
                        for new_color in range(num_colors):
                            if new_color == old_color:
                                continue
                            
                            # Update penalty for affected K4s
                            old_penalty = 0
                            new_penalty = 0
                            affected_k4s = edge_to_k4s_np[edge_idx]
                            
                            for ak4 in affected_k4s:
                                # Old state
                                old_k4 = current[k4_edges[ak4]]
                                old_distinct = len(set(old_k4))
                                old_penalty += max(0, 5 - old_distinct)
                                
                                # New state
                                temp = old_k4.copy()
                                temp[k4_edges[ak4] == edge_idx] = new_color
                                new_distinct = len(set(temp))
                                new_penalty += max(0, 5 - new_distinct)
                            
                            delta = new_penalty - old_penalty
                            if delta < 0:
                                # Apply change
                                current[edge_idx] = new_color
                                total_penalty += delta
                                for ak4 in affected_k4s:
                                    k4 = current[k4_edges[ak4]]
                                    k4_penalties[ak4] = max(0, 5 - len(set(k4)))
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved and total_penalty == 0:
                # Try to reduce number of colors
                for target in range(num_colors - 1, -1, -1):
                    edges_with_target = np.where(current == target)[0]
                    if len(edges_with_target) == 0:
                        num_colors -= 1
                        improved = True
                        break
                    
                    for edge_idx in edges_with_target:
                        moved = False
                        for new_color in range(num_colors - 1):
                            if new_color == target:
                                continue
                            
                            old_penalty = 0
                            new_penalty = 0
                            affected_k4s = edge_to_k4s_np[edge_idx]
                            
                            for ak4 in affected_k4s:
                                old_k4 = current[k4_edges[ak4]]
                                old_distinct = len(set(old_k4))
                                old_penalty += max(0, 5 - old_distinct)
                                
                                temp = old_k4.copy()
                                temp[k4_edges[ak4] == edge_idx] = new_color
                                new_distinct = len(set(temp))
                                new_penalty += max(0, 5 - new_distinct)
                            
                            delta = new_penalty - old_penalty
                            if delta <= 0:
                                current[edge_idx] = new_color
                                total_penalty += delta
                                for ak4 in affected_k4s:
                                    k4 = current[k4_edges[ak4]]
                                    k4_penalties[ak4] = max(0, 5 - len(set(k4)))
                                moved = True
                                break
                        if moved:
                            improved = True
                            break
                    if improved:
                        break
            
            if total_penalty < best_penalty:
                best = current.copy()
                best_penalty = total_penalty
        
        return best, best_penalty
    
    # Run local search
    if time.time() < start_time + 120.0:
        improved_colors, final_penalty = local_search(np.array(colors, dtype=np.int32), max_time=110.0)
        if final_penalty == 0:
            colors = improved_colors.tolist()
    
    # Final verification
    final_colors = np.array(colors, dtype=np.int32)
    edge_colors = final_colors[k4_edges]
    valid = True
    for k4 in edge_colors:
        if len(set(k4)) < 5:
            valid = False
            break
    
    if not valid:
        # Fallback
        colors = [0] * total_edges
        for r in range(15):
            for (u, v) in factors[r]:
                colors[edge_to_idx[(u, v)]] = r
    
    return colors
