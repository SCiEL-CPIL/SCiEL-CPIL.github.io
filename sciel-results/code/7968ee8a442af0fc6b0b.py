import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    if not components:
        return [0]
        
    target_comp = components[0]
    target_set = set(target_comp)
    comp_size = len(target_comp)
    
    # Build adjacency list for the component
    comp_adj = {v: set() for v in target_comp}
    for u, v in P['edges']:
        if u in target_set and v in target_set:
            comp_adj[u].add(v)
            comp_adj[v].add(u)
    
    # Get degrees in the component
    deg = {v: len(comp_adj[v]) for v in target_comp}
    
    deadline = time.time() + 55.0  # Safety margin
    
    best_len = 0
    best_path_found = []
    
    # Strategy: Try multiple starting points, prioritized by degree
    # Low-degree vertices are often endpoints of long paths
    starts = sorted(target_comp, key=lambda x: (deg[x], x))
    
    # Also try high-degree vertices as they may be central
    high_degree_starts = sorted(target_comp, key=lambda x: (-deg[x], x))
    
    # Combine: first low-degree, then high-degree
    all_starts = starts[:min(20, len(starts))] + high_degree_starts[:min(10, len(high_degree_starts))]
    # Remove duplicates while preserving order
    seen = set()
    unique_starts = []
    for s in all_starts:
        if s not in seen:
            seen.add(s)
            unique_starts.append(s)
    
    for start in unique_starts:
        if time.time() > deadline:
            break
            
        # DFS with Warnsdorff's rule and backtracking
        visited = {start}
        path = [start]
        
        # Stack-based DFS: (current_node, visited_frozenset, path_list, neighbor_index)
        # Use a list of states for backtracking
        stack = [(start, frozenset(visited), path, 0)]
        
        while stack:
            if time.time() > deadline:
                break
                
            node, visited_frozen, path_list, next_idx = stack[-1]
            
            # Update best if we found a longer path
            if len(path_list) > best_len:
                best_len = len(path_list)
                best_path_found = path_list[:]
                
            # Check if we've found a Hamiltonian path
            if len(visited_frozen) == comp_size:
                return best_path_found
            
            # Get unvisited neighbors in the component
            unvisited_neighbors = [nb for nb in comp_adj[node] if nb not in visited_frozen]
            
            if not unvisited_neighbors:
                # Dead end, backtrack
                stack.pop()
                continue
            
            # Warnsdorff's Rule: sort by number of unvisited neighbors (ascending)
            unvisited_neighbors.sort(key=lambda x: len([y for y in comp_adj[x] if y not in visited_frozen]))
            
            if next_idx < len(unvisited_neighbors):
                nb = unvisited_neighbors[next_idx]
                # Update stack top with incremented index
                stack[-1] = (node, visited_frozen, path_list, next_idx + 1)
                
                # Move to neighbor
                new_visited = visited_frozen | {nb}
                new_path = path_list + [nb]
                stack.append((nb, new_visited, new_path, 0))
            else:
                # No more neighbors, backtrack
                stack.pop()
        
        # If we found a Hamiltonian path, we're done
        if best_len == comp_size:
            break
    
    # If DFS didn't find a good path, try a greedy approach as fallback
    if best_len < 10:
        for start in unique_starts[:5]:
            if time.time() > deadline:
                break
            visited = {start}
            path = [start]
            current = start
            while True:
                if time.time() > deadline:
                    break
                neighbors = [nb for nb in comp_adj[current] if nb not in visited]
                if not neighbors:
                    break
                neighbors.sort(key=lambda x: len(comp_adj[x]))
                current = neighbors[0]
                path.append(current)
                visited.add(current)
            if len(path) > best_len:
                best_len = len(path)
                best_path_found = path[:]
    
    if best_path_found:
        return best_path_found
    
    # Fallback: return a single vertex
    return [target_comp[0]]

# EVOLVE-BLOCK-END
