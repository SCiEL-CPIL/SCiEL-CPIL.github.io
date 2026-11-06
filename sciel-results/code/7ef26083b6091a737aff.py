"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # 55s safety margin
    rng = random.Random(748)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to list for faster iteration
    adj_list = [list(s) for s in adj]
    
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
                for nb in adj_list[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size descending (search largest first)
    components.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Greedy initial solution: extend path greedily from each vertex
    def greedy_extend(start, adj_list, n, rng=None):
        path = [start]
        visited = {start}
        while True:
            last = path[-1]
            candidates = [nb for nb in adj_list[last] if nb not in visited]
            if not candidates:
                break
            if rng is not None and rng.random() < 0.3:
                # Random choice with some bias toward low-degree
                weights = [1.0 / (1 + len([y for y in adj_list[x] if y not in visited])) for x in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break
            else:
                # Deterministic: pick candidate with fewest unvisited neighbors
                candidates.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited]))
                chosen = candidates[0]
            path.append(chosen)
            visited.add(chosen)
        return path
    
    # Get initial greedy solutions
    for start in range(n):
        gpath = greedy_extend(start, adj_list, n)
        if len(gpath) > best_len:
            best_path = gpath[:]
            best_len = len(gpath)
    
    # Iterative deepening DFS with randomized restarts
    def dfs_iterative_deepening(start, comp, comp_size, max_depth, deadline, rng):
        """DFS with iterative deepening and time budget."""
        nonlocal best_path, best_len
        
        # Use a stack-based approach for efficiency
        # State: (path, visited_set, next_index_in_candidates)
        
        # Precompute: for each vertex in comp, its neighbors within comp
        comp_set = set(comp)
        
        def dfs(path, visited):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return False
            
            cur_len = len(path)
            if cur_len > best_len:
                best_path = path[:]
                best_len = cur_len
            
            if cur_len == max_depth:
                return True
            
            last = path[-1]
            # Get unvisited neighbors in component
            unvisited_neighbors = [nb for nb in adj_list[last] if nb not in visited and nb in comp_set]
            if not unvisited_neighbors:
                return False
            
            # Sort by number of unvisited neighbors (ascending) - prefer dead ends
            unvisited_neighbors.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited and y in comp_set]))
            
            for neighbor in unvisited_neighbors:
                path.append(neighbor)
                visited.add(neighbor)
                if dfs(path, visited):
                    path.pop()
                    visited.remove(neighbor)
                    return True
                path.pop()
                visited.remove(neighbor)
                if time.time() > deadline:
                    return False
            return False
        
        return dfs([start], {start})
    
    # Try each component, starting with largest
    for comp in components:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= best_len:
            break
        
        # Try vertices in this component as start, ordered by degree ascending (likely endpoints)
        comp_vertices = sorted(comp, key=lambda x: len(adj_list[x]))
        
        # For each start, try increasing path lengths
        for start in comp_vertices:
            if time.time() > deadline:
                break
            
            # Try to find paths of increasing length
            # Start from best_len+1 and go up to comp_size
            target_len = best_len + 1
            while target_len <= comp_size and time.time() < deadline:
                found = dfs_iterative_deepening(start, comp, comp_size, target_len, deadline, rng)
                if found:
                    target_len += 1
                else:
                    # If we couldn't find a path of this length, try a slightly different approach
                    # with randomized greedy to get a new starting point
                    break
        
        # Randomized restarts: try random starting points and greedy extensions
        restarts = 0
        while time.time() < deadline and restarts < 50:
            start = rng.choice(comp)
            gpath = greedy_extend(start, adj_list, n, rng)
            if len(gpath) > best_len:
                best_path = gpath[:]
                best_len = len(gpath)
            restarts += 1
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
