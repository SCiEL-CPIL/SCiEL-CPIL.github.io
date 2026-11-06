import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEADLINE = time.time() + 85.0

def get_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in ADJ[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            comps.append(comp)
    return comps

def bfs_longest_path_from(start, comp_set):
    """
    Returns (farthest_node, distance, path) from start within the component.
    """
    dist = {start: 0}
    prev = {start: None}
    queue = deque([start])
    farthest = start
    max_dist = 0
    
    while queue:
        u = queue.popleft()
        if dist[u] > max_dist:
            max_dist = dist[u]
            farthest = u
        for v in ADJ[u]:
            if v in comp_set and v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                queue.append(v)
    
    # Reconstruct path
    path = []
    cur = farthest
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return farthest, max_dist, path

def find_diameter(comp):
    """
    Find the longest path in a tree component.
    """
    if not comp:
        return []
    if len(comp) == 1:
        return comp
    
    comp_set = set(comp)
    
    # Step 1: BFS from any node to find farthest node u
    u, _, _ = bfs_longest_path_from(comp[0], comp_set)
    
    # Step 2: BFS from u to find farthest node v
    v, dist, path = bfs_longest_path_from(u, comp_set)
    
    return path

def find_longest_path_in_cyclic_comp(comp):
    """
    Find longest path in a small component with cycles using DFS.
    """
    if not comp:
        return []
    
    best_path = []
    comp_set = set(comp)
    
    def dfs(node, visited, path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        
        for nei in ADJ[node]:
            if nei in comp_set and nei not in visited:
                visited.add(nei)
                path.append(nei)
                dfs(nei, visited, path)
                path.pop()
                visited.remove(nei)
        
        # Time check
        if time.time() > DEADLINE:
            raise TimeoutError()

    # Try starting from each node in the component (small enough)
    for start in comp:
        if time.time() > DEADLINE:
            break
        dfs(start, {start}, [start])
        
    return best_path

def construct():
    t0 = time.time()
    comps = get_components()
    
    best_path = []
    
    for comp in comps:
        if time.time() > DEADLINE:
            break
            
        n_comp = len(comp)
        e_comp = sum(len(ADJ[u]) for u in comp) // 2
        
        # Heuristic: if it's a tree (e = n - 1), use diameter algorithm
        # If it has cycles (e >= n), use DFS (only if small)
        is_tree = (e_comp == n_comp - 1)
        
        if is_tree:
            path = find_diameter(comp)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Component has cycles
            if n_comp < 30:
                try:
                    path = find_longest_path_in_cyclic_comp(comp)
                    if len(path) > len(best_path):
                        best_path = path
                except TimeoutError:
                    pass
            else:
                # Large cyclic component? Unlikely, but handle gracefully
                # Fall back to diameter-like heuristic (BFS from arbitrary node)
                # Or simple greedy
                comp_set = set(comp)
                u, _, _ = bfs_longest_path_from(comp[0], comp_set)
                v, _, path = bfs_longest_path_from(u, comp_set)
                if len(path) > len(best_path):
                    best_path = path

    # Fallback if no path found (should not happen)
    if not best_path:
        best_path = [0]
        
    return best_path
