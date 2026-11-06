import sys
import time

# Increase recursion limit if necessary, though iterative DFS is safer
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    best_path = []
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # For each component, find the longest path
    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        if len(comp) == 0:
            continue
            
        # If component is small, try exact search
        # Heuristic: if size < 25, exact search is fast
        if len(comp) <= 25:
            # Map component nodes to 0..k-1 for simpler bitmasks or sets
            # But using sets of original node IDs is fine for N=25
            
            def find_longest_path_in_comp(comp_nodes):
                if len(comp_nodes) == 1:
                    return comp_nodes
                best = []
                # Try each node as start
                for start in comp_nodes:
                    # DFS to find longest path starting from 'start'
                    # State: current node, visited set
                    # To optimize, we can use a recursive function that returns the longest path
                    # from 'current' given 'visited'
                    
                    def dfs(current, visited_set):
                        # Pruning: if remaining nodes cannot exceed best length, return
                        # But we don't know best length globally for this component yet in this inner loop easily.
                        # Let's just do standard DFS.
                        local_best = [current]
                        for nei in adj[current]:
                            if nei not in visited_set:
                                sub_path = dfs(nei, visited_set | {nei})
                                if len(sub_path) > len(local_best):
                                    local_best = sub_path
                        return local_best

                    path = dfs(start, {start})
                    if len(path) > len(best):
                        best = path
                return best

            path = find_longest_path_in_comp(comp)
            if len(path) > len(best_path):
                best_path = path
        else:
            # If component is large, use a heuristic (greedy with random restarts)
            # This is unlikely given the parameters, but just in case.
            # Greedy: start from each node, always go to neighbor with fewest unvisited options
            for start in comp:
                path = [start]
                seen = {start}
                while True:
                    next_node = None
                    min_opts = float('inf')
                    for nei in adj[path[-1]]:
                        if nei not in seen:
                            opts = len(adj[nei] - seen)
                            if opts < min_opts:
                                min_opts = opts
                                next_node = nei
                    if next_node is None:
                        break
                    path.append(next_node)
                    seen.add(next_node)
                if len(path) > len(best_path):
                    best_path = path

    return best_path
