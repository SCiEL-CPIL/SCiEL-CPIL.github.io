import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = [0]
    best_len = 1
    
    # Helper: time check
    def time_ok():
        return time.time() < deadline
    
    # Strategy 1: Warnsdorff greedy from all nodes
    def greedy_warnsdorff(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: (degrees[x], x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_warnsdorff(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 2: Max-degree greedy from all nodes
    def greedy_max_degree(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            next_node = max(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_max_degree(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 3: Forced-move greedy with random tie-breaking
    rng = random.Random(748)
    
    def greedy_forced_random(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            if not time_ok():
                break
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            forced = [x for x in neighbors if len(adj[x] - visited) == 1]
            if forced:
                next_node = forced[0]
            else:
                next_node = rng.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
        return path
    
    for i in range(n):
        if not time_ok():
            break
        path = greedy_forced_random(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # More random restarts of forced greedy
    for _ in range(200):
        if not time_ok():
            break
        start_node = rng.randint(0, n - 1)
        path = greedy_forced_random(start_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 4: IDDFS with dead-end pruning
    # Key insight: if moving to neighbor x would make x a dead-end
    # (all neighbors of x except current are already visited), skip x
    # unless it's the only option.
    
    def is_dead_end_after_move(current, neighbor, visited):
        """Check if after moving from current to neighbor, neighbor becomes a dead-end."""
        # After moving to neighbor, the visited set includes neighbor
        # neighbor's remaining unvisited neighbors (excluding current which is now in visited)
        remaining = [x for x in adj[neighbor] if x not in visited and x != current]
        return len(remaining) == 0
    
    def find_path_iddfs_pruned(target_len, start_node):
        """Try to find a simple path of exactly target_len vertices starting from start_node,
        with dead-end pruning."""
        if not time_ok():
            return None
        
        # Stack entries: (current_node, path_tuple, visited_set)
        stack = [(start_node, (start_node,), frozenset([start_node]))]
        
        while stack:
            if not time_ok():
                return None
            
            current, path, visited = stack.pop()
            
            if len(path) == target_len:
                return list(path)
            
            neighbors = [x for x in adj[current] if x not in visited]
            
            # Apply dead-end pruning: prefer neighbors that don't create dead ends
            # Sort: non-dead-ends first, then by degree (Warnsdorff)
            def neighbor_sort_key(nb):
                de = 1 if is_dead_end_after_move(current, nb, visited) else 0
                return (de, degrees[nb], nb)
            
            neighbors.sort(key=neighbor_sort_key)
            
            for nb in reversed(neighbors):  # reverse so lowest-degree/non-dead-end is on top of stack
                if not time_ok():
                    return None
                new_path = path + (nb,)
                new_visited = visited | {nb}
                stack.append((nb, new_path, new_visited))
        
        return None
    
    # Try to improve by finding paths of length best_len + 1, best_len + 2, etc.
    for target_len in range(best_len + 1, n + 1):
        if not time_ok():
            break
        
        found = False
        for start_node in range(n):
            if not time_ok():
                break
            
            path = find_path_iddfs_pruned(target_len, start_node)
            if path is not None:
                best_path = path
                best_len = target_len
                found = True
                break
        
        if not found:
            # If we can't find a path of this length, stop trying longer ones
            break
    
    # Strategy 5: Bidirectional extension - try to extend best_path from both ends
    def try_extend_end(path, end, max_depth):
        """Try to extend path at the given end (0=start, 1=end) using DFS with pruning."""
        if not time_ok():
            return path
        
        visited = set(path)
        current = path[0] if end == 0 else path[-1]
        extend_path = []
        
        # DFS to extend
        stack = [(current, extend_path[:])]
        best_extension = []
        
        while stack:
            if not time_ok():
                break
            
            node, ext = stack.pop()
            
            if len(ext) > len(best_extension):
                best_extension = ext[:]
            
            if len(ext) >= max_depth:
                continue
            
            neighbors = [x for x in adj[node] if x not in visited]
            
            # Sort by Warnsdorff
            neighbors.sort(key=lambda x: (degrees[x], x))
            
            for nb in reversed(neighbors):
                new_visited = visited | {nb}
                stack.append((nb, ext + [nb]))
        
        if best_extension:
            if end == 0:
                return best_extension[::-1] + path
            else:
                return path + best_extension
        
        return path
    
    # Try extending from both ends
    if best_len < n:
        max_ext = min(20, n - best_len)
        extended1 = try_extend_end(best_path, 0, max_ext)
        if len(extended1) > best_len:
            best_path = extended1
            best_len = len(best_path)
        
        if best_len < n:
            max_ext = min(20, n - best_len)
            extended2 = try_extend_end(best_path, 1, max_ext)
            if len(extended2) > best_len:
                best_path = extended2
                best_len = len(best_path)
        
        # Try again from both ends
        if best_len < n:
            max_ext = min(20, n - best_len)
            extended1 = try_extend_end(best_path, 0, max_ext)
            if len(extended1) > best_len:
                best_path = extended1
                best_len = len(best_path)
            
            if best_len < n:
                max_ext = min(20, n - best_len)
                extended2 = try_extend_end(best_path, 1, max_ext)
                if len(extended2) > best_len:
                    best_path = extended2
                    best_len = len(best_path)
    
    return best_path

# EVOLVE-BLOCK-END
