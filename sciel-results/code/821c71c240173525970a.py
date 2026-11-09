import time
import random
import sys
from collections import deque

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
        
    # Find largest connected component
    visited = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            comp_sizes.append((len(comp), comp))
            
    if not comp_sizes:
        return [0]
        
    comp_sizes.sort(reverse=True)
    largest_comp = comp_sizes[0][1]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    start_time = time.time()
    time_limit = 75.0  # Conservative margin
    
    best_path = []
    
    # Helper: check if a path is valid
    def is_valid_path(path):
        if not path:
            return False
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    # Helper: try to extend a path from both ends
    def extend_path(path, visited_set, max_extend=50):
        """Try to extend a path greedily from both ends"""
        if not path:
            return path
        
        path = list(path)
        visited_set = set(path)
        improved = True
        count = 0
        
        while improved and count < max_extend:
            improved = False
            count += 1
            
            # Try to extend from the start
            if path:
                start_node = path[0]
                candidates = [nei for nei in adj[start_node] if nei not in visited_set]
                if candidates:
                    # Pick the candidate with fewest unvisited neighbors (Warnsdorff)
                    best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
                    path.insert(0, best_cand)
                    visited_set.add(best_cand)
                    improved = True
            
            # Try to extend from the end
            if path:
                end_node = path[-1]
                candidates = [nei for nei in adj[end_node] if nei not in visited_set]
                if candidates:
                    best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
                    path.append(best_cand)
                    visited_set.add(best_cand)
                    improved = True
        
        return path
    
    # Helper: local search - try to improve path by 2-opt style moves
    def local_search(path, visited_set, max_iters=1000):
        """Try to improve the path using local moves"""
        if len(path) < 2:
            return path
        
        path = list(path)
        best_path = path[:]
        best_len = len(path)
        visited_set = set(path)
        rng = random.Random(42)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a vertex not in path that can be inserted
            unvisited = [v for v in comp_set if v not in visited_set]
            if not unvisited:
                break
            
            # Try inserting each unvisited vertex at each possible position
            best_insert = None
            best_pos = -1
            best_gain = 0
            
            for v in unvisited[:20]:  # Limit candidates
                # Find positions where v can be inserted
                for i in range(len(path)):
                    if i < len(path) and v in adj[path[i]]:
                        if i + 1 < len(path) and path[i+1] in adj[v]:
                            # Can insert between path[i] and path[i+1]
                            gain = 1
                            if gain > best_gain:
                                best_gain = gain
                                best_insert = v
                                best_pos = i + 1
                                break
                    elif i == 0 and v in adj[path[0]]:
                        # Can insert at beginning
                        gain = 1
                        if gain > best_gain:
                            best_gain = gain
                            best_insert = v
                            best_pos = 0
                            break
                    elif i == len(path) - 1 and v in adj[path[-1]]:
                        # Can insert at end
                        gain = 1
                        if gain > best_gain:
                            best_gain = gain
                            best_insert = v
                            best_pos = len(path)
                            break
            
            if best_insert is not None:
                path.insert(best_pos, best_insert)
                visited_set.add(best_insert)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            else:
                # No improvement, try a random swap or break
                break
        
        return best_path if len(best_path) >= len(path) else path
    
    # DFS with Warnsdorff's rule (iterative to avoid recursion depth issues)
    def dfs_iterative(start_node, comp_set, adj, time_limit, start_time):
        """Iterative DFS with Warnsdorff's rule"""
        best_path = []
        stack = [(start_node, [start_node], {start_node})]
        
        while stack:
            if time.time() - start_time > time_limit:
                break
                
            node, path, visited_set = stack.pop()
            
            if len(path) > len(best_path):
                best_path = path[:]
                
            if len(path) == len(comp_set):
                return best_path  # Found Hamiltonian path!
                
            # Get unvisited neighbors
            candidates = [nei for nei in adj[node] if nei not in visited_set]
            
            if not candidates:
                continue
                
            # Warnsdorff's rule: sort by number of unvisited neighbors
            def score(nei):
                unvisited_nei = len([x for x in adj[nei] if x not in visited_set])
                return (unvisited_nei, nei)
            
            candidates.sort(key=score)
            
            # Push neighbors in reverse order so first candidate is processed first
            for nei in reversed(candidates):
                new_visited = visited_set | {nei}
                new_path = path + [nei]
                stack.append((nei, new_path, new_visited))
        
        return best_path
    
    # Strategy 1: Try DFS from various starting nodes
    nodes_to_try = largest_comp[:]
    rng = random.Random(12345)
    rng.shuffle(nodes_to_try)
    
    # Also try nodes sorted by degree (low degree first for Warnsdorff)
    degree_sorted = sorted(largest_comp, key=lambda x: len(adj[x]))
    
    # Combine: low degree nodes first, then shuffled
    start_nodes = degree_sorted[:10] + nodes_to_try[:20]
    # Remove duplicates while preserving order
    seen = set()
    unique_starts = []
    for node in start_nodes:
        if node not in seen:
            seen.add(node)
            unique_starts.append(node)
    
    for start_node in unique_starts:
        if time.time() - start_time > time_limit:
            break
            
        path = dfs_iterative(start_node, comp_set, adj, time_limit, start_time)
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Try to extend the found path
        if best_path:
            visited_set = set(best_path)
            extended = extend_path(best_path, visited_set, max_extend=100)
            if len(extended) > len(best_path):
                best_path = extended
                visited_set = set(best_path)
            
            # Try local search
            if time.time() - start_time < time_limit - 5:
                polished = local_search(best_path, visited_set, max_iters=500)
                if len(polished) > len(best_path):
                    best_path = polished
        
        if len(best_path) == comp_size:
            break
    
    # Strategy 2: Greedy construction with random restarts
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Random start
        start_node = rng.choice(largest_comp)
        path = [start_node]
        visited_set = {start_node}
        
        while True:
            node = path[-1]
            candidates = [nei for nei in adj[node] if nei not in visited_set]
            if not candidates:
                break
            
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
            path.append(best_cand)
            visited_set.add(best_cand)
        
        if len(path) > len(best_path):
            best_path = path[:]
            visited_set = set(best_path)
        
        # Extend from both ends
        extended = extend_path(best_path, visited_set, max_extend=100)
        if len(extended) > len(best_path):
            best_path = extended
            visited_set = set(best_path)
        
        # Local search
        if time.time() - start_time < time_limit - 5:
            polished = local_search(best_path, visited_set, max_iters=500)
            if len(polished) > len(best_path):
                best_path = polished
    
    # Final polish
    if best_path:
        visited_set = set(best_path)
        if time.time() - start_time < time_limit - 2:
            polished = local_search(best_path, visited_set, max_iters=2000)
            if len(polished) > len(best_path):
                best_path = polished
    
    # Validate final path
    if not is_valid_path(best_path):
        # Fallback: return a valid path
        best_path = [largest_comp[0]]
    
    return best_path
