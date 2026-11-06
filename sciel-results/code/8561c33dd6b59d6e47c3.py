import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 87  # Safety margin (90s limit, 3s buffer)
    
    P_data = {
        'n': 100,
        'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    }
    
    n = P_data['n']
    adj = [set() for _ in range(n)]
    for u, v in P_data['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find all connected components
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
            
    # Sort components by size, work on the largest
    components.sort(key=len, reverse=True)
    largest_comp = components[0]
    largest_comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Precompute degrees within the component
    comp_deg = {v: len(adj[v] & largest_comp_set) for v in largest_comp}
    
    # Identify vertices that MUST be endpoints (degree 1 in component)
    degree1_vertices = [v for v in largest_comp if comp_deg[v] == 1]
    degree2_vertices = [v for v in largest_comp if comp_deg[v] == 2]
    
    # If there are exactly 2 degree-1 vertices, they MUST be the endpoints of any Hamiltonian path
    # If there are more, we need to find a subset that works as endpoints
    
    best_path = []
    
    def validate_path(path):
        """Check if path is valid: distinct vertices, all consecutive pairs are edges."""
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    def extend_path_both_ends(path):
        """Extend path at both ends greedily."""
        path = path[:]
        visited_set = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            if path:
                start_node = path[0]
                for nei in sorted(adj[start_node], key=lambda x: comp_deg.get(x, 0)):
                    if nei not in visited_set and nei in largest_comp_set:
                        path.insert(0, nei)
                        visited_set.add(nei)
                        changed = True
                        break
            # Extend end
            if path:
                end_node = path[-1]
                for nei in sorted(adj[end_node], key=lambda x: comp_deg.get(x, 0)):
                    if nei not in visited_set and nei in largest_comp_set:
                        path.append(nei)
                        visited_set.add(nei)
                        changed = True
                        break
        return path
    
    def warnsdorff_walk(start, max_steps=None):
        """Warnsdorff's rule: prefer moves to vertices with fewest unvisited neighbors."""
        if max_steps is None:
            max_steps = comp_size
        path = [start]
        visited_set = {start}
        steps = 0
        while steps < max_steps:
            steps += 1
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited_set and v in largest_comp_set]
            if not neighbors:
                break
            # Sort by number of unvisited neighbors (Warnsdorff's rule)
            def score(v):
                return len([u for u in adj[v] if u not in visited_set and u in largest_comp_set])
            neighbors.sort(key=score)
            # With some randomness for diversity
            if len(neighbors) > 1 and random.random() < 0.3:
                # Sometimes pick from top 3
                k = min(3, len(neighbors))
                next_node = random.choice(neighbors[:k])
            else:
                next_node = neighbors[0]
            path.append(next_node)
            visited_set.add(next_node)
        return path
    
    def dfs_path_search(start, end=None, max_depth=100000):
        """
        DFS to find a path from start, optionally ending at end.
        Uses Warnsdorff ordering and backtracking.
        """
        best = [start]
        path = [start]
        visited_set = {start}
        depth = 0
        nodes_explored = 0
        
        def dfs():
            nonlocal best, depth, nodes_explored
            if time.time() > deadline - 2:
                return
            if len(path) > len(best):
                best = path[:]
            if end is not None and path[-1] == end and len(path) > len(best):
                best = path[:]
            if len(path) == comp_size:
                return  # Found Hamiltonian path
            
            depth += 1
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited_set and v in largest_comp_set]
            if not neighbors:
                depth -= 1
                return
            
            # Warnsdorff ordering
            def score(v):
                unvisited_nei = len([u for u in adj[v] if u not in visited_set and u in largest_comp_set])
                # Bonus if v is the target end
                if end is not None and v == end:
                    return -1000
                return unvisited_nei
            
            neighbors.sort(key=score)
            
            for nei in neighbors:
                nodes_explored += 1
                if time.time() > deadline - 2:
                    depth -= 1
                    return
                if nodes_explored > max_depth:
                    depth -= 1
                    return
                path.append(nei)
                visited_set.add(nei)
                dfs()
                path.pop()
                visited_set.remove(nei)
                if len(best) >= comp_size:
                    break
            depth -= 1
        
        dfs()
        return best
    
    def try_hamiltonian_with_endpoints(e1, e2, max_depth=50000):
        """Try to find a Hamiltonian path from e1 to e2."""
        if e1 == e2:
            return None
        if e2 not in adj[e1] and comp_size > 2:
            # Can't start with e1->e2 directly, but we can still try
            pass
        
        path = [e1]
        visited_set = {e1}
        best = [e1]
        nodes_explored = 0
        
        def dfs():
            nonlocal best, nodes_explored
            if time.time() > deadline - 3:
                return
            if len(path) > len(best):
                best = path[:]
            if len(path) == comp_size:
                if path[-1] == e2:
                    best = path[:]
                    return
                # Not ending at e2, but still a valid long path
                return
            if len(path) >= comp_size:
                return
            
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited_set and v in largest_comp_set]
            if not neighbors:
                return
            
            # Warnsdorff ordering with bias toward e2
            def score(v):
                unvisited_nei = len([u for u in adj[v] if u not in visited_set and u in largest_comp_set])
                if v == e2:
                    return -10000  # Strong bias toward target endpoint
                return unvisited_nei
            
            neighbors.sort(key=score)
            
            for nei in neighbors:
                nodes_explored += 1
                if time.time() > deadline - 3:
                    return
                if nodes_explored > max_depth:
                    return
                path.append(nei)
                visited_set.add(nei)
                dfs()
                path.pop()
                visited_set.remove(nei)
                if len(best) >= comp_size:
                    return
        
        dfs()
        return best
    
    def local_search_improve(path, max_iters=500):
        """
        Local search: try to insert unvisited vertices into the path.
        Also try 2-opt style rearrangements.
        """
        path = path[:]
        visited_set = set(path)
        unvisited = largest_comp_set - visited_set
        
        for _ in range(max_iters):
            if time.time() > deadline - 1:
                break
            improved = False
            
            # Try to insert a vertex between consecutive pairs
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                # Find common neighbors of u and v that are unvisited
                common = (adj[u] & adj[v]) & unvisited
                if common:
                    # Pick the one with fewest other unvisited neighbors
                    best_w = min(common, key=lambda w: len(adj[w] & unvisited))
                    new_path = path[:i+1] + [best_w] + path[i+1:]
                    if validate_path(new_path):
                        path = new_path
                        visited_set.add(best_w)
                        unvisited.discard(best_w)
                        improved = True
                        break
            
            # Try to extend endpoints
            if not improved and path:
                start_node = path[0]
                for nei in sorted(adj[start_node], key=lambda x: comp_deg.get(x, 0)):
                    if nei in unvisited:
                        path.insert(0, nei)
                        visited_set.add(nei)
                        unvisited.discard(nei)
                        improved = True
                        break
            
            if not improved and path:
                end_node = path[-1]
                for nei in sorted(adj[end_node], key=lambda x: comp_deg.get(x, 0)):
                    if nei in unvisited:
                        path.append(nei)
                        visited_set.add(nei)
                        unvisited.discard(nei)
                        improved = True
                        break
            
            # Try 2-opt: reverse a segment and reconnect
            if not improved and len(path) >= 4:
                for i in range(len(path) - 2):
                    for j in range(i + 2, len(path)):
                        # Check if reversing path[i:j+1] creates valid connections
                        a = path[i-1] if i > 0 else None
                        b = path[i]
                        c = path[j]
                        d = path[j+1] if j + 1 < len(path) else None
                        
                        # Original: ...a - b ... c - d...
                        # New: ...a - c ... b - d...
                        if a is not None and c in adj[a] and b in adj[d]:
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            if validate_path(new_path) and len(set(new_path)) == len(new_path):
                                # Check if this helps (it might allow further extension)
                                path = new_path
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return path
    
    def multi_start_search(num_starts=50):
        """Run multiple greedy walks from different starts and take the best."""
        best = []
        starts = list(largest_comp)
        random.shuffle(starts)
        
        # Prioritize degree-1 and degree-2 vertices as starts
        if degree1_vertices:
            starts = degree1_vertices + [v for v in starts if v not in set(degree1_vertices)]
        elif degree2_vertices:
            starts = degree2_vertices + [v for v in starts if v not in set(degree2_vertices)]
        
        for start in starts[:num_starts]:
            if time.time() > deadline - 5:
                break
            
            # Multiple walks from same start
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                path = warnsdorff_walk(start)
                path = extend_path_both_ends(path)
                path = local_search_improve(path, max_iters=100)
                if len(path) > len(best):
                    best = path
                if len(best) >= comp_size:
                    return best
        
        return best
    
    # Strategy 1: If we have exactly 2 degree-1 vertices, they must be endpoints
    if len(degree1_vertices) == 2:
        e1, e2 = degree1_vertices
        print(f"Two degree-1 vertices: {e1}, {e2} - these must be endpoints")
        
        # Try Hamiltonian path with these endpoints
        result = try_hamiltonian_with_endpoints(e1, e2, max_depth=100000)
        if result and len(result) > len(best_path):
            best_path = result
        
        # Also try reversed
        result = try_hamiltonian_with_endpoints(e2, e1, max_depth=50000)
        if result and len(result) > len(best_path):
            best_path = result
        
        # Multi-start search as backup
        if len(best_path) < comp_size:
            result = multi_start_search(num_starts=30)
            if len(result) > len(best_path):
                best_path = result
    
    # Strategy 2: If we have 0 degree-1 vertices, try all pairs of low-degree vertices
    elif len(degree1_vertices) == 0:
        # Try pairs of degree-2 vertices as potential endpoints
        candidates = degree2_vertices if degree2_vertices else largest_comp
        random.shuffle(candidates)
        
        tried_pairs = set()
        for i, e1 in enumerate(candidates[:15]):
            if time.time() > deadline - 10:
                break
            for e2 in candidates[:15]:
                if e1 == e2:
                    continue
                pair = (min(e1, e2), max(e1, e2))
                if pair in tried_pairs:
                    continue
                tried_pairs.add(pair)
                
                result = try_hamiltonian_with_endpoints(e1, e2, max_depth=30000)
                if result and len(result) > len(best_path):
                    best_path = result
                if len(best_path) >= comp_size:
                    break
            if len(best_path) >= comp_size:
                break
        
        # Multi-start search
        if len(best_path) < comp_size:
            result = multi_start_search(num_starts=50)
            if len(result) > len(best_path):
                best_path = result
    
    # Strategy 3: General case - multi-start with various strategies
    if len(best_path) < comp_size:
        result = multi_start_search(num_starts=100)
        if len(result) > len(best_path):
            best_path = result
    
    # Final local search polish
    if best_path and len(best_path) < comp_size:
        polished = local_search_improve(best_path, max_iters=500)
        if len(polished) > len(best_path):
            best_path = polished
    
    # Final extension
    if best_path:
        best_path = extend_path_both_ends(best_path)
    
    # If we still haven't found a good path, do a final DFS attempt
    if len(best_path) < comp_size and time.time() < deadline - 5:
        # Pick the most constrained start
        start_candidates = sorted(largest_comp, key=lambda v: comp_deg[v])
        for start in start_candidates[:5]:
            if time.time() > deadline - 3:
                break
            result = dfs_path_search(start, max_depth=50000)
            result = extend_path_both_ends(result)
            result = local_search_improve(result, max_iters=200)
            if len(result) > len(best_path):
                best_path = result
            if len(best_path) >= comp_size:
                break
    
    # Fallback: ensure we have a valid path
    if not best_path:
        best_path = [largest_comp[0]]
    
    # Final validation
    if not validate_path(best_path):
        # Try to recover
        valid = [best_path[0]]
        seen = {best_path[0]}
        for node in best_path[1:]:
            if node not in seen and node in adj[valid[-1]]:
                valid.append(node)
                seen.add(node)
        best_path = valid
    
    # Ensure at least one vertex
    if not best_path:
        best_path = [largest_comp[0]]
    
    return best_path
