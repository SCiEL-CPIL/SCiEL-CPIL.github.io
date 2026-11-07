"""Rewritten program: maximum clique subdivision via branch-and-bound + disjoint path packing."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def time_up():
        return time.time() >= deadline
    
    # For a candidate branch set B, try to find vertex-disjoint paths for all pairs
    def try_pack_paths(branch_set):
        """
        Try to find internally vertex-disjoint paths connecting each pair in branch_set.
        Returns dict mapping (u,v) -> path, or None if impossible.
        """
        if time_up():
            return None
        
        branches = list(branch_set)
        k = len(branches)
        if k < 2:
            return None
        
        # Pairs to connect
        pairs = list(combinations(branches, 2))
        
        # Non-branch vertices available for path interiors
        non_branches = set(range(n)) - set(branches)
        
        # We'll assign paths one pair at a time, using backtracking
        # For efficiency, order pairs by "difficulty" - shortest possible path first
        # Precompute for each pair the shortest path avoiding other branches
        
        def bfs_shortest_path(src, dst, forbidden):
            """Find shortest path from src to dst avoiding forbidden vertices (except src, dst)."""
            if src == dst:
                return [src]
            visited = {src}
            queue = [(src, [src])]
            while queue:
                node, path = queue.pop(0)
                for nb in sorted(adj[node]):
                    if nb == dst:
                        return path + [nb]
                    if nb not in forbidden and nb not in visited:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
            return None
        
        # Order pairs by their shortest path length (ascending) for better pruning
        pair_infos = []
        for u, v in pairs:
            # Forbidden: all branches except u and v
            forbidden = set(branches) - {u, v}
            sp = bfs_shortest_path(u, v, forbidden)
            if sp is None:
                return None  # Can't even connect this pair
            pair_infos.append((len(sp) - 1, u, v, sp))  # -1 for interior vertices count
        
        pair_infos.sort(key=lambda x: x[0])
        
        # Backtracking search
        used_interior = set()
        paths = {}
        
        def backtrack(idx):
            if time_up():
                return False
            if idx == len(pair_infos):
                return True
            
            _, u, v, sp = pair_infos[idx]
            
            # Try the precomputed shortest path first
            # Check if its interior vertices are available
            interior = sp[1:-1]
            if all(vv not in used_interior for vv in interior):
                for vv in interior:
                    used_interior.add(vv)
                paths[(u, v)] = sp
                if backtrack(idx + 1):
                    return True
                del paths[(u, v)]
                for vv in interior:
                    used_interior.discard(vv)
            
            # Try to find alternative paths
            # Use BFS to find paths with increasing length
            forbidden = set(branches) - {u, v}
            max_len = n - 1  # Upper bound on path length
            
            # BFS with increasing path length, skipping already-used interior vertices
            def find_paths(src, dst, forbidden, max_depth, exclude_vertices):
                """Find all simple paths from src to dst up to max_depth, avoiding forbidden and exclude_vertices."""
                results = []
                visited = {src}
                queue = [(src, [src])]
                while queue:
                    node, path = queue.pop(0)
                    if len(path) > max_depth:
                        continue
                    for nb in sorted(adj[node]):
                        if nb == dst:
                            results.append(path + [nb])
                        elif nb not in forbidden and nb not in exclude_vertices and nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
                return results
            
            # Try paths of increasing length
            current_len = len(sp)
            while current_len <= max_len:
                # Find paths of exactly current_len
                candidates = find_paths(u, v, forbidden, current_len, used_interior)
                for cand in candidates:
                    interior = cand[1:-1]
                    if all(vv not in used_interior for vv in interior):
                        for vv in interior:
                            used_interior.add(vv)
                        paths[(u, v)] = cand
                        if backtrack(idx + 1):
                            return True
                        del paths[(u, v)]
                        for vv in interior:
                            used_interior.discard(vv)
                current_len += 1
                if time_up():
                    return False
            
            return False
        
        if backtrack(0):
            # Convert to proper format
            result = {}
            for (u, v), path in paths.items():
                result[(min(u, v), max(u, v))] = path
            return result
        else:
            return None
    
    # Find maximum clique subdivision
    # Strategy: use Bron-Kerbosch to enumerate maximal cliques in the "connection graph"
    # where two vertices are connected if there's a path between them (always true in connected graph)
    # Instead, we'll try all subsets in decreasing size order
    
    # First, check if the graph is connected
    def is_connected():
        visited = set([0])
        queue = [0]
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == n
    
    if not is_connected():
        # Fallback: just return an edge
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Try all subsets of vertices in decreasing size
    # For efficiency, start with large sets and work down
    best_result = None
    best_size = 0
    
    # Precompute: for each pair, whether they can be connected (always true in connected graph)
    # The constraint is on disjointness of paths
    
    # Use a smarter approach: start with a large candidate set and try to shrink it
    # Or: use local search - start with a valid small set and try to grow it
    
    # Approach 1: Try all k-subsets for k from n down to 2
    # This is too expensive for n=24, so we need heuristics
    
    # Approach 2: Greedy + local search
    # Start with a random or heuristic initial set, then try to add vertices
    
    def greedy_grow(initial_branches):
        """Try to grow the branch set by adding vertices one at a time."""
        if time_up():
            return None
        
        branches = set(initial_branches)
        paths = try_pack_paths(branches)
        if paths is None:
            return None
        
        # Try to add vertices
        changed = True
        while changed and not time_up():
            changed = False
            for v in range(n):
                if v in branches:
                    continue
                new_branches = branches | {v}
                new_paths = try_pack_paths(new_branches)
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
                    changed = True
                    break  # Restart with new branch set
        
        return {'branches': list(branches), 'paths': [list(paths[(min(u,v), max(u,v))]) for u, v in combinations(sorted(branches), 2)]}
    
    # Try multiple starting points
    # Start with high-degree vertices
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # Try starting with top-k vertices for various k
    for k in range(2, min(n+1, 10)):
        if time_up():
            break
        initial = sorted_vertices[:k]
        result = greedy_grow(initial)
        if result is not None:
            if len(result['branches']) > best_size:
                best_size = len(result['branches'])
                best_result = result
    
    # Also try random starts
    import random
    random.seed(42)
    for _ in range(50):
        if time_up():
            break
        k = random.randint(2, 8)
        initial = random.sample(range(n), k)
        result = greedy_grow(initial)
        if result is not None:
            if len(result['branches']) > best_size:
                best_size = len(result['branches'])
                best_result = result
    
    # Also try to improve by swapping vertices
    if best_result is not None and not time_up():
        current_branches = set(best_result['branches'])
        improved = True
        while improved and not time_up():
            improved = False
            for v_in in current_branches:
                for v_out in range(n):
                    if v_out in current_branches:
                        continue
                    new_branches = (current_branches - {v_in}) | {v_out}
                    new_paths = try_pack_paths(new_branches)
                    if new_paths is not None:
                        # Check if this is actually an improvement (same size, but maybe enables future growth)
                        # For now, accept it and try to grow
                        test_result = greedy_grow(list(new_branches))
                        if test_result is not None and len(test_result['branches']) > best_size:
                            best_size = len(test_result['branches'])
                            best_result = test_result
                            current_branches = set(test_result['branches'])
                            improved = True
                            break
                if improved:
                    break
    
    if best_result is None:
        # Fallback
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure proper types
    branches = [int(b) for b in best_result['branches']]
    paths = [list(p) for p in best_result['paths']]
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
