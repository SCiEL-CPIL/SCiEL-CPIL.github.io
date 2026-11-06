"""Path decomposition for Gallai's conjecture - Structural Search Approach."""
import time
from collections import defaultdict, deque
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_all_simple_paths(adj, start, end, max_depth=10):
        """Find all simple paths from start to end up to max_depth."""
        paths = []
        visited = {start}
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node == end:
                if len(path) <= max_depth:
                    paths.append(path)
                continue
            if len(path) >= max_depth:
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
                    visited.remove(neighbor)
        return paths

    def solve_with_fixed_endpoints(edges, pairs):
        """
        Try to decompose 'edges' into len(pairs) edge-disjoint simple paths,
        where the i-th path connects pairs[i][0] and pairs[i][1].
        """
        if not pairs:
            return [] if not edges else None
        
        # Heuristic: Pick the pair with the fewest available paths first
        best_pair_idx = -1
        best_paths = None
        min_count = float('inf')
        
        adj = build_adj(edges)
        
        for i, (s, t) in enumerate(pairs):
            # Check if s and t are connected
            if s not in adj or t not in adj:
                # If s==t, path is [s] (but we need edges, so invalid for edge cover unless no edges)
                # If disconnected, fail
                return None
            paths = find_all_simple_paths(adj, s, t, max_depth=8)
            if len(paths) < min_count:
                min_count = len(paths)
                best_pair_idx = i
                best_paths = paths
            if min_count == 0:
                return None
        
        if min_count == 0:
            return None
            
        s, t = pairs[best_pair_idx]
        remaining_pairs = [p for i, p in enumerate(pairs) if i != best_pair_idx]
        
        # Sort paths by length (shorter first to leave more flexibility? or longer to cover more?)
        # Usually shorter is safer to not "waste" edges, but longer covers more.
        # Let's try shortest first.
        best_paths.sort(key=len)
        
        for path in best_paths:
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_edges.add((min(u, v), max(u, v)))
            
            if not path_edges.issubset(edges):
                continue
                
            new_edges = edges - path_edges
            result = solve_with_fixed_endpoints(new_edges, remaining_pairs)
            if result is not None:
                return [path] + result
                
        return None

    def generate_pairings(vertices):
        """Generate all possible pairings of a list of 2k vertices."""
        if not vertices:
            yield []
            return
        v = vertices[0]
        rest = vertices[1:]
        for i, w in enumerate(rest):
            pair = (v, w)
            new_rest = rest[:i] + rest[i+1:]
            for sub_pairings in generate_pairings(new_rest):
                yield [pair] + sub_pairings

    def decompose_graph(edges):
        if not edges:
            return []
        
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        K = len(odd_vertices) // 2
        
        # Try K_min
        if K > 0:
            # Limit number of pairings to try to avoid explosion
            # For K=10, 9!! = 94590. This might be too many if backtracking is slow.
            # We will try a subset or limit time.
            pairings = list(generate_pairings(odd_vertices))
            random.shuffle(pairings)
            
            # Try first 1000 pairings or until time runs out
            for pairing in pairings:
                if time.time() > DEADLINE - 10:
                    break
                result = solve_with_fixed_endpoints(set(edges), pairing)
                if result is not None:
                    return result
        else:
            # K=0 means Eulerian graph.
            # Decompose into cycles?
            # If K=0, we need 0 paths? No, we need to cover edges.
            # If all degrees are even, we can decompose into cycles.
            # But we need simple paths.
            # A cycle is not a simple path.
            # So we must break cycles.
            # This case is rare for random graphs with path backbone.
            pass
            
        # Fallback: Greedy
        return greedy_decompose(edges)

    def greedy_decompose(edges):
        """Fallback greedy decomposition."""
        remaining = set(edges)
        paths = []
        while remaining:
            adj = build_adj(remaining)
            # Find a long path
            start = None
            for u, vs in adj.items():
                if vs:
                    start = u
                    break
            if start is None:
                break
            
            # Simple BFS/DFS for a path
            visited = {start}
            path = [start]
            stack = [start]
            while stack:
                node = stack[-1]
                next_node = None
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        next_node = neighbor
                        break
                if next_node:
                    visited.add(next_node)
                    path.append(next_node)
                    stack.append(next_node)
                else:
                    stack.pop()
            
            if len(path) > 1:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                paths.append(path)
            else:
                # Single edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.discard(e)
                
        return paths

    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 20:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        paths = decompose_graph(edges)
        results.append(paths)
    
    return results
