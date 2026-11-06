"""Path decomposition of connected graphs - structural redesign using greedy trail extension."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_greedy(edges, n, deadline, rng):
    """Greedy trail decomposition: build paths by extending at both ends."""
    # Build adjacency with edge tracking
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
    
    used_edges = set()
    paths = []
    
    def try_extend(path, end_vertex, direction):
        """Try to extend path at end_vertex in given direction.
        direction=1: extend at front, direction=-1: extend at back
        Returns True if extended."""
        if not adj[end_vertex]:
            return False
        # Get unused neighbors
        candidates = []
        for nb in adj[end_vertex]:
            e = tuple(sorted((end_vertex, nb)))
            if e not in used_edges:
                # Check simplicity
                if nb not in path:
                    candidates.append(nb)
        if not candidates:
            return False
        # Choose neighbor with fewest remaining unused edges (to avoid getting stuck)
        def score(nb):
            cnt = 0
            for nn in adj[nb]:
                e = tuple(sorted((nb, nn)))
                if e not in used_edges:
                    cnt += 1
            return cnt
        candidates.sort(key=score)
        nb = candidates[0]
        e = tuple(sorted((end_vertex, nb)))
        used_edges.add(e)
        if direction == 1:
            path.insert(0, nb)
        else:
            path.append(nb)
        return True
    
    # Sort edges by... let's just iterate
    # Start with each unused edge as a seed path
    remaining = list(edge_set)
    random.Random(rng).shuffle(remaining)
    
    for e in remaining:
        if e in used_edges:
            continue
        u, v = e
        used_edges.add(e)
        path = [u, v]
        
        # Extend at both ends greedily
        changed = True
        while changed:
            changed = False
            # Try to extend at front
            if try_extend(path, path[0], 1):
                changed = True
            # Try to extend at back
            if try_extend(path, path[-1], -1):
                changed = True
        
        paths.append(path)
    
    return paths

def _decompose_with_local_search(edges, n, deadline, rng):
    """Improved: use better heuristics and local search."""
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
    
    used_edges = set()
    paths = []
    
    def try_extend(path, end_vertex, direction):
        if not adj[end_vertex]:
            return False
        candidates = []
        for nb in adj[end_vertex]:
            e = tuple(sorted((end_vertex, nb)))
            if e not in used_edges and nb not in path:
                candidates.append(nb)
        if not candidates:
            return False
        # Prefer neighbors with fewer options (Warnsdorff-like)
        def score(nb):
            cnt = 0
            for nn in adj[nb]:
                e = tuple(sorted((nb, nn)))
                if e not in used_edges:
                    cnt += 1
            return cnt
        candidates.sort(key=score)
        nb = candidates[0]
        e = tuple(sorted((end_vertex, nb)))
        used_edges.add(e)
        if direction == 1:
            path.insert(0, nb)
        else:
            path.append(nb)
        return True
    
    # Process edges in order of endpoint degree (high degree first for better coverage)
    edge_list = list(edge_set)
    # Sort by min degree of endpoints
    deg = {u: len(adj[u]) for u in range(n)}
    edge_list.sort(key=lambda e: min(deg[e[0]], deg[e[1]]), reverse=True)
    
    for e in edge_list:
        if e in used_edges:
            continue
        u, v = e
        used_edges.add(e)
        path = [u, v]
        
        # Aggressive extension
        for _ in range(100):  # max extensions per path
            extended = False
            if try_extend(path, path[0], 1):
                extended = True
            if try_extend(path, path[-1], -1):
                extended = True
            if not extended:
                break
        
        paths.append(path)
    
    return paths

def _decompose_eulerian_based(edges, n, deadline):
    """Use Eulerian trail decomposition with parity fixing."""
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
    
    # Find odd-degree vertices
    deg = {u: len(adj[u]) for u in range(n)}
    odd_vertices = [u for u in range(n) if deg[u] % 2 == 1]
    
    # We need to pair up odd vertices with paths
    # Minimum number of paths >= ceil(len(odd_vertices)/2)
    
    # Strategy: find paths between pairs of odd vertices that cover edges
    # Use a greedy approach: repeatedly find a longest path between two unused odd vertices
    # that covers as many edges as possible
    
    used_edges = set()
    paths = []
    remaining_odd = list(odd_vertices)
    
    # First, handle even-degree components by finding Eulerian circuits
    # Actually, let's use a simpler approach: just do greedy trail finding
    # but start from odd vertices
    
    # Build list of edges to process
    edge_list = list(edge_set)
    
    # Start with paths from odd vertices
    for start in remaining_odd:
        if all(tuple(sorted((start, nb))) in used_edges for nb in adj[start]):
            continue
        # Find a long trail starting from start
        path = [start]
        used_in_path = {start}
        current = start
        while True:
            candidates = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if e not in used_edges and nb not in used_in_path:
                    candidates.append(nb)
            if not candidates:
                break
            # Choose next vertex
            def score(nb):
                cnt = 0
                for nn in adj[nb]:
                    e = tuple(sorted((nb, nn)))
                    if e not in used_edges:
                        cnt += 1
                return cnt
            candidates.sort(key=score)
            nb = candidates[0]
            e = tuple(sorted((current, nb)))
            used_edges.add(e)
            path.append(nb)
            used_in_path.add(nb)
            current = nb
        
        if len(path) > 1:
            paths.append(path)
    
    # Now handle remaining edges (even-degree subgraph)
    # Find Eulerian trails in remaining
    remaining_edges = [e for e in edge_set if e not in used_edges]
    if not remaining_edges:
        return paths
    
    # Build adjacency for remaining
    rem_adj = defaultdict(list)
    for u, v in remaining_edges:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
    
    # Find connected components and do Eulerian trails
    visited_nodes = set()
    for start in range(n):
        if start in rem_adj and start not in visited_nodes:
            # BFS to find component
            comp = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in comp:
                    continue
                comp.add(node)
                for nb in rem_adj[node]:
                    if nb not in comp:
                        queue.append(nb)
            visited_nodes |= comp
            
            # Find Eulerian trail in this component
            # Check degrees
            rem_deg = {u: len(rem_adj[u]) for u in comp}
            odd_in_comp = [u for u in comp if rem_deg[u] % 2 == 1]
            
            # Find Eulerian trail using Hierholzer's
            trail = []
            if not comp:
                continue
            
            # Use iterative Hierholzer
            from collections import defaultdict as dd
            rem_adj2 = dd(list)
            for u, v in remaining_edges:
                if u in comp and v in comp:
                    rem_adj2[u].append(v)
                    rem_adj2[v].append(u)
            
            if not rem_adj2:
                continue
            
            # Find start vertex
            if odd_in_comp:
                trail_start = odd_in_comp[0]
            else:
                trail_start = list(comp)[0]
            
            # Hierholzer's algorithm
            stack = [trail_start]
            circuit = []
            adj_copy = dd(list)
            for u in rem_adj2:
                adj_copy[u] = list(rem_adj2[u])
            
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    w = adj_copy[v].pop()
                    stack.append(w)
                else:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            if len(circuit) > 1:
                # Mark edges as used
                for i in range(len(circuit) - 1):
                    e = tuple(sorted((circuit[i], circuit[i+1])))
                    used_edges.add(e)
                paths.append(circuit)
    
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        
        if time.time() > deadline - 20:
            # Fallback: each edge as its own path
            results.append([[u, v] for u, v in edges])
            continue
        
        # Try multiple strategies and pick best
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Greedy with Warnsdorff
        paths1 = _decompose_with_local_search(edges, n, deadline, seed)
        count1 = len(paths1)
        if count1 < best_count:
            best_count = count1
            best_paths = paths1
        
        # Strategy 2: Eulerian-based
        if time.time() < deadline - 10:
            paths2 = _decompose_eulerian_based(edges, n, deadline)
            count2 = len(paths2)
            if count2 < best_count:
                best_count = count2
                best_paths = paths2
        
        # Strategy 3: Simple greedy
        if time.time() < deadline - 5:
            paths3 = _decompose_greedy(edges, n, deadline, seed)
            count3 = len(paths3)
            if count3 < best_count:
                best_count = count3
                best_paths = paths3
        
        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]
        
        results.append(best_paths)
    
    return results

