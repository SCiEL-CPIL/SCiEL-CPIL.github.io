import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 80.0  # Safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > _TIMEOUT

def _get_adjacency(edges, n=12):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_all_cycle_lengths(adj, n=12):
    """Find all distinct simple cycle lengths in the graph."""
    lengths = set()
    
    def dfs_cycles(start):
        """Find all simple cycles starting and ending at 'start'."""
        cycles = []
        for neighbor in sorted(adj[start]):
            if neighbor < start:
                continue
            path = [start, neighbor]
            visited = {start, neighbor}
            
            def dfs(current, path, visited):
                if len(path) > n:
                    return
                for next_v in sorted(adj[current]):
                    if next_v == start and len(path) >= 4:
                        cycles.append(len(path))
                    elif next_v not in visited and next_v > start:
                        visited.add(next_v)
                        path.append(next_v)
                        dfs(next_v, path, visited)
                        path.pop()
                        visited.discard(next_v)
            
            dfs(neighbor, path, visited)
        return cycles
    
    for start in range(n):
        if _check_deadline():
            break
        cys = dfs_cycles(start)
        for c in cys:
            lengths.add(c)
    
    return lengths

def _validate_edges(edges, n=12):
    """Check if edges form a valid graph: simple, undirected, min degree >= 3, triangle-free, has cycle."""
    if not edges:
        return False
    adj = _get_adjacency(edges, n)
    
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    
    # Check no triangles
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return False
    
    # Check has at least one cycle
    lengths = _find_all_cycle_lengths(adj, n)
    if not lengths:
        return False
    
    return True

def _count_distinct_lengths(edges, n=12):
    """Count distinct simple cycle lengths."""
    adj = _get_adjacency(edges, n)
    lengths = _find_all_cycle_lengths(adj, n)
    return len(lengths)

def _two_k33():
    """Fallback: two disjoint K_{3,3} components (2 distinct cycle lengths)."""
    edges = set()
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.add(tuple(sorted((u, v))))
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(edges)]

def _try_circulant_bipartite(offsets):
    """Try a circulant bipartite graph with given offsets."""
    edges = set()
    for i in range(6):
        for j in offsets:
            u = i
            v = 6 + (i + j) % 6
            edges.add(tuple(sorted((u, v))))
    
    edge_list = [list(e) for e in sorted(edges)]
    if _validate_edges(edge_list):
        return edge_list
    return None

def construct():
    """
    Search for a graph on 12 vertices minimizing distinct simple cycle lengths.
    Constraints: min degree >= 3, triangle-free, contains a cycle.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_edges = None
    best_count = 3
    
    # First, try the known solution (two K_{3,3})
    fallback = _two_k33()
    best_edges = fallback
    best_count = _count_distinct_lengths(fallback)
    
    # Try various circulant bipartite patterns
    # Pattern 1: {0, 1, 2}
    circ1 = _try_circulant_bipartite([0, 1, 2])
    if circ1:
        count = _count_distinct_lengths(circ1)
        if count < best_count:
            best_count = count
            best_edges = circ1
    
    # Pattern 2: {0, 1, 3}
    circ2 = _try_circulant_bipartite([0, 1, 3])
    if circ2:
        count = _count_distinct_lengths(circ2)
        if count < best_count:
            best_count = count
            best_edges = circ2
    
    # Pattern 3: {0, 2, 3}
    circ3 = _try_circulant_bipartite([0, 2, 3])
    if circ3:
        count = _count_distinct_lengths(circ3)
        if count < best_count:
            best_count = count
            best_edges = circ3
    
    # Pattern 4: {1, 2, 3}
    circ4 = _try_circulant_bipartite([1, 2, 3])
    if circ4:
        count = _count_distinct_lengths(circ4)
        if count < best_count:
            best_count = count
            best_edges = circ4
    
    # Pattern 5: {0, 2, 4}
    circ5 = _try_circulant_bipartite([0, 2, 4])
    if circ5:
        count = _count_distinct_lengths(circ5)
        if count < best_count:
            best_count = count
            best_edges = circ5
    
    # If we found a 1-length solution, return it
    if best_count == 1:
        return best_edges
    
    # Try random local search from the two K_{3,3} structure
    if best_count > 1 and not _check_deadline():
        random.seed(42)
        current_edges = set(tuple(sorted(e)) for e in fallback)
        current_count = best_count
        
        for iteration in range(2000):
            if _check_deadline():
                break
            
            # Try a random edge swap
            edge_list = list(current_edges)
            if len(edge_list) < 2:
                break
            
            e1 = random.choice(edge_list)
            e2 = random.choice(edge_list)
            if e1 == e2:
                continue
            
            # Remove e1, add e2 (if not already present)
            if e2 in current_edges:
                continue
                
            new_edges = set(current_edges)
            new_edges.discard(e1)
            new_edges.add(e2)
            
            # Ensure no duplicate edges
            if len(new_edges) != len(current_edges):
                continue
            
            edge_list_new = [list(e) for e in sorted(new_edges)]
            
            if _validate_edges(edge_list_new):
                new_count = _count_distinct_lengths(edge_list_new)
                if new_count < current_count:
                    current_edges = new_edges
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_edges = edge_list_new
                    if best_count == 1:
                        break
    
    return best_edges

def _validate(edges):
    """
    Internal validation to ensure the graph meets all constraints.
    """
    n = 12
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    
    for u, v in edges:
        if u == v:
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Check min degree
    if min(deg) < 3:
        return False
        
    # Check for triangles (girth >= 4)
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v:
                    if (min(u, w), max(u, w)) in edge_set:
                        return False
                        
    # Check if it contains a cycle
    return True

def _count_distinct_cycle_lengths(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycle_lengths = set()
    
    def dfs(node, start, path, visited):
        for neighbor in adj[node]:
            if neighbor == start and len(path) >= 3:
                cycle_lengths.add(len(path))
            elif not visited[neighbor] and neighbor > start:
                visited[neighbor] = True
                dfs(neighbor, start, path + [neighbor], visited)
                visited[neighbor] = False

    for i in range(n):
        dfs(i, i, [i], {i: True})
        
    return len(cycle_lengths)

def _get_all_cycle_lengths(edges, n):
    return _count_distinct_cycle_lengths(edges, n)
