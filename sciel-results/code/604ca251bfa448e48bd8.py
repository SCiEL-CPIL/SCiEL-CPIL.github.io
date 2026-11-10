"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 85.0  # Safety margin before 90s kill

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
    # Use DFS to find all simple cycles (for small graphs this is feasible)
    # We'll find cycles by checking all pairs of vertices and paths between them
    
    # For each pair of vertices, find shortest path and all path lengths
    # Actually, let's use a more direct approach for small n
    
    # Find all simple cycles using Johnson's algorithm simplified for n=12
    # Since n is small, we can use a different approach:
    # For each starting vertex, do DFS and track visited
    
    def dfs_cycles(start):
        """Find all simple cycles starting and ending at 'start'."""
        cycles = []
        # Try each neighbor as first step
        for neighbor in sorted(adj[start]):
            if neighbor < start:
                continue  # Only count cycles where start is the smallest vertex
            path = [start, neighbor]
            visited = {start, neighbor}
            
            def dfs(current, path, visited):
                if len(path) > 12:
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

def construct():
    """
    Search for a graph on 12 vertices minimizing distinct simple cycle lengths.
    Constraints: min degree >= 3, triangle-free, contains a cycle.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_edges = None
    best_count = 3  # Start with a known upper bound
    
    # First, try the known solution (two K_{3,3})
    fallback = _two_k33()
    best_edges = fallback
    best_count = _count_distinct_lengths(fallback)
    
    # Try to find a graph with 1 distinct cycle length
    # Strategy: search over bipartite graphs with specific structures
    
    # Approach 1: Try graphs where all cycles are length 4
    # This requires a bipartite graph with no 6-cycles, 8-cycles, etc.
    
    # Let's try a specific construction:
    # A 3-regular bipartite graph on 12 vertices that avoids cycles longer than 4
    
    # Construction idea: Use a "circular" arrangement
    # Part A: {0,1,2,3,4,5}, Part B: {6,7,8,9,10,11}
    # Connect with a pattern that creates only 4-cycles
    
    # Let me try: each a_i connects to b_{i}, b_{i+1}, b_{i+2} mod 6
    # This is a circulant bipartite graph
    
    # Test this construction
    edges_circ = set()
    for i in range(6):
        for j in [i, (i+1)%6, (i+2)%6]:
            u = i
            v = 6 + j
            edges_circ.add(tuple(sorted((u, v))))
    
    if _validate_edges(list(edges_circ)):
        count = _count_distinct_lengths(list(edges_circ))
        if count < best_count:
            best_count = count
            best_edges = [list(e) for e in sorted(edges_circ)]
    
    # Try another circulant pattern: b_i, b_{i+1}, b_{i+3}
    edges_circ2 = set()
    for i in range(6):
        for j in [i, (i+1)%6, (i+3)%6]:
            u = i
            v = 6 + j
            edges_circ2.add(tuple(sorted((u, v))))
    
    if _validate_edges(list(edges_circ2)):
        count = _count_distinct_lengths(list(edges_circ2))
        if count < best_count:
            best_count = count
            best_edges = [list(e) for e in sorted(edges_circ2)]
    
    # Try: b_i, b_{i+2}, b_{i+3}
    edges_circ3 = set()
    for i in range(6):
        for j in [i, (i+2)%6, (i+3)%6]:
            u = i
            v = 6 + j
            edges_circ3.add(tuple(sorted((u, v))))
    
    if _validate_edges(list(edges_circ3)):
        count = _count_distinct_lengths(list(edges_circ3))
        if count < best_count:
            best_count = count
            best_edges = [list(e) for e in sorted(edges_circ3)]
    
    # If we found a 1-length solution, return it
    if best_count == 1:
        return best_edges
    
    # Try random search for improvements
    if best_count > 1 and not _check_deadline():
        # Random local search from the two K_{3,3} structure
        random.seed(42)
        current_edges = set(tuple(sorted(e)) for e in fallback)
        current_count = best_count
        
        for iteration in range(5000):
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
            
            new_edges = current_edges - {e1, e2}
            # Add a random new edge
            all_pairs = list(combinations(range(12), 2))
            candidates = [p for p in all_pairs if p not in new_edges]
            if not candidates:
                continue
            new_e = tuple(sorted(random.choice(candidates)))
            new_edges.add(new_e)
            
            if len(new_edges) != len(current_edges):
                continue
            
            new_edge_list = list(new_edges)
            if _validate_edges(new_edge_list):
                new_count = _count_distinct_lengths(new_edge_list)
                if new_count < current_count:
                    current_edges = new_edges
                    current_count = new_count
                    if current_count == 1:
                        break
    
        if current_count < best_count:
            best_count = current_count
            best_edges = [list(e) for e in sorted(current_edges)]
    
    return best_edges

# EVOLVE-BLOCK-END
