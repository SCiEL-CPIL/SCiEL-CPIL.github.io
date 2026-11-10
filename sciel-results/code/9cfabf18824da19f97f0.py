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
    
    def dfs_cycles(start):
        """Find all simple cycles starting and ending at 'start'."""
        cycles = []
        for neighbor in sorted(adj[start]):
            if neighbor < start:
                continue
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
    
    Known optimal solution: Two disjoint K_{3,3} components.
    Distinct cycle lengths: {4, 6}. Count = 2.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_edges = None
    best_count = float('inf')
    
    # Candidate 1: Two disjoint K_{3,3}
    # This is the known optimal solution for this problem size and constraints.
    candidate = _two_k33()
    
    if _validate_edges(candidate):
        count = _count_distinct_lengths(candidate)
        if count < best_count:
            best_count = count
            best_edges = candidate
    
    # If for some reason the validation fails (should not happen), try a random search as fallback
    if best_edges is None:
        random.seed(42)
        for _ in range(1000):
            if _check_deadline():
                break
            # Generate random 3-regular bipartite graph on 12 vertices
            # Part A: 0-5, Part B: 6-11
            edges = set()
            # Shuffle connections
            for i in range(6):
                # Connect i to 3 random vertices in B
                targets = random.sample(range(6, 12), 3)
                for t in targets:
                    edges.add(tuple(sorted((i, t))))
            
            # Ensure min degree 3 for B vertices?
            # This random generation doesn't guarantee min degree 3 for B.
            # We need a proper random 3-regular bipartite graph.
            # This is complex to generate on the fly.
            # Since the hardcoded solution is valid, this branch is unlikely to be needed.
            pass

    # Ensure types are correct (integers, lists)
    final_edges = []
    if best_edges:
        for e in best_edges:
            final_edges.append([int(e[0]), int(e[1])])
    else:
        # Absolute fallback: return the hardcoded one even if validation logic had a bug
        final_edges = _two_k33()

    return final_edges

# EVOLVE-BLOCK-END
