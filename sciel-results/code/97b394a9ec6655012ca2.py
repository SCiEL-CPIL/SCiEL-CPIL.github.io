"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj_from_edges(n, edges):
    """Build adjacency list from a list of undirected edges."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    """Check if graph with adjacency list is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                # Check if u and v share a common neighbor
                common = adj[u] & adj[v]
                if common:
                    return False
    return True

def _diameter_leq(n, adj, max_diam):
    """Check if graph diameter is at most max_diam. Returns True if so."""
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diam:
                        return False
                    queue.append(v)
        # Check all nodes are reachable
        if -1 in dist:
            return False
    return True

def _is_valid(n, base_edges, added_edges):
    """Check if the union of base_edges and added_edges is triangle-free, connected, diameter <= 4."""
    all_edges = list(base_edges) + list(added_edges)
    adj = _build_adj_from_edges(n, all_edges)
    
    # Check triangle-free
    if not _is_triangle_free(n, adj):
        return False
    
    # Check diameter <= 4
    if not _diameter_leq(n, adj, 4):
        return False
    
    return True

def _try_remove_edge(n, base_edges, current_edges, idx_to_remove):
    """Try removing the edge at idx_to_remove from current_edges. Return new list if valid, else None."""
    new_edges = [e for i, e in enumerate(current_edges) if i != idx_to_remove]
    if _is_valid(n, base_edges, new_edges):
        return new_edges
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    base_edges = [(u, v) for u, v in P['edges']]
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Initial solution: 10-edge star
    initial_solution = [
        (0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
        (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)
    ]
    
    # Verify initial solution is valid
    if not _is_valid(n, base_edges, initial_solution):
        # Fallback: return empty if initial is somehow invalid
        return []
    
    best = list(initial_solution)
    
    # Phase 1: Greedy edge removal - try to remove edges one at a time
    # Repeat until no more edges can be removed
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            new_edges = _try_remove_edge(n, base_edges, best, i)
            if new_edges is not None:
                best = new_edges
                improved = True
                break  # Restart scan from beginning after a removal
    
    # Phase 2: Try alternative initial solutions and reduce them too
    # Generate different star-like solutions with different centers
    for center in range(4, 20):
        if (time.time() - start_time) > deadline:
            break
        
        # Find valid targets for this center (no triangle, not already edge)
        valid_targets = []
        for v in range(n):
            if v == center:
                continue
            if abs(v - center) <= 2:
                continue  # Would create triangle with path edges
            if (min(center, v), max(center, v)) in base_edge_set:
                continue
            valid_targets.append(v)
        
        # Try connecting center to all valid targets, then reduce
        candidate = [(center, t) for t in valid_targets]
        if not _is_valid(n, base_edges, candidate):
            continue
        
        # Reduce this candidate
        reduced = list(candidate)
        imp = True
        while imp and (time.time() - start_time) < deadline:
            imp = False
            for i in range(len(reduced)):
                if (time.time() - start_time) > deadline:
                    break
                new_edges = _try_remove_edge(n, base_edges, reduced, i)
                if new_edges is not None:
                    reduced = new_edges
                    imp = True
                    break
        
        if len(reduced) < len(best):
            best = reduced
    
    # Phase 3: Try mixed solutions - combine edges from different centers
    # For pairs of centers, try combining their best reduced solutions
    if (time.time() - start_time) < deadline:
        # Try all pairs of centers
        for c1 in range(4, 20):
            if (time.time() - start_time) > deadline:
                break
            for c2 in range(c1 + 1, 20):
                if (time.time() - start_time) > deadline:
                    break
                
                # Get valid targets for each center
                vt1 = []
                vt2 = []
                for v in range(n):
                    if v == c1 or v == c2:
                        continue
                    if abs(v - c1) <= 2 or abs(v - c2) <= 2:
                        continue
                    if (min(c1, v), max(c1, v)) in base_edge_set or (min(c2, v), max(c2, v)) in base_edge_set:
                        continue
                    if abs(c1 - c2) <= 2:
                        continue
                    vt1.append(v)
                    vt2.append(v)
                
                # Try combining: connect c1 to some, c2 to some
                # Start with all, then reduce
                candidate = [(c1, t) for t in vt1] + [(c2, t) for t in vt2]
                if not _is_valid(n, base_edges, candidate):
                    continue
                
                reduced = list(candidate)
                imp = True
                while imp and (time.time() - start_time) < deadline:
                    imp = False
                    for i in range(len(reduced)):
                        if (time.time() - start_time) > deadline:
                            break
                        new_edges = _try_remove_edge(n, base_edges, reduced, i)
                        if new_edges is not None:
                            reduced = new_edges
                            imp = True
                            break
                
                if len(reduced) < len(best):
                    best = reduced
    
    # Final verification
    if _is_valid(n, base_edges, best):
        return [[u, v] for u, v in best]
    else:
        # Fallback to initial solution
        return [[u, v] for u, v in initial_solution]

# EVOLVE-BLOCK-END
