"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_adj(base_edges, added_edges, n):
    """Build adjacency lists from base and added edges."""
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_distances(adj, n, start):
    """BFS from start, return distance array."""
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _check_diameter(adj, n, max_diam=4):
    """Check if diameter <= max_diam. Returns True if valid."""
    for start in range(n):
        dist = _bfs_distances(adj, n, start)
        if -1 in dist:
            return False
        if max(dist) > max_diam:
            return False
    return True


def _is_valid(added_edges, base_edges, n):
    """Check all constraints."""
    # Check no duplicate/existing edges
    existing = set()
    for u, v in base_edges:
        existing.add(frozenset((u, v)))
    for u, v in added_edges:
        fs = frozenset((u, v))
        if fs in existing:
            return False
        existing.add(fs)
    
    # Build adjacency
    adj = _build_adj(base_edges, added_edges, n)
    
    # Check bipartite (triangle-free for this structure)
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
    
    # Check diameter
    if not _check_diameter(adj, n, 4):
        return False
    
    return True


def _get_candidate_edges(base_edges, n):
    """Get all valid candidate edges (opposite parity, not in base)."""
    base_set = set()
    for u, v in base_edges:
        base_set.add(frozenset((u, v)))
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if u % 2 != v % 2:  # opposite parity ensures bipartite
                if frozenset((u, v)) not in base_set:
                    candidates.append((u, v))
    return candidates


def _greedy_initial(base_edges, n, deadline):
    """Greedy: add edges to reduce diameter."""
    added = []
    current_edges = list(base_edges)
    
    while time.time() < deadline:
        adj = _build_adj(current_edges, added, n)
        if _check_diameter(adj, n, 4):
            break
        
        # Find pair with max distance
        best_pair = None
        best_dist = 0
        for start in range(n):
            dist = _bfs_distances(adj, n, start)
            for end in range(n):
                if dist[end] > best_dist:
                    best_dist = dist[end]
                    best_pair = (start, end)
        
        if best_pair is None or best_dist <= 4:
            break
        
        u, v = best_pair
        if u % 2 == v % 2:
            # Can't add same-parity edge (would break bipartite)
            # Try nearby opposite-parity pairs
            found = False
            for du in [-1, 1]:
                for dv in [-1, 1]:
                    nu, nv = u + du, v + dv
                    if 0 <= nu < n and 0 <= nv < n and nu % 2 != nv % 2:
                        fs = frozenset((nu, nv))
                        base_set = set(frozenset((a, b)) for a, b in base_edges)
                        if fs not in base_set and fs not in set(frozenset((a, b)) for a, b in added):
                            test_added = added + [(nu, nv)]
                            test_adj = _build_adj(base_edges, test_added, n)
                            if _check_diameter(test_adj, n, 4):
                                added.append((nu, nv))
                                found = True
                                break
                if found:
                    break
            if not found:
                break
        else:
            added.append((u, v))
    
    return added


def construct():
    deadline = time.time() + 85  # Safety margin
    
    n = P['n']
    base_edges = P['edges']
    
    # Get all candidate edges (opposite parity, not in base)
    candidates = _get_candidate_edges(base_edges, n)
    
    # Start with greedy solution
    best_added = _greedy_initial(base_edges, n, deadline)
    
    if not _is_valid(best_added, base_edges, n):
        # Fallback: complete bipartite
        best_added = [list(e) for e in candidates]
        if not _is_valid(best_added, base_edges, n):
            return []
    
    best_count = len(best_added)
    
    # Iterative removal: try removing each edge if still valid
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(best_added)):
            if time.time() > deadline - 5:
                break
            candidate = best_added[:i] + best_added[i+1:]
            if _is_valid(candidate, base_edges, n):
                best_added = candidate
                best_count = len(best_added)
                improved = True
                break
    
    # Try edge swaps: remove one, add another that enables further removal
    if time.time() < deadline - 10:
        for i in range(len(best_added)):
            if time.time() > deadline - 10:
                break
            for edge in candidates:
                if edge in best_added:
                    continue
                candidate = best_added[:i] + best_added[i+1:] + [list(edge)]
                if _is_valid(candidate, base_edges, n):
                    # Check if this enables removing another edge
                    for j in range(len(candidate)):
                        if candidate[j] == list(edge):
                            continue
                        test = candidate[:j] + candidate[j+1:]
                        if _is_valid(test, base_edges, n):
                            best_added = test
                            best_count = len(best_added)
                            improved = True
                            break
                    if improved:
                        break
    
    # Final removal pass
    improved2 = True
    while improved2 and time.time() < deadline - 3:
        improved2 = False
        for i in range(len(best_added)):
            candidate = best_added[:i] + best_added[i+1:]
            if _is_valid(candidate, base_edges, n):
                best_added = candidate
                best_count = len(best_added)
                improved2 = True
                break
    
    # Ensure proper types
    result = []
    for e in best_added:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
