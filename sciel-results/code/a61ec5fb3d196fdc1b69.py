"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    """Compute all-pairs shortest paths via BFS from each vertex."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    queue.append(v)
    return dist

def _check_solution(added_edges, n):
    """Check triangle-free, connected, diameter <= 4."""
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check triangle-free
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj[v]:
                    return False
    
    # Check connected and diameter
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def _build_adj(added_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _diameter_ok(added_edges, n):
    """Check if diameter <= 4 for the given added edges."""
    adj = _build_adj(added_edges, n)
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] > 4:
                return False
    return True

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 85
    
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u, v), max(u, v)))
    
    even_vertices = list(range(0, n, 2))
    odd_vertices = list(range(1, n, 2))
    
    # Generate all valid candidate edges (even-odd pairs not in path)
    candidates = []
    for e in even_vertices:
        for o in odd_vertices:
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
    
    # Helper: check if a set of edges gives valid solution
    def check_and_count(edge_list):
        if time.time() - start_time > deadline:
            return None
        if _check_solution(edge_list, n):
            return len(edge_list)
        return None
    
    # Strategy 1: Try small edge sets systematically
    # For k edges, try to find valid configurations
    
    best_count = 999
    best_solution = None
    
    # Try k=1 to k=8
    for k in range(1, 9):
        if time.time() - start_time > deadline:
            break
        
        # Generate all combinations of k candidates is too expensive for large k
        # Instead, use greedy + local search for each k
        
        # Greedy: pick edges that reduce diameter most
        selected = []
        adj = _build_adj([], n)
        dist = _bfs_distances(adj, n)
        
        for _ in range(k):
            if time.time() - start_time > deadline:
                break
            
            # Find pairs with distance > 4
            bad_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if dist[i][j] > 4]
            if not bad_pairs:
                break
            
            best_edge = None
            best_score = -1
            
            for c in candidates:
                if c in selected:
                    continue
                u, v = c
                # Count how many bad pairs this edge would fix
                fixed = 0
                for (a, b) in bad_pairs:
                    # New distance via this edge
                    nd1 = dist[a][u] + 1 + dist[v][b] if dist[a][u] >= 0 and dist[v][b] >= 0 else 999
                    nd2 = dist[a][v] + 1 + dist[u][b] if dist[a][v] >= 0 and dist[u][b] >= 0 else 999
                    new_d = min(nd1, nd2, dist[a][b])
                    if new_d <= 4:
                        fixed += 1
                if fixed > best_score:
                    best_score = fixed
                    best_edge = c
            
            if best_edge is None or best_score == 0:
                break
            
            selected.append(list(best_edge))
            adj = _build_adj(selected, n)
            dist = _bfs_distances(adj, n)
        
        # Check if greedy solution is valid
        if selected:
            cnt = check_and_count(selected)
            if cnt is not None and cnt < best_count:
                best_count = cnt
                best_solution = selected
        
        # Local search: try swapping edges
        if best_solution and best_count > k:
            # Try to improve by swapping
            improved = True
            while improved and time.time() - start_time < deadline:
                improved = False
                current = [list(e) for e in best_solution]
                for i in range(len(current)):
                    if time.time() - start_time > deadline:
                        break
                    for c in candidates:
                        if c in current:
                            continue
                        # Try replacing current[i] with c
                        trial = [list(e) for e in current]
                        trial[i] = list(c)
                        # Check for duplicates
                        if len(set(map(tuple, trial))) != len(trial):
                            continue
                        cnt = check_and_count(trial)
                        if cnt is not None and cnt < best_count:
                            best_count = cnt
                            best_solution = trial
                            improved = True
                            break
    
    # Strategy 2: Hub-based with optimization
    # Try pairs of hubs
    for h1_idx in range(len(even_vertices)):
        if time.time() - start_time > deadline:
            break
        for h2_idx in range(h1_idx + 1, len(even_vertices)):
            if time.time() - start_time > deadline:
                break
            h1 = even_vertices[h1_idx]
            h2 = even_vertices[h2_idx]
            
            # Assign each odd vertex to closer hub
            edges = []
            for o in odd_vertices:
                d1 = abs(o - h1)
                d2 = abs(o - h2)
                if d1 <= d2:
                    if (min(h1, o), max(h1, o)) not in path_edges:
                        edges.append([h1, o])
                else:
                    if (min(h2, o), max(h2, o)) not in path_edges:
                        edges.append([h2, o])
            
            if len(edges) < best_count:
                cnt = check_and_count(edges)
                if cnt is not None:
                    best_count = cnt
                    best_solution = edges
    
    # Strategy 3: Try specific known good patterns
    # Pattern: connect vertices to create "highways"
    # E.g., edges (0,5), (5,10), (10,15), (15,20) create shortcuts
    
    # Try a set of strategic edges
    strategic_sets = [
        # Set 1: Long-range connections
        [[0, 5], [5, 10], [10, 15], [15, 20]],
        [[0, 6], [6, 12], [12, 18], [18, 23]],
        [[1, 7], [7, 13], [13, 19]],
        [[0, 5], [5, 11], [11, 17], [17, 23]],
        [[1, 6], [6, 12], [12, 18]],
        [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20]],
        [[1, 5], [5, 9], [9, 13], [13, 17], [17, 21]],
        [[0, 5], [4, 9], [9, 14], [14, 19], [19, 23]],
        [[1, 6], [6, 11], [11, 16], [16, 21]],
        [[0, 4], [5, 9], [10, 14], [15, 19], [20, 23]],
    ]
    
    for s in strategic_sets:
        if time.time() - start_time > deadline:
            break
        # Filter out path edges
        valid_edges = []
        for u, v in s:
            if (min(u, v), max(u, v)) not in path_edges:
                valid_edges.append([u, v])
        if valid_edges and len(valid_edges) < best_count:
            cnt = check_and_count(valid_edges)
            if cnt is not None:
                best_count = cnt
                best_solution = valid_edges
    
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: hub at 12
    hub = 12
    fallback = []
    for v in odd_vertices:
        if (min(hub, v), max(hub, v)) not in path_edges:
            fallback.append([hub, v])
    return [[int(u), int(v)] for u, v in fallback]
# EVOLVE-BLOCK-END
