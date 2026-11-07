"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_cycle_lengths(n, edges):
    """Return set of distinct simple cycle lengths in a graph using DFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited, depth):
        if depth > 25:
            return
        for neighbor in adj[current]:
            if neighbor == start and depth >= 4:
                cycle_lengths.add(depth)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, visited, depth + 1)
                visited.remove(neighbor)
    
    for v in range(n):
        dfs(v, v, {v}, 1)
    
    return cycle_lengths


def _validate_and_score(edges, n=12):
    """Validate constraints and return number of distinct cycle lengths, or None if invalid."""
    edge_set = set()
    for u, v in edges:
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d < 3:
            return None
    
    # Triangle check
    adj_set = set()
    for u, v in edges:
        adj_set.add((min(u,v), max(u,v)))
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in adj_set:
                for w in range(v+1, n):
                    if (u, w) in adj_set and (v, w) in adj_set:
                        return None
    
    cycle_lengths = _get_cycle_lengths(n, edges)
    if not cycle_lengths:
        return None
    
    return len(cycle_lengths)


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    n = 12
    best_score = 3
    best_edges = None
    
    # Known good: two K_{3,3}
    edges_k33 = []
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges_k33.append([u, v])
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges_k33.append([u, v])
    
    score = _validate_and_score(edges_k33, n)
    if score is not None and score < best_score:
        best_score = score
        best_edges = edges_k33
    
    rng = random.Random(42)
    
    # Try random bipartite graphs (6+6 split)
    for trial in range(2000):
        if time.time() > deadline:
            break
        edges = []
        for i in range(6):
            for j in range(6):
                if rng.random() < 0.5:
                    edges.append([i, j + 6])
        deg = [0] * 12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            continue
        s = _validate_and_score(edges, n)
        if s is not None and s < best_score:
            best_score = s
            best_edges = edges
            if best_score == 1:
                break
    
    # Try modifying K_{3,3}+K_{3,3}
    if time.time() < deadline:
        for trial in range(2000):
            if time.time() > deadline:
                break
            edges = [list(e) for e in edges_k33]
            for _ in range(rng.randint(1, 2)):
                idx = rng.randint(0, len(edges) - 1)
                edges.pop(idx)
            for _ in range(rng.randint(1, 2)):
                u = rng.randint(0, 11)
                v = rng.randint(0, 11)
                if u != v and [u, v] not in edges and [v, u] not in edges:
                    edges.append([u, v])
            s = _validate_and_score(edges, n)
            if s is not None and s < best_score:
                best_score = s
                best_edges = edges
                if best_score == 1:
                    break
    
    if best_edges is None:
        best_edges = edges_k33
    
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
