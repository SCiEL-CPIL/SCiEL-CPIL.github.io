"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_allpairs(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            du = dist[s][u]
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = du + 1
                    queue.append(v)
    return dist

def _check_solution(added_edges, n):
    adj = _build_adj(added_edges, n)
    # triangle-free check
    for u in range(n):
        nbrs = adj[u]
        nbr_set = set(nbrs)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] in nbr_set and nbrs[i] in adj[nbrs[j]]:
                    # both neighbors of u, check if they are adjacent
                    if nbrs[j] in adj[nbrs[i]]:
                        return False
    # connected + diameter <= 4
    dist = _bfs_allpairs(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def _diameter_only(added_edges, n):
    adj = _build_adj(added_edges, n)
    dist = _bfs_allpairs(adj, n)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 75  # safety margin before 90s kill

    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u, v), max(u, v)))

    even_vertices = list(range(0, n, 2))
    odd_vertices = list(range(1, n, 2))

    def elapsed():
        return time.time() - start_time

    # Fallback: known-good single-hub construction (10 edges)
    def single_hub(hub):
        edges = []
        for v in odd_vertices:
            if v == hub - 1 or v == hub + 1:
                continue
            if (min(hub, v), max(hub, v)) not in path_edges:
                edges.append([hub, v])
        return edges

    best_solution = single_hub(12)
    best_count = len(best_solution)

    # Pattern A: two hubs, assign each odd vertex to nearest hub
    def two_hubs(h1, h2):
        edges = []
        seen = set()
        for o in odd_vertices:
            d1 = abs(o - h1)
            d2 = abs(o - h2)
            if d1 <= d2:
                h = h1
            else:
                h = h2
            if h == o - 1 or h == o + 1:
                continue
            key = (min(h, o), max(h, o))
            if key in path_edges or key in seen:
                continue
            seen.add(key)
            edges.append([h, o])
        return edges

    # Pattern B: three hubs
    def three_hubs(h1, h2, h3):
        edges = []
        seen = set()
        for o in odd_vertices:
            ds = sorted([(abs(o - h), h) for h in (h1, h2, h3)])
            h = ds[0][1]
            if h == o - 1 or h == o + 1:
                continue
            key = (min(h, o), max(h, o))
            if key in path_edges or key in seen:
                continue
            seen.add(key)
            edges.append([h, o])
        return edges

    def try_candidate(edges):
        nonlocal best_solution, best_count
        if not edges:
            return
        if len(edges) >= best_count:
            return
        if _check_solution(edges, n):
            best_solution = edges
            best_count = len(edges)

    # Search two-hub configurations
    for i in range(len(even_vertices)):
        if elapsed() > deadline:
            break
        for j in range(i + 1, len(even_vertices)):
            if elapsed() > deadline:
                break
            h1, h2 = even_vertices[i], even_vertices[j]
            edges = two_hubs(h1, h2)
            if len(edges) < best_count:
                try_candidate(edges)
            if best_count <= 5:
                break
        if best_count <= 5:
            break

    # Search three-hub configurations (smaller sets)
    for i in range(len(even_vertices)):
        if elapsed() > deadline:
            break
        for j in range(i + 1, len(even_vertices)):
            if elapsed() > deadline:
                break
            for k in range(j + 1, len(even_vertices)):
                if elapsed() > deadline:
                    break
                h1, h2, h3 = even_vertices[i], even_vertices[j], even_vertices[k]
                edges = three_hubs(h1, h2, h3)
                if len(edges) < best_count:
                    try_candidate(edges)
                if best_count <= 5:
                    break
            if best_count <= 5:
                break
        if best_count <= 5:
            break

    # Greedy local search: try to improve best_solution by swapping/removing
    # Start from best, try removing one edge at a time
    if best_count > 5 and elapsed() < deadline:
        cur = [list(e) for e in best_solution]
        improved = True
        while improved and elapsed() < deadline:
            improved = False
            for idx in range(len(cur)):
                if elapsed() > deadline:
                    break
                trial = cur[:idx] + cur[idx + 1:]
                if _check_solution(trial, n):
                    cur = trial
                    best_solution = cur
                    best_count = len(cur)
                    improved = True
                    break

    # Greedy add to fill gaps if current best is invalid (shouldn't happen)
    if not _check_solution(best_solution, n):
        best_solution = single_hub(12)
        best_count = len(best_solution)

    result = [[int(u), int(v)] for u, v in best_solution]
    return result
# EVOLVE-BLOCK-END
