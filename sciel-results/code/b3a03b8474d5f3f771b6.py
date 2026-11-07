"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, base_edges, added_edges):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_bipartite(n, adj):
    # Check bipartite (triangle-free for this structure since base is a path)
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, None
    return True, color

def _diameter(n, adj):
    # BFS from each node, find max shortest path
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
    return max_dist

def _is_valid(n, base_edges, added_edges):
    adj = _build_adj(n, base_edges, added_edges)
    # Check bipartite (triangle-free)
    bip, color = _is_bipartite(n, adj)
    if not bip:
        return False
    # Check diameter <= 4
    d = _diameter(n, adj)
    if d < 0 or d > 4:
        return False
    return True

def _get_candidate_edges(n, base_edges):
    # Only edges between opposite parity vertices (bipartite constraint)
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u,v), max(u,v)))
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 == 1:  # opposite parity
                if (u, v) not in base_set:
                    candidates.append((u, v))
    return candidates

def _try_k_edges(n, base_edges, candidates, k, deadline):
    """Try to find k edges that make the graph valid."""
    # For small k, do combinatorial search
    # But C(len(candidates), k) can be huge, so use pruning
    # Strategy: try combinations but with early termination
    
    best = None
    
    if k == 1:
        for edge in candidates:
            if time.time() > deadline:
                break
            if _is_valid(n, base_edges, [edge]):
                best = [edge]
                break
    elif k == 2:
        for i in range(len(candidates)):
            if time.time() > deadline:
                return best
            e1 = candidates[i]
            for j in range(i+1, len(candidates)):
                e2 = candidates[j]
                if _is_valid(n, base_edges, [e1, e2]):
                    best = [e1, e2]
                    return best
    elif k == 3:
        # Smarter: pre-filter edges that help with diameter
        # Focus on edges that create shortcuts
        # Use a reduced candidate set: edges that span large distances
        # Sort by distance descending, take top ~40
        sorted_cands = sorted(candidates, key=lambda e: e[1]-e[0], reverse=True)
        top = sorted_cands[:50]
        for i in range(len(top)):
            if time.time() > deadline:
                return best
            e1 = top[i]
            for j in range(i+1, len(top)):
                if time.time() > deadline:
                    return best
                e2 = top[j]
                for l in range(j+1, len(top)):
                    e3 = top[l]
                    if _is_valid(n, base_edges, [e1, e2, e3]):
                        best = [e1, e2, e3]
                        return best
    elif k == 4:
        # Even smarter: use a much smaller candidate set
        sorted_cands = sorted(candidates, key=lambda e: e[1]-e[0], reverse=True)
        top = sorted_cands[:35]
        for i in range(len(top)):
            if time.time() > deadline:
                return best
            e1 = top[i]
            for j in range(i+1, len(top)):
                if time.time() > deadline:
                    return best
                e2 = top[j]
                for l in range(j+1, len(top)):
                    if time.time() > deadline:
                        return best
                    e3 = top[l]
                    for m in range(l+1, len(top)):
                        e4 = top[m]
                        if _is_valid(n, base_edges, [e1, e2, e3, e4]):
                            best = [e1, e2, e3, e4]
                            return best
    return best

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80  # safety margin
    
    candidates = _get_candidate_edges(n, base_edges)
    
    # Try increasing k from 1 to 5
    for k in range(1, 6):
        if time.time() > deadline:
            break
        result = _try_k_edges(n, base_edges, candidates, k, deadline)
        if result is not None:
            # Return as list of [u, v] pairs
            return [[int(u), int(v)] for u, v in result]
    
    # Fallback: use the original bipartite complete solution
    base_set = {tuple(e) for e in P['edges']}
    return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v) % 2 == 1 and (u, v) not in base_set]
# EVOLVE-BLOCK-END
