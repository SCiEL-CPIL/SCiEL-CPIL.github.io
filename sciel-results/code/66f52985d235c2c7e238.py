import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin (limit 90s)
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Bipartite strategy: even vertices = {0,2,4,...,22}, odd vertices = {1,3,...,23}
    # Path edges connect even-odd, so all base edges are cross-parity.
    # Only add cross-parity edges => triangle-free guaranteed.
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # All possible cross-parity non-base edges
    all_cross = []
    for u in even_verts:
        for v in odd_verts:
            key = (min(u,v), max(u,v))
            if key not in base_edges:
                all_cross.append((u, v))
    
    def make_adj(added):
        adj = [set() for _ in range(n)]
        for u,v in base_edges:
            adj[u].add(v); adj[v].add(u)
        for u,v in added:
            adj[u].add(v); adj[v].add(u)
        return adj
    
    def diameter_le_4(adj):
        for s in range(n):
            dist = [-1]*n
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                if dist[node] == 4:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node]+1
                        q.append(nb)
            for d in dist:
                if d == -1:
                    return False
        return True
    
    def full_diameter(adj):
        maxd = 0
        for s in range(n):
            dist = [-1]*n
            dist[s]=0
            q=deque([s])
            while q:
                node=q.popleft()
                for nb in adj[node]:
                    if dist[nb]==-1:
                        dist[nb]=dist[node]+1
                        q.append(nb)
            for d in dist:
                if d>maxd: maxd=d
        return maxd
    
    def greedy_bipartite():
        adj = make_adj([])
        added = []
        added_set = set()
        while True:
            if time.time() > deadline:
                break
            if diameter_le_4(adj):
                break
            # Pick edge that most reduces diameter
            best = None
            best_diam = full_diameter(adj) + 1
            for u,v in all_cross:
                if (u,v) in added_set: continue
                adj[u].add(v); adj[v].add(u)
                d = full_diameter(adj)
                adj[u].discard(v); adj[v].discard(u)
                if d < best_diam:
                    best_diam = d
                    best = (u,v)
            if best is None:
                break
            u,v = best
            adj[u].add(v); adj[v].add(u)
            added_set.add((u,v))
            added.append((u,v))
        return added
    
    def local_search(added, time_limit):
        added_set = set(added)
        adj = make_adj(added)
        # Iterative removal
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for e in list(added_set):
                u,v = e
                adj[u].discard(v); adj[v].discard(u)
                if diameter_le_4(adj):
                    added_set.discard(e)
                    changed = True
                else:
                    adj[u].add(v); adj[v].add(u)
        # Try swaps: remove one, add another
        if time.time() < time_limit:
            for e_remove in list(added_set):
                if time.time() > time_limit:
                    break
                u1,v1 = e_remove
                adj[u1].discard(v1); adj[v1].discard(u1)
                added_set.discard(e_remove)
                # Try adding a new edge
                best_swap = None
                best_diam = full_diameter(adj) + 1
                for u2,v2 in all_cross:
                    if (u2,v2) in added_set: continue
                    adj[u2].add(v2); adj[v2].add(u2)
                    d = full_diameter(adj)
                    adj[u2].discard(v2); adj[v2].discard(u2)
                    if d < best_diam:
                        best_diam = d
                        best_swap = (u2,v2)
                if best_swap is not None and best_diam <= 4:
                    u2,v2 = best_swap
                    adj[u2].add(v2); adj[v2].add(u2)
                    added_set.add((u2,v2))
                else:
                    # revert removal
                    adj[u1].add(v1); adj[v1].add(u1)
                    added_set.add(e_remove)
        return list(added_set)
    
    # Run greedy
    added = greedy_bipartite()
    
    # Verify feasibility
    adj = make_adj(added)
    if not diameter_le_4(adj):
        # Fallback: try more aggressive greedy
        added = []
        added_set = set()
        adj = make_adj([])
        while not diameter_le_4(adj) and time.time() < deadline:
            best = None
            best_diam = full_diameter(adj) + 1
            for u,v in all_cross:
                if (u,v) in added_set: continue
                adj[u].add(v); adj[v].add(u)
                d = full_diameter(adj)
                adj[u].discard(v); adj[v].discard(u)
                if d < best_diam:
                    best_diam = d
                    best = (u,v)
            if best is None:
                break
            u,v = best
            adj[u].add(v); adj[v].add(u)
            added_set.add((u,v))
            added.append((u,v))
    
    # Local search to minimize
    if time.time() < deadline - 5:
        added = local_search(added, deadline - 5)
    
    # Final verification
    adj = make_adj(added)
    if not diameter_le_4(adj):
        # Emergency: just use greedy result
        added = greedy_bipartite()
        adj = make_adj(added)
    
    return [[int(u), int(v)] for u,v in added]
