import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    def build_adj_deg(edge_list):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def has_violation(adj, deg):
        """Return True if any same-degree pair has a 3-edge path between them."""
        deg_map = defaultdict(list)
        for i in range(n):
            deg_map[deg[i]].append(i)
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                return True
        return False

    def is_valid(edge_list):
        adj, deg = build_adj_deg(edge_list)
        return not has_violation(adj, deg)

    # Base: complete bipartite K_{7,8} - always valid
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)

    # All possible edges not in base
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    base_set = set(base_edges)
    remaining = [e for e in all_possible if e not in base_set]

    # Greedy add with multiple random orderings
    for attempt in range(500):
        if time.time() > deadline - 10:
            break
        current = base_edges[:]
        current_set = set(current)
        shuffled = remaining[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            if edge in current_set:
                continue
            test = current + [edge]
            if is_valid(test):
                current = test
                current_set.add(edge)
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current[:]
            # If we've reached a high count, try more attempts
            if best_count >= 60:
                # Do more attempts to squeeze out extra edges
                for extra in range(200):
                    if time.time() > deadline - 5:
                        break
                    cur2 = best_edges[:]
                    cur2_set = set(cur2)
                    rem2 = [e for e in all_possible if e not in cur2_set]
                    random.shuffle(rem2)
                    for edge in rem2:
                        test = cur2 + [edge]
                        if is_valid(test):
                            cur2 = test
                            cur2_set.add(edge)
                    if len(cur2) > best_count:
                        best_count = len(cur2)
                        best_edges = cur2[:]

    out = [[u, v] for u, v in best_edges]
    return out
