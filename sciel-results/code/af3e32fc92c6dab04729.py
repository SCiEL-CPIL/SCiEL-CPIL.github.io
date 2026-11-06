import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def min_vertex_cover(edges, n, time_limit):
        if not edges:
            return []

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Greedy initialization: repeatedly pick vertex covering most uncovered edges
        def greedy_init():
            uncovered = set(edges)
            cover = []
            while uncovered:
                # Count how many uncovered edges each vertex covers
                best_v = -1
                best_count = -1
                for v in range(n):
                    count = 0
                    for u in adj[v]:
                        e = (min(u, v), max(u, v))
                        if e in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v == -1 or best_count == 0:
                    break
                cover.append(best_v)
                # Remove all edges incident to best_v
                new_uncovered = set()
                for e in uncovered:
                    if best_v not in e:
                        new_uncovered.add(e)
                uncovered = new_uncovered
            return cover

        cover = greedy_init()
        best_cover = list(cover)
        best_size = len(cover)

        # Local search: try to remove a vertex if still valid
        improved = True
        while improved:
            improved = False
            cover_set = set(cover)
            for v in list(cover):
                if len(cover) <= best_size - 1:
                    break
                cover_set.remove(v)
                valid = True
                for u, w in edges:
                    if u not in cover_set and w not in cover_set:
                        valid = False
                        break
                if valid:
                    cover = [x for x in cover if x != v]
                    cover_set = set(cover)
                    best_size = len(cover)
                    improved = True
                    break

        best_cover = list(cover)
        best_size = len(cover)

        # Branch and bound for polishing - use bitmask for speed
        # Represent edges as bitmask of vertices
        # For n up to 40, we can use Python integers as bitmasks
        
        def edge_to_mask(u, v):
            return (1 << u) | (1 << v)
        
        edge_masks = [edge_to_mask(u, v) for u, v in edges]
        
        # Precompute for each vertex, which edges it covers (as a bitmask of edge indices)
        # Actually, let's use a different approach: branch on vertices directly
        
        # State: remaining edges (as a list of (u,v)), current size, current cover
        # Use a stack-based BFS/DFS with bitmasks for edge sets
        
        # Let's use a recursive B&B with better pruning
        def bbs(remaining_edges, size, cover_list):
            nonlocal best_size, best_cover
            if time.time() > time_limit:
                return
            if not remaining_edges:
                if size < best_size:
                    best_size = size
                    best_cover = list(cover_list)
                return

            # Lower bound: greedy maximal matching
            remaining = list(remaining_edges)
            matching = 0
            used = set()
            for u, v in remaining:
                if u not in used and v not in used:
                    matching += 1
                    used.add(u)
                    used.add(v)
            if size + matching >= best_size:
                return

            # Pick edge with highest degree sum for branching
            degs = {}
            for u, v in remaining_edges:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            max_score = -1
            be = None
            for u, v in remaining_edges:
                s = degs.get(u, 0) + degs.get(v, 0)
                if s > max_score:
                    max_score = s
                    be = (u, v)
            if be is None:
                return
            u, v = be

            # Branch on u
            new1 = []
            for e in remaining_edges:
                if u not in e:
                    new1.append(e)
            bbs(new1, size + 1, cover_list + [u])

            # Branch on v
            new2 = []
            for e in remaining_edges:
                if v not in e:
                    new2.append(e)
            bbs(new2, size + 1, cover_list + [v])

        # Start B&B from current best
        if time.time() < time_limit - 5:
            # Find uncovered edges with current cover
            cover_set = set(best_cover)
            uncovered_edges = []
            for u, v in edges:
                if u not in cover_set and v not in cover_set:
                    uncovered_edges.append((u, v))
            if uncovered_edges:
                bbs(uncovered_edges, len(best_cover), list(best_cover))

        return sorted(best_cover)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append([])
            continue
        edges = instance(n, p, seed)
        cover = min_vertex_cover(edges, n, deadline)
        out.append(sorted(cover))

    return out
