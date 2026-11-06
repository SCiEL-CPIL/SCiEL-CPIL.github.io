"""Decomposition into few cycles and edges via greedy init + local search."""
import time
import random
from collections import defaultdict, deque

def construct():
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_decompose(edges):
        """Greedy: repeatedly find longest cycle via DFS, remove its edges."""
        if not edges:
            return []
        remaining = set(tuple(sorted(e)) for e in edges)
        pieces = []
        while remaining:
            if time.time() > DEADLINE:
                break
            adj = build_adj(remaining)
            # Find longest cycle via DFS
            best_cycle = None
            best_len = 0
            for start in sorted(adj.keys()):
                # Try DFS to find a cycle back to start
                visited = set()
                path = []
                stack = [(start, iter(sorted(adj[start])))]
                visited.add(start)
                path.append(start)
                while stack:
                    node, it = stack[-1]
                    found = False
                    for nb in it:
                        if nb == start and len(path) >= 3:
                            # Found a cycle
                            cycle = path[:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                            found = True
                            break
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(sorted(adj[nb]))))
                            found = True
                            break
                    if not found:
                        stack.pop()
                        path.pop()
                        visited.discard(node)
                if best_len >= 3:
                    break
            if best_cycle is None or len(best_cycle) < 3:
                # No cycle found; take all remaining edges as single-edge pieces
                for e in remaining:
                    pieces.append(list(e))
                remaining = set()
                break
            # Remove cycle edges from remaining
            cycle_edges = set()
            for i in range(len(best_cycle)):
                u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                cycle_edges.add(tuple(sorted((u, v))))
            pieces.append(best_cycle)
            remaining -= cycle_edges
        return pieces

    def pieces_to_edge_set(pieces):
        es = set()
        for p in pieces:
            if len(p) == 2:
                es.add(tuple(sorted(p)))
            else:
                for i in range(len(p)):
                    u, v = p[i], p[(i + 1) % len(p)]
                    es.add(tuple(sorted((u, v))))
        return es

    def local_search(pieces, edges_set, max_iters=20000):
        """Try to merge two pieces into one cycle, or split to reduce count."""
        pieces = [list(p) for p in pieces]
        # Precompute edge sets for each piece
        def piece_edges(p):
            if len(p) == 2:
                return {tuple(sorted(p))}
            return {tuple(sorted((p[i], p[(i + 1) % len(p)]))) for i in range(len(p))}

        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try merging two pieces
            n_p = len(pieces)
            for i in range(n_p):
                if time.time() > DEADLINE:
                    return pieces
                for j in range(i + 1, n_p):
                    pi, pj = pieces[i], pieces[j]
                    ei = piece_edges(pi)
                    ej = piece_edges(pj)
                    if ei & ej:
                        continue
                    # Combined edge set
                    combined = ei | ej
                    # Check if combined edges form a single cycle
                    adj = defaultdict(set)
                    for u, v in combined:
                        adj[u].add(v)
                        adj[v].add(u)
                    # All degrees must be 2
                    if all(len(adj[u]) == 2 for u in adj):
                        # Find the cycle
                        start = next(iter(adj))
                        cycle = [start]
                        prev = None
                        cur = start
                        while True:
                            nxts = [x for x in adj[cur] if x != prev]
                            if not nxts:
                                break
                            nxt = nxts[0]
                            if nxt == start and len(cycle) >= 3:
                                break
                            if nxt in cycle:
                                break
                            prev = cur
                            cur = nxt
                            cycle.append(nxt)
                        if len(cycle) >= 3:
                            # Verify it uses all edges
                            cycle_e = set()
                            for k in range(len(cycle)):
                                u, v = cycle[k], cycle[(k + 1) % len(cycle)]
                                cycle_e.add(tuple(sorted((u, v))))
                            if cycle_e == combined:
                                pieces[i] = cycle
                                pieces[j] = []
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                # Try splitting a cycle into a smaller cycle + edges (reduces count if we can merge edges elsewhere)
                # For now, just stop if no merges possible
                break
            pieces = [p for p in pieces if p]
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            results.append([])
            continue
        edges = instance(*spec)
        edges_set = set(tuple(sorted(e)) for e in edges)
        pieces = greedy_decompose(edges)
        # Verify coverage
        covered = pieces_to_edge_set(pieces)
        if covered != edges_set:
            # Fallback: all edges as single pieces
            pieces = [list(e) for e in edges]
        else:
            # Local search refinement
            pieces = local_search(pieces, edges_set)
            # Final verification
            covered = pieces_to_edge_set(pieces)
            if covered != edges_set:
                pieces = [list(e) for e in edges]
        results.append(pieces)

    return results
