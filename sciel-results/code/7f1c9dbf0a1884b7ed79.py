"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges):
    """Decompose edges into cycles and single edges, preferring longer cycles."""
    if not edges:
        return []

    edges_set = set(tuple(sorted(e)) for e in edges)
    pieces = []

    def build_adj(eset):
        adj = {}
        for u, v in eset:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    adj = build_adj(edges_set)

    def find_longest_cycle(adj, eset):
        """Find a simple cycle, preferring longer ones via multiple starts."""
        if not adj:
            return None
        n = max(adj.keys()) + 1
        best = None
        # Try each vertex as start, keep the longest cycle found
        for start in range(n):
            if start not in adj or not adj[start]:
                continue
            stack = [(start, [start])]
            visited_in_path = {start}
            while stack:
                node, path = stack.pop()
                for neighbor in sorted(adj.get(node, [])):
                    edge = tuple(sorted((node, neighbor)))
                    if edge not in eset:
                        continue
                    if neighbor == start and len(path) >= 3:
                        cand = path + [start]
                        if best is None or len(cand) > len(best):
                            best = cand
                        break  # don't extend past closing
                    if neighbor not in visited_in_path:
                        visited_in_path.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))
        return best

    def remove_cycle(cycle, adj, eset):
        for i in range(len(cycle) - 1):
            edge = tuple(sorted((cycle[i], cycle[i+1])))
            eset.discard(edge)
            if edge[0] in adj and edge[1] in adj[edge[0]]:
                adj[edge[0]].discard(edge[1])
            if edge[1] in adj and edge[0] in adj[edge[1]]:
                adj[edge[1]].discard(edge[0])
            for v in (edge[0], edge[1]):
                if v in adj and not adj[v]:
                    del adj[v]

    max_iter = len(edges_set) * 3
    for _ in range(max_iter):
        if not edges_set:
            break
        cycle = find_longest_cycle(adj, edges_set)
        if cycle is None:
            break
        remove_cycle(cycle, adj, edges_set)
        pieces.append(cycle)

    # Remaining edges as single pieces
    singles = [list(e) for e in sorted(edges_set)]
    pieces.extend(singles)

    # Local improvement: try to form cycles from remaining single edges
    # Build adjacency of the leftover single edges
    if len(singles) >= 2:
        single_edges = set(tuple(sorted(e)) for e in singles)
        s_adj = {}
        for u, v in single_edges:
            s_adj.setdefault(u, set()).add(v)
            s_adj.setdefault(v, set()).add(u)

        # Try to find cycles among singles and replace the singles with cycles
        new_pieces = []
        # Separate: pieces that are cycles (len>=3) vs singles (len==2)
        cycle_pieces = [p for p in pieces if len(p) >= 3]
        # Re-solve singles into cycles + leftover singles
        leftover = set(single_edges)
        # Reuse find_longest_cycle on the single-edge subgraph
        def find_cycle_in(eset, sadj):
            if not sadj:
                return None
            n = max(sadj.keys()) + 1
            best = None
            for start in range(n):
                if start not in sadj or not sadj[start]:
                    continue
                stack = [(start, [start])]
                visited_in_path = {start}
                while stack:
                    node, path = stack.pop()
                    for neighbor in sorted(sadj.get(node, [])):
                        edge = tuple(sorted((node, neighbor)))
                        if edge not in eset:
                            continue
                        if neighbor == start and len(path) >= 3:
                            cand = path + [start]
                            if best is None or len(cand) > len(best):
                                best = cand
                            break
                        if neighbor not in visited_in_path:
                            visited_in_path.add(neighbor)
                            stack.append((neighbor, path + [neighbor]))
            return best

        for _ in range(len(leftover) * 2):
            if not leftover:
                break
            c = find_cycle_in(leftover, s_adj)
            if c is None:
                break
            for i in range(len(c) - 1):
                edge = tuple(sorted((c[i], c[i+1])))
                leftover.discard(edge)
                if edge[0] in s_adj and edge[1] in s_adj[edge[0]]:
                    s_adj[edge[0]].discard(edge[1])
                if edge[1] in s_adj and edge[0] in s_adj[edge[1]]:
                    s_adj[edge[1]].discard(edge[0])
                for v in (edge[0], edge[1]):
                    if v in s_adj and not s_adj[v]:
                        del s_adj[v]
            new_pieces.append(c)

        for e in sorted(leftover):
            new_pieces.append([e[0], e[1]])

        pieces = cycle_pieces + new_pieces

    return pieces

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
