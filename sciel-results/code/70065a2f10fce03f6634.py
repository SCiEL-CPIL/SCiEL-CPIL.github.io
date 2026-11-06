import time
import random
import sys
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def find_all_short_cycles(edges, max_len=7):
        """Find all simple cycles up to max_len using DFS."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        if not adj:
            return []

        cycles = []
        seen = set()
        nodes = sorted(adj.keys())

        def norm(cycle):
            # canonical representation: start at min vertex, try both directions
            if not cycle:
                return None
            m = min(cycle)
            idx = cycle.index(m)
            c1 = cycle[idx:] + cycle[:idx]
            c2 = [c1[0]] + list(reversed(c1[1:]))
            return tuple(min((c1, c2), key=tuple))

        # DFS from each node as root
        for root in nodes:
            visited = {root}
            path = [root]

            def dfs(current, depth):
                if len(cycles) > 5000:
                    return
                if depth >= max_len:
                    return
                for nb in sorted(adj[current]):
                    if nb == root and depth >= 2:
                        # found a cycle
                        key = norm(path)
                        if key and key not in seen:
                            seen.add(key)
                            cycles.append(list(key))
                    elif nb not in visited and nb > root:
                        # only visit nodes greater than root to avoid duplicates
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.remove(nb)

            dfs(root, 0)

        return cycles

    def decompose_ilp(edges, time_limit):
        """
        Formulate as ILP:
        Variables x_c for each candidate cycle c.
        Maximize sum_c x_c * (len(c) - 1)  (equivalent to minimizing total pieces)
        Subject to: for each edge e, sum_{c: e in c} x_c <= 1
                    x_c in {0, 1}
        """
        if not edges:
            return []

        edge_set = set(edges)
        n_edges = len(edges)

        # Find candidate cycles
        cycles = find_all_short_cycles(edges, max_len=8)

        if not cycles:
            return [list(e) for e in edges]

        # Also try slightly longer cycles if time permits
        if time.time() < time_limit - 2:
            longer = find_all_short_cycles(edges, max_len=10)
            # merge: keep longer cycles, dedupe
            existing = set(tuple(c) for c in cycles)
            for c in longer:
                t = tuple(c)
                if t not in existing:
                    cycles.append(c)
                    existing.add(t)

        n_cycles = len(cycles)
        if n_cycles == 0:
            return [list(e) for e in edges]

        # Build incidence: for each cycle, which edges does it use?
        cycle_edges = []
        valid = []
        for c in cycles:
            es = set()
            ok = True
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    ok = False
                    break
                es.add(e)
            if ok:
                cycle_edges.append(es)
                valid.append(True)
            else:
                valid.append(False)

        # Filter valid cycles
        valid_cycles = [c for c, v in zip(cycles, valid) if v]
        valid_edges = [ce for ce, v in zip(cycle_edges, valid) if v]
        n_cycles = len(valid_cycles)

        if n_cycles == 0:
            return [list(e) for e in edges]

        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        # Constraint matrix: A[i, j] = 1 if edge i is in cycle j
        A = np.zeros((n_edges, n_cycles), dtype=np.int8)
        for j, ces in enumerate(valid_edges):
            for e in ces:
                i = edge_to_idx[e]
                A[i, j] = 1

        # Objective: maximize sum x_j * (len(c_j) - 1) => minimize -sum
        c_obj = -np.array([len(c) - 1 for c in valid_cycles], dtype=float)

        constraints = LinearConstraint(A, ub=np.ones(n_edges))
        bounds = Bounds(0, 1)
        integrality = np.ones(n_cycles)

        try:
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        except Exception:
            return decompose_greedy(edges, time_limit)

        if res.status != 0 or res.x is None:
            return decompose_greedy(edges, time_limit)

        selected = []
        covered = set()
        for j in range(n_cycles):
            if res.x[j] > 0.5:
                selected.append(valid_cycles[j])
                for e in valid_edges[j]:
                    covered.add(e)

        pieces = list(selected)
        for e in edges:
            if e not in covered:
                pieces.append(list(e))

        return pieces

    def decompose_greedy(edges, time_limit):
        """Greedy: repeatedly find a long cycle, remove its edges."""
        remaining = set(edges)
        pieces = []

        while remaining and time.time() < time_limit:
            # Find a cycle in remaining
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)

            best = None
            # Try to find a reasonably long cycle via DFS from each node
            for start in sorted(adj.keys()):
                if time.time() > time_limit:
                    break
                visited = {start}
                path = [start]

                def dfs(cur, depth):
                    nonlocal best
                    if best is not None and len(path) <= len(best):
                        if depth >= 4:
                            return
                    if depth >= 10:
                        return
                    for nb in adj[cur]:
                        if nb == start and depth >= 2:
                            cand = list(path)
                            if best is None or len(cand) > len(best):
                                best = cand
                        elif nb not in visited and nb > start:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb, depth + 1)
                            path.pop()
                            visited.remove(nb)

                dfs(start, 0)
                if best is not None:
                    break  # found a cycle, use it

            if best:
                pieces.append(best)
                for i in range(len(best)):
                    u, v = best[i], best[(i + 1) % len(best)]
                    remaining.discard(tuple(sorted((u, v))))
            else:
                # No cycle: take one edge
                e = next(iter(remaining))
                pieces.append(list(e))
                remaining.discard(e)

        for e in remaining:
            pieces.append(list(e))

        return pieces

    def verify_and_fix(pieces, edges):
        """Verify pieces cover all edges exactly once; if not, fall back."""
        edge_set = set(edges)
        used = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used:
                    valid = False
                    break
                used.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used:
                        valid = False
                        break
                    used.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used:
                    valid = False
                    break
                used.add(e)
            else:
                valid = False
                break

        if not valid or used != edge_set:
            return [list(e) for e in edges]
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed
        if remaining_time < 2.0:
            results.append([list(e) for e in edges])
            continue

        graphs_left = len(graphs) - i
        per_graph = remaining_time / graphs_left
        t_limit = time.time() + min(per_graph * 0.85, 30.0)

        try:
            pieces = decompose_ilp(edges, t_limit)
        except Exception:
            pieces = decompose_greedy(edges, t_limit)

        pieces = verify_and_fix(pieces, edges)
        results.append(pieces)

    return results


if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
