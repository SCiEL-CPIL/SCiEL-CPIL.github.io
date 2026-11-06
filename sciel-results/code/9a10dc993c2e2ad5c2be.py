"""Fresh design: greedy cycle peeling + merge heuristic + optional ILP refinement."""
import time
import sys
import random
import itertools

_T0 = time.time()
_DEADLINE = _T0 + 118.0  # safety margin before the 150s kill

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def find_cycles(edges, n, budget):
        """Iteratively extract vertex-disjoint cycles by DFS until none remain or budget hit."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        used_edges = set()
        used_vertices = set()
        cycles = []

        def find_one(start):
            # DFS for a cycle that returns to start without reusing vertices/edges
            stack = [(start, [start])]
            visited = {start}
            while stack:
                node, path = stack.pop()
                for nb in sorted(adj[node], reverse=True):
                    if nb in used_vertices and nb != start:
                        continue
                    if nb == start and len(path) >= 3:
                        # check all edges on path are unused
                        ok = True
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i + 1]
                            if (min(a, b), max(a, b)) in used_edges:
                                ok = False
                                break
                        if ok:
                            return path
                        continue
                    if nb in visited:
                        continue
                    if (min(node, nb), max(node, nb)) in used_edges:
                        continue
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
            return None

        for start in range(n):
            if time.time() > budget:
                break
            if start in used_vertices:
                continue
            c = find_one(start)
            if c is None:
                continue
            cycles.append(c)
            for i in range(len(c) - 1):
                a, b = c[i], c[i + 1]
                used_edges.add((min(a, b), max(a, b)))
            a, b = c[-1], c[0]
            used_edges.add((min(a, b), max(a, b)))
            for v in c:
                used_vertices.add(v)

        return cycles, used_edges, used_vertices

    def merge_pass(cycles, n, budget):
        """Try to merge two cycles sharing a vertex into one cycle (reduces count)."""
        from collections import defaultdict
        changed = True
        cycles = [list(c) for c in cycles]
        while changed and time.time() < budget:
            changed = False
            # map vertex -> list of cycle indices containing it
            vmap = defaultdict(list)
            for i, c in enumerate(cycles):
                for v in c:
                    vmap[v].append(i)
            # find a vertex shared by exactly two cycles
            merged = False
            for v, idxs in vmap.items():
                if time.time() > budget:
                    break
                if len(idxs) != 2:
                    continue
                i, j = idxs
                ci, cj = cycles[i], cycles[j]
                if len(ci) < 3 or len(cj) < 3:
                    continue
                # split each cycle at v into two arcs
                # find positions of v
                pos_i = ci.index(v)
                pos_j = cj.index(v)
                # arc A: from v going one way back to v; arc B the other way
                # represent arcs as sequences of vertices starting and ending at v
                def split(c, pos):
                    L = len(c)
                    # arc1: v, c[pos+1], ..., c[pos-1], v  (forward)
                    arc1 = [c[(pos + k) % L] for k in range(L + 1)]
                    # arc2: v, c[pos-1], ..., c[pos+1], v (backward)
                    arc2 = [c[(pos - k) % L] for k in range(L + 1)]
                    return arc1, arc2
                a1, a2 = split(ci, pos_i)
                b1, b2 = split(cj, pos_j)
                # combine a1 + b1 (drop the shared middle v)
                def combine(x, y):
                    return x[:-1] + y
                cand1 = combine(a1, b1)
                cand2 = combine(a2, b2)
                # validate: all vertices distinct except first==last
                def valid(c):
                    if len(c) < 4:
                        return False
                    if c[0] != c[-1]:
                        return False
                    body = c[1:-1]
                    return len(set(body)) == len(body)
                newc = None
                if valid(cand1):
                    newc = cand1
                elif valid(cand2):
                    newc = cand2
                if newc is None:
                    continue
                # replace cycles i and j with newc
                new_cycles = []
                for k, cc in enumerate(cycles):
                    if k in (i, j):
                        continue
                    new_cycles.append(cc)
                new_cycles.append(newc)
                cycles = new_cycles
                merged = True
                changed = True
                break
            if merged:
                continue
        return cycles

    def solve_one(n, p, seed, budget):
        edges = instance(n, p, seed)
        edge_set = set(tuple(sorted(e)) for e in edges)
        # greedy cycle extraction
        cycles, used_edges, used_vertices = find_cycles(edge_set, n, budget)
        # merge pass
        cycles = merge_pass(cycles, n, budget)
        # remaining edges
        remaining = [e for e in edge_set if e not in used_edges]
        pieces = [list(c) for c in cycles] + [list(e) for e in remaining]
        # try ILP refinement if time permits
        if time.time() < budget - 5:
            try:
                refined = ilp_refine(edge_set, pieces, n, budget - 5)
                if refined is not None:
                    pieces = refined
            except Exception:
                pass
        return pieces

    def ilp_refine(edge_set, initial_pieces, n, budget):
        """Try PuLP ILP to minimize number of pieces over all cycles+edges. Fallback to initial."""
        try:
            import pulp
        except Exception:
            return None
        if time.time() > budget:
            return None
        # enumerate candidate cycles (all simple cycles up to length limit) - can be large;
        # restrict to cycles found + short cycles. For safety, only use initial cycles + edges.
        # Build variables: x_e for each edge (must be covered exactly once),
        # and y_c for each candidate cycle.
        # Constraints: for each edge e, sum_{c: e in c} y_c + x_e = 1
        # Minimize sum y_c + sum x_e
        cycles_list = [tuple(c) for c in initial_pieces if len(c) >= 3]
        edges_list = [tuple(e) for e in initial_pieces if len(e) == 2]
        # Also add a few extra candidate cycles from the graph (3-cycles)
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[u]:
                        if w != v and w in adj[v] and u < v < w:
                            cyc = (u, v, w)
                            if cyc not in cycles_list:
                                cycles_list.append(cyc)
        # build model
        prob = pulp.LpProblem("decomp", pulp.LpMinimize)
        y = {c: pulp.LpVariable(f"y_{i}", cat="Binary") for i, c in enumerate(cycles_list)}
        x = {e: pulp.LpVariable(f"x_{i}", cat="Binary") for i, e in enumerate(edges_list)}
        # objective
        prob += pulp.lpSum(y.values()) + pulp.lpSum(x.values())
        # edge coverage
        from collections import defaultdict
        edge_to_cycles = defaultdict(list)
        for ci, c in enumerate(cycles_list):
            for i in range(len(c)):
                a, b = c[i], c[(i + 1) % len(c)]
                e = (min(a, b), max(a, b))
                edge_to_cycles[e].append(ci)
        for ei, e in enumerate(edges_list):
            terms = [y[ci] for ci in edge_to_cycles.get(e, [])]
            terms.append(x[ei])
            prob += pulp.lpSum(terms) == 1
        # solve with time limit
        import time as _t
        t0 = _t.time()
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(budget - (_t.time() - t0) - 1)))
        except Exception:
            return None
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
        # extract
        result = []
        for ci, c in enumerate(cycles_list):
            if y[ci].value() and y[ci].value() > 0.5:
                result.append(list(c))
        for ei, e in enumerate(edges_list):
            if x[ei].value() and x[ei].value() > 0.5:
                result.append(list(e))
        # validate coverage
        used = set()
        for p in result:
            if len(p) == 2:
                e = (min(p[0], p[1]), max(p[0], p[1]))
                if e in used:
                    return None
                used.add(e)
            else:
                for i in range(len(p)):
                    a, b = p[i], p[(i + 1) % len(p)]
                    e = (min(a, b), max(a, b))
                    if e in used:
                        return None
                    used.add(e)
        if used != edge_set:
            return None
        return result

    results = []
    for spec in specs:
        if time.time() > _DEADLINE - 3:
            # not enough time; fall back to trivial for remaining
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        # allocate remaining budget split among remaining specs
        remaining_specs = specs[specs.index(spec):]
        per = max(5.0, (_DEADLINE - 3 - time.time()) / len(remaining_specs))
        pieces = solve_one(spec[0], spec[1], spec[2], time.time() + per)
        results.append(pieces)
    return results
