import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    n = 40
    m = 401
    rng = random.Random(12345)

    def neighbors_of(e, adj):
        a, b = e
        na = adj[a]
        nb = adj[b]
        common = [v for v in na if v in nb]
        return common

    def max_book(adj, edge_set):
        best = 0
        for (a, b) in edge_set:
            c = 0
            na = adj[a]
            nb = adj[b]
            # count common neighbors
            for v in na:
                if v in nb:
                    c += 1
            if c > best:
                best = c
        return best

    def init_graph():
        # Start from K20,20 + one intra edge (the seed), then randomize slightly
        adj = [set() for _ in range(n)]
        edge_set = set()
        for a in range(20):
            for b in range(20, 40):
                adj[a].add(b)
                adj[b].add(a)
                edge_set.add((a, b))
        # add one intra edge
        adj[0].add(1)
        adj[1].add(0)
        edge_set.add((0, 1))
        return adj, edge_set

    adj, edge_set = init_graph()
    best_score = max_book(adj, edge_set)
    best_edge_set = set(edge_set)
    best_adj = [set(s) for s in adj]

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def do_swap(remove_e, add_e):
        # remove edge remove_e, add add_e; returns True if valid (no dup)
        if add_e in edge_set:
            return False
        a, b = remove_e
        c, d = add_e
        if a == c and b == d:
            return False
        if a == d and b == c:
            return False
        # remove
        adj[a].discard(b)
        adj[b].discard(a)
        edge_set.discard(remove_e)
        # add
        adj[c].add(d)
        adj[d].add(c)
        edge_set.add(add_e)
        return True

    def undo_swap(remove_e, add_e):
        a, b = remove_e
        c, d = add_e
        # undo add
        adj[c].discard(d)
        adj[d].discard(c)
        edge_set.discard(add_e)
        # undo remove
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add(remove_e)

    # Local search: try to reduce the max book
    improved = True
    stall_rounds = 0
    while time.time() < deadline and stall_rounds < 200:
        # find an edge with the maximum book
        maxb = 0
        max_edge = None
        for e in edge_set:
            a, b = e
            c = 0
            na = adj[a]
            nb = adj[b]
            for v in na:
                if v in nb:
                    c += 1
            if c > maxb:
                maxb = c
                max_edge = e

        if maxb <= best_score and max_edge is None:
            break

        # Try to improve by removing an edge involved in many triangles and adding a "sparse" edge
        # Candidate removal edges: edges with book >= maxb
        rem_candidates = []
        for e in edge_set:
            a, b = e
            c = 0
            na = adj[a]
            nb = adj[b]
            for v in na:
                if v in nb:
                    c += 1
            if c >= maxb:
                rem_candidates.append((c, e))
        rem_candidates.sort(reverse=True)

        # Candidate add edges: non-edges with low potential book
        add_candidates = []
        for e in all_edges:
            if e in edge_set:
                continue
            a, b = e
            c = 0
            na = adj[a]
            nb = adj[b]
            for v in na:
                if v in nb:
                    c += 1
            add_candidates.append((c, e))
        add_candidates.sort()

        found_improve = False
        for _, rem_e in rem_candidates[:50]:
            if time.time() > deadline:
                break
            for _, add_e in add_candidates[:200]:
                if not do_swap(rem_e, add_e):
                    continue
                # quick score: recompute max book
                mb = 0
                for e in edge_set:
                    a, b = e
                    c = 0
                    na = adj[a]
                    nb = adj[b]
                    for v in na:
                        if v in nb:
                            c += 1
                    if c > mb:
                        mb = c
                if mb < best_score:
                    best_score = mb
                    best_edge_set = set(edge_set)
                    best_adj = [set(s) for s in adj]
                    found_improve = True
                else:
                    undo_swap(rem_e, add_e)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

        if found_improve:
            stall_rounds = 0
        else:
            stall_rounds += 1

    # Return best found
    result = []
    for (a, b) in best_edge_set:
        result.append([a, b])
    # Ensure exactly 401
    if len(result) != m:
        # fallback to seed
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    return result
