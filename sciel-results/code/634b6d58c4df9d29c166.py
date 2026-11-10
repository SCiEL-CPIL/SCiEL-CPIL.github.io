"""Rewritten program for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all edges
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105

    def build_adj_deg(adj_sets, deg):
        # adj_sets: list of sets; deg: list
        return adj_sets, deg

    def is_valid(adj, deg):
        """Check: for every edge (u,v), for s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]!=deg[t]."""
        for (u, v) in edge_list:
            if v not in adj[u]:
                continue
            ns = adj[u] - {v}
            nt = adj[v] - {u}
            if not ns or not nt:
                continue
            # Map degree -> set of vertices in ns
            deg_to_s = {}
            for s in ns:
                deg_to_s.setdefault(deg[s], set()).add(s)
            for t in nt:
                dt = deg[t]
                if dt in deg_to_s:
                    # Need s != t for violation
                    if len(deg_to_s[dt]) >= 2 or (t in deg_to_s[dt] and len(deg_to_s[dt]) >= 1 and t != t):
                        pass
                    # violation if there exists s in deg_to_s[dt] with s != t
                    if len(deg_to_s[dt]) > 1:
                        return False
                    elif len(deg_to_s[dt]) == 1:
                        s_only = next(iter(deg_to_s[dt]))
                        if s_only != t:
                            return False
        return True

    def make_adj_deg(mask):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return adj, deg

    def valid_mask(mask):
        adj, deg = make_adj_deg(mask)
        return is_valid(adj, deg)

    # Build a threshold graph with distinct degrees (max edges)
    # Threshold graph: order vertices 0..n-1, vertex i connected to all j<i if "dominating"
    # Use a construction: assign weights, connect if sum exceeds threshold.
    # Simpler: build a graph where degrees are all distinct.
    # Construction: start with empty, add vertices one by one, each new vertex connects
    # to a chosen subset of existing vertices to give it a unique degree.

    # Let's try: vertex i (0-indexed) gets degree d_i. We want distinct d_i and many edges.
    # A known dense graph with distinct degrees: the "almost complete" threshold graph.
    # Threshold graph from a binary string of length n: 1=dominating, 0=isolated (at addition).
    # Degrees are distinct if the string is chosen well.

    # Let me just search: try the threshold graph defined by a bitmask of dominating/isolated.
    # For a threshold graph, the degree sequence is determined by the "threshold sequence".
    # I'll enumerate a few promising threshold sequences and pick the densest valid one.

    best_mask = 0
    best_count = 0

    # Try K_{7,8} as a strong baseline
    A = set(range(7))
    B = set(range(7, 15))
    km_mask = 0
    for a in A:
        for b in B:
            km_mask |= (1 << edge_to_idx[(a, b)])
    if valid_mask(km_mask):
        best_mask = km_mask
        best_count = 56

    # Try to build a threshold graph with distinct degrees
    # Threshold graph: process vertices in order; each is either isolated (0) or dominating (1)
    # The resulting graph is a threshold graph. Degrees are distinct iff no two vertices have same degree.
    # Let's try all 2^15 threshold sequences? Too many (32768) but each check is fast.
    # Actually 2^15 = 32768, each validity check ~ O(E * avg_deg^2) which is fine.
    # But we want max edges, so try sequences with many 1s.

    # Generate threshold graph from a sequence of 0/1 (length n)
    def threshold_graph(seq):
        """seq[i] = 1 if vertex i (added in order 0..n-1) is dominating, 0 if isolated."""
        adj = [set() for _ in range(n)]
        # When vertex i is added:
        #   if dominating (1): connects to all previously added vertices
        #   if isolated (0): connects to none
        for i in range(n):
            if seq[i] == 1:
                for j in range(i):
                    adj[i].add(j)
                    adj[j].add(i)
        mask = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    mask |= (1 << edge_to_idx[(i, j)])
        return mask

    # Try threshold sequences with many dominating vertices
    # The number of edges in a threshold graph = sum over i of (if seq[i]==1 then i else 0)
    # To maximize edges, we want many 1s, especially late ones.
    # But we need distinct degrees for automatic validity, OR just valid.
    # Let's try sequences and check validity, track max edges.

    # Try the all-dominating (complete graph) - invalid, but try near-complete
    # Try sequences: all 1s except a few 0s
    rng = random.Random(12345)

    # First, try systematic: sequences with k zeros for k=0..6
    for k in range(0, 7):
        if time.time() > deadline - 10:
            break
        # Choose which k positions are 0
        # To maximize edges, put zeros early (so late vertices are dominating)
        # Actually zeros early reduce edges less? A 0 at position i means vertex i is isolated,
        # losing i potential edges (connections to earlier). A 0 at position i also means
        # later dominating vertices don't connect to it, losing (n-1-i) edges.
        # Total edges lost by a 0 at position i: i (not connected to earlier) + (n-1-i) (later don't connect) = n-1.
        # So every 0 loses exactly n-1 = 14 edges regardless of position! Interesting.
        # So all sequences with same number of zeros have same edge count.
        # Edges = C(n,2) - k*(n-1) = 105 - 14k.
        # For k=0: 105 (complete, invalid). k=1: 91. k=2: 77. k=3: 63. k=4: 49.
        # We need to find which are valid.
        # Let's try k=1,2,3 and check validity for all combinations.
        from itertools import combinations
        for zeros in combinations(range(n), k):
            seq = [1] * n
            for z in zeros:
                seq[z] = 0
            mask = threshold_graph(seq)
            cnt = bin(mask).count('1')
            if cnt > best_count and valid_mask(mask):
                best_mask = mask
                best_count = cnt

    # Now local search: try adding edges
    def try_add_edges(mask, count):
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            changed = False
            for i in range(E):
                if mask & (1 << i):
                    continue
                new_mask = mask | (1 << i)
                if valid_mask(new_mask):
                    mask = new_mask
                    count += 1
                    changed = True
            if not changed:
                break
        return mask, count

    best_mask, best_count = try_add_edges(best_mask, best_count)

    # Local search with swaps to escape local optima
    rng = random.Random(42)
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        if not present or not absent:
            break
        improved = False
        # Single add
        for i in absent:
            new_mask = best_mask | (1 << i)
            if valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                improved = True
                break
        if improved:
            continue
        # 1-for-2 swap
        if len(absent) >= 2 and len(present) >= 1:
            attempts = min(500, max(100, len(absent) * 2))
            for _ in range(attempts):
                if time.time() > deadline - 5:
                    break
                ri = rng.choice(present)
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break
        # 2-for-3 swap
        if not improved and len(absent) >= 3 and len(present) >= 2:
            attempts = min(300, max(50, len(absent)))
            for _ in range(attempts):
                if time.time() > deadline - 5:
                    break
                r1 = rng.choice(present)
                r2 = rng.choice(present)
                if r1 == r2:
                    continue
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                a3 = rng.choice(absent)
                if len({a1, a2, a3}) < 3:
                    continue
                new_mask = (best_mask & ~(1 << r1) & ~(1 << r2)) | (1 << a1) | (1 << a2) | (1 << a3)
                if valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 2
                    improved = True
                    break

    # Final greedy
    best_mask, best_count = try_add_edges(best_mask, best_count)

    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
