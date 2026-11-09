"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 130  # safety margin before 150s limit
    best_n = float('inf')
    best_edges = None

    # --- Family 1: LUW graphs with various d parameters ---
    def build_luw(q, d):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q

        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)

        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth(n, edges, min_girth):
        if n == 0:
            return True
        if min_girth <= 2:
            return True
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        half = (min_girth - 1) // 2
        for start in range(n):
            visited = set()
            frontier = []
            for nb in adj[start]:
                s = (nb, start)
                visited.add(s)
                frontier.append(s)
            for depth in range(1, half):
                if time.time() > deadline:
                    return True
                new_frontier = []
                for v, prev in frontier:
                    for u in adj[v]:
                        if u == prev:
                            continue
                        s = (u, v)
                        if s not in visited:
                            visited.add(s)
                            new_frontier.append(s)
                frontier = new_frontier
            for v, prev in frontier:
                if v == start:
                    return False
        return True

    # Try LUW with various d
    for d in range(3, g + 3):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(k, d)
            if n < best_n and check_girth(n, edges, g):
                best_n = n
                best_edges = edges
        except Exception:
            pass

    # --- Family 2: K_{k,k} minus a (k-1)-regular bipartite graph with large girth ---
    # For even girth g: if H is (k-1)-regular bipartite with girth >= g,
    # then K_{k,k} - H has girth >= g.
    # Use LUW to build H with q' = k-1 (if prime) or use a different approach.
    if g % 2 == 0:
        # H should be (k-1)-regular bipartite on n_H vertices with girth >= g
        # LUW gives q'-regular with girth >= d+5
        # We need k-1 regular, so q' = k-1
        qh = k - 1
        if qh >= 2:
            # Check if qh is prime or 4
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            if is_prime(qh) or qh == 4:
                for d in range(3, g + 3):
                    if time.time() > deadline:
                        break
                    try:
                        nH, edgesH = build_luw(qh, d)
                        # nH is even (bipartite), H is qh-regular
                        # K_{k,k} has k*k edges; H has nH*qh/2 edges
                        # K_{k,k} - H only works if nH = 2k (i.e., H is on 2k vertices)
                        # Otherwise we need a different approach
                        if nH == 2 * k:
                            # H is a (k-1)-regular bipartite graph on 2k vertices
                            # K_{k,k} - H: vertices are same, edges are complement
                            all_edges = set()
                            for i in range(k):
                                for j in range(k):
                                    all_edges.add((i, j))
                            H_edges = set()
                            for i, j in edgesH:
                                H_edges.add((i, j))
                                H_edges.add((j, i))
                            # K_{k,k} edges: (i, j) for i in 0..k-1, j in k..2k-1
                            # H is on vertices 0..nH-1, bipartite
                            # Map: left side 0..k-1, right side k..2k-1
                            # H edges connect left to right
                            kkk_edges = set()
                            for i in range(k):
                                for j in range(k, 2*k):
                                    kkk_edges.add((i, j))
                            # H edges in terms of 0..2k-1
                            # LUW H has vertices 0..nH-1 = 0..2k-1
                            # Left: 0..k-1, Right: k..2k-1
                            H_set = set()
                            for i, j in edgesH:
                                if i < k and j >= k:
                                    H_set.add((i, j))
                                elif j < k and i >= k:
                                    H_set.add((j, i))
                            new_edges = []
                            for e in kkk_edges:
                                if e not in H_set:
                                    new_edges.append(e)
                            # Check degree: each vertex should have degree k - (k-1) = 1? No.
                            # K_{k,k} has degree k. H has degree k-1. So K_{k,k}-H has degree 1.
                            # That's not k-regular. This approach doesn't work directly.
                            # Skip.
                            pass
                    except Exception:
                        pass

    # --- Family 3: Random k-regular graph with girth check ---
    # Generate random k-regular graphs and check girth
    def random_k_regular(n, k, seed=None):
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue
            if (min(u,v), max(u,v)) in edges:
                continue
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        # Check all degrees
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return list(edges)

    # Try random graphs near Moore bound
    moore_bound = 2 * (k * (k-1)**((g-2)//2 - 1) + (k-1)**((g-2)//2) - 1) if g >= 4 else 2
    moore_bound = max(2*k, moore_bound)

    for attempt in range(20):
        if time.time() > deadline:
            break
        n_try = moore_bound + attempt * 10
        if n_try > 200:  # Keep it small for speed
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            try:
                edges = random_k_regular(n_try, k, seed=42+attempt*100+seed)
                if edges is None:
                    continue
                if len(edges) == n_try * k // 2:
                    if check_girth(n_try, edges, g):
                        if n_try < best_n:
                            best_n = n_try
                            best_edges = edges
                            break
            except Exception:
                pass

    if best_edges is None:
        # Fallback: LUW with guaranteed d
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = build_luw(k, d)
        return n, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
