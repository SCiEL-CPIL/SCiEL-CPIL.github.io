import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start = time.time()
    deadline = start + 105  # safety margin below the 120s limit
    trees = instance_trees(20, 611)
    N = 20
    base_order = list(range(19, -1, -1))  # large first: keeps graph dense for hard trees

    def embed_csp(tree_edges, k, used_edges, rng, node_limit=500000):
        if k == 1:
            return [0]
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            deg[a] += 1; deg[b] += 1
            adj[a].append(b); adj[b].append(a)
        
        # Precompute host degree and remaining capacity
        host_deg = [0] * N
        for (x, y) in used_edges:
            host_deg[x] += 1; host_deg[y] += 1
        host_rem = [N - 1 - host_deg[h] for h in range(N)]
        
        # Use dynamic variable ordering: pick unassigned vertex with highest degree among those with assigned neighbors
        # For trees, we can use a BFS-like ordering from high-degree vertices
        var_order = sorted(range(k), key=lambda v: (-deg[v], v))
        
        mapping = [-1] * k
        used_hosts = set()
        nodes = [0]
        
        # Precompute for each tree vertex, which host vertices are invalid due to edge conflicts with already-mapped neighbors
        def get_invalid_hosts(v):
            """Return set of host vertices that would create an edge conflict for tree vertex v"""
            invalid = set()
            for w in adj[v]:
                mw = mapping[w]
                if mw != -1:
                    # Host h is invalid if (h, mw) is in used_edges
                    for h in range(N):
                        if h not in used_hosts and (min(h, mw), max(h, mw)) in used_edges:
                            invalid.add(h)
            return invalid
        
        def backtrack(idx):
            if nodes[0] > node_limit:
                return False
            if idx == k:
                return True
            
            # Dynamic variable ordering: among unassigned, pick the one with most assigned neighbors
            # (this is more adaptive than static ordering)
            best_v = -1
            best_assigned_neighbors = -1
            best_deg = -1
            for i in range(k):
                if mapping[i] != -1:
                    continue
                # Count assigned neighbors
                assigned_count = 0
                for w in adj[i]:
                    if mapping[w] != -1:
                        assigned_count += 1
                # Prefer vertices with more assigned neighbors (more constrained), then higher degree
                if (assigned_count > best_assigned_neighbors or 
                    (assigned_count == best_assigned_neighbors and deg[i] > best_deg)):
                    best_v = i
                    best_assigned_neighbors = assigned_count
                    best_deg = deg[i]
            
            if best_v == -1:
                return False  # Should not happen if idx < k
            
            v = best_v
            cands = []
            for h in range(N):
                if h in used_hosts:
                    continue
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1 and (min(h, mw), max(h, mw)) in used_edges:
                        ok = False; break
                if ok:
                    cands.append(h)
            
            if not cands:
                return False
            
            # Order candidates: prefer hosts with high remaining degree (more flexible for future)
            # But also consider which hosts are "critical" (low remaining degree)
            if len(cands) > 1:
                # Use a heuristic: sort by remaining degree, but with some randomness to avoid getting stuck
                cands.sort(key=lambda h: (-host_rem[h], rng.random()))
            
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_limit:
                    return False
                mapping[v] = h; used_hosts.add(h)
                added = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e); added.append(e)
                if backtrack(idx + 1):
                    return True
                for e in added:
                    used_edges.discard(e)
                used_hosts.discard(h); mapping[v] = -1
            return False

        if backtrack(0):
            return mapping
        return None

    def evaluate(order, rng):
        used_edges = set()
        result = [None] * N
        total = 0
        for idx in order:
            if time.time() > deadline:
                break
            k = idx + 1
            m = embed_csp(trees[idx], k, used_edges, rng)
            if m is not None:
                result[idx] = [int(x) for x in m]
                total += len(trees[idx])
        return result, total

    master_rng = Random(42)
    orders = [base_order]
    # More structured orders: try different anchor points for the largest trees
    for _ in range(100):  # large-first with varied middle
        o = list(range(19, -1, -1))
        # Keep T20, T19 at front, shuffle the rest
        master_rng.shuffle(o[2:])
        orders.append(o)
    # Try embedding smaller trees first (sometimes easier to place)
    for _ in range(200):
        o = list(range(20))
        master_rng.shuffle(o)
        orders.append(o)
    # Hybrid: put a few large trees first, then random
    for _ in range(300):
        o = list(range(20))
        master_rng.shuffle(o)
        # Move 2-3 largest to front
        large = [19, 18, 17]
        master_rng.shuffle(large)
        o[:len(large)] = large
        master_rng.shuffle(o[len(large):])
        orders.append(o)
    # Fully random restarts
    for _ in range(3000):
        o = list(range(20))
        master_rng.shuffle(o)
        orders.append(o)

    best_result = None
    best_val = -1
    attempt = 0
    for o in orders:
        if time.time() > deadline - 3:
            break
        attempt += 1
        r, v = evaluate(o, Random(attempt))
        if v > best_val:
            best_val = v
            best_result = r
        if best_val == 190:
            break

    # If we didn't get perfect, try a few more targeted attempts with different seeds
    if best_val < 190:
        for extra_attempt in range(500):
            if time.time() > deadline - 1:
                break
            attempt += 1
            o = list(range(20))
            master_rng.shuffle(o)
            r, v = evaluate(o, Random(attempt))
            if v > best_val:
                best_val = v
                best_result = r
            if best_val == 190:
                break

    if best_result is not None and best_val > 0:
        return best_result
    return [None] * 19 + [list(range(20))]
