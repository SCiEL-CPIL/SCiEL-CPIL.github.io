import time
from random import Random
from collections import Counter

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
        
        # Precompute host degrees and remaining capacity
        host_deg = [0] * N
        for (x, y) in used_edges:
            host_deg[x] += 1; host_deg[y] += 1
        host_rem = [N - 1 - host_deg[h] for h in range(N)]
        
        # Dynamic MRV ordering: sort by degree initially, but we will check feasibility dynamically
        # For now, keep static degree-based order as a good heuristic for trees
        var_order = sorted(range(k), key=lambda v: -deg[v])
        
        mapping = [-1] * k
        used_hosts = set()
        nodes = [0]
        
        # Precompute adjacency sets for faster lookup? No, lists are fine for small k.
        # But we can precompute which tree vertices are adjacent to each other.
        
        def backtrack(idx):
            if nodes[0] > node_limit:
                return False
            if idx == k:
                return True
            
            v = var_order[idx]
            cands = []
            
            # Find valid candidates
            for h in range(N):
                if h in used_hosts:
                    continue
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1 and (min(h, mw), max(h, mw)) in used_edges:
                        ok = False
                        break
                if ok:
                    cands.append(h)
            
            if not cands:
                return False
            
            # Heuristic: prioritize candidates with higher remaining degree in host graph
            # This helps fit high-degree tree nodes into high-degree host nodes
            if len(cands) > 1:
                # Sort by remaining host degree descending
                # Using a key function that accesses host_rem
                cands.sort(key=lambda h: -host_rem[h])
                # Add slight randomization to escape local minima?
                # The outer loop handles restarts, so deterministic sort is fine for a single run
                # But maybe shuffle the top few? Let's stick to deterministic for now to be fast.
            
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_limit:
                    return False
                
                mapping[v] = h
                used_hosts.add(h)
                added = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e)
                        added.append(e)
                
                if backtrack(idx + 1):
                    return True
                
                for e in added:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1
            
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
    # Generate a diverse set of orders to try
    # 1. Large-first (standard)
    # 2. Large-first with shuffles
    # 3. Random orders
    # 4. Small-first (sometimes easier to place leaves first?) - unlikely to be better but worth a shot
    # 5. Degree-sorted based on tree structure?
    
    # Let's just use a mix of large-first shuffles and full randoms.
    # The current code generates 300 large-first and 5000 random.
    # Given the 120s limit, we might not use all 5300.
    # Let's increase the number of large-first variants slightly, as they are often good.
    
    orders = [base_order]
    # 500 large-first variants
    for _ in range(500):
        o = list(range(19, -1, -1))
        master_rng.shuffle(o[1:])
        orders.append(o)
    # 5000 random variants
    for _ in range(5000):
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

    if best_result is not None and best_val > 0:
        return best_result
    # Fallback: try to embed T20 alone? Or just return None for all?
    # The task says "Entry k-1 is None or a list...". So None is valid.
    # But we want to maximize edges.
    # If we failed everything, return all None.
    return [None] * 20
