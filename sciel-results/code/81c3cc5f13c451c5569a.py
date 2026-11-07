import time
import random

N = 15
DEADLINE = 85.0  # Safety margin under the 90s limit

def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        trial = edges + [(u, v)]
        if _valid(trial):
            edges = trial
            present.add(key)
    return edges

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)

    # K_{7,8}: 56 edges, valid
    k78 = [(a, b) for a in range(7) for b in range(7, 15)]
    best = k78[:]
    best_len = len(k78)

    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand[:]

    # Phase 1: Greedy from K_{7,8} with many random orders
    for _ in range(200):
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, k78)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 2: Remove 1-2 edges from K_{7,8}, then greedy add
    # Removing edges within A or B doesn't help (they don't exist in K_{7,8})
    # Removing cross edges (a,b) reduces degrees, might allow intra-partition edges
    for _ in range(300):
        if time.time() - t0 > DEADLINE:
            break
        base = k78[:]
        # Remove 1 or 2 random edges
        n_remove = rng.choice([1, 2])
        for _ in range(n_remove):
            if base:
                idx = rng.randint(0, len(base) - 1)
                base.pop(idx)
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, base)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 3: Random dense graphs with repair + greedy
    for _ in range(100):
        if time.time() - t0 > DEADLINE:
            break
        adj = [set() for _ in range(N)]
        p = rng.uniform(0.5, 0.9)
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        edges = [[u, v] for u in range(N) for v in adj[u] if u < v]
        
        # Repair: remove edges from violations
        max_steps = 50
        steps = 0
        valid = False
        while not valid and steps < max_steps:
            deg = _degrees(adj)
            viol = None
            for u in range(N):
                au = adj[u]
                for v in au:
                    if v <= u:
                        continue
                    av = adj[v]
                    for s in au:
                        if s == v:
                            continue
                        ds = deg[s]
                        for t in av:
                            if t == u or t == s:
                                continue
                            if ds == deg[t]:
                                viol = (s, u, v, t)
                                break
                        if viol:
                            break
                    if viol:
                        break
                if viol:
                    break
            
            if viol is None:
                valid = True
                break
            
            s, u, v, t = viol
            e_options = [(s, u), (u, v), (v, t)]
            a, b = rng.choice(e_options)
            adj[a].remove(b)
            adj[b].remove(a)
            steps += 1
        
        if valid:
            cand = [[u, v] for u in range(N) for v in adj[u] if u < v]
            consider(cand)
            if time.time() - t0 < DEADLINE:
                order = all_edges[:]
                rng.shuffle(order)
                cand2 = _greedy(order, cand)
                consider(cand2)

    # Phase 4: Local search on best - swap edges
    for _ in range(200):
        if time.time() - t0 > DEADLINE:
            break
        base = best[:]
        if len(base) < 2:
            continue
        # Remove 1-2 edges
        n_remove = rng.choice([1, 2])
        removed = []
        for _ in range(n_remove):
            if base:
                idx = rng.randint(0, len(base) - 1)
                removed.append(base.pop(idx))
        
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, base)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 5: Try other bipartite splits as seeds
    for a in range(4, 11):
        b = N - a
        seed = [(u, v) for u in range(a) for v in range(a, N)]
        if time.time() - t0 > DEADLINE:
            break
        for _ in range(5):
            if time.time() - t0 > DEADLINE:
                break
            order = all_edges[:]
            rng.shuffle(order)
            cand = _greedy(order, seed)
            consider(cand)

    result = [[int(u), int(v)] for (u, v) in best]
    return result
