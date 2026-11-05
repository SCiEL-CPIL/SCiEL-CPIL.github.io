import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _canonical(cycle):
    if len(cycle) < 3:
        return None
    m = len(cycle)
    best = None
    for k in range(m):
        rot = cycle[k:] + cycle[:k]
        rev = list(reversed(rot))
        cand = rot if rot < rev else rev
        if best is None or cand < best:
            best = cand
    return tuple(best)


def _generate_cycle_pool(adj, n, edge_set, rng, time_limit):
    """Generate a pool of simple cycles (canonical tuples of vertices)."""
    start_time = time.time()
    pool = []
    seen = set()

    def _add(cycle):
        if len(cycle) < 3:
            return
        c = _canonical(cycle)
        if c is None:
            return
        # verify edges exist
        m = len(c)
        for i in range(m):
            a = c[i]
            b = c[(i + 1) % m]
            if tuple(sorted((a, b))) not in edge_set:
                return
        if c not in seen:
            seen.add(c)
            pool.append(list(c))

    # Strategy 1: many randomized DFS walks, harvest any back-edge cycle
    def _random_walk_cycle(start):
        # iterative DFS with random neighbor order; record parent to extract cycle
        visited = {start}
        parent = {start: None}
        stack = [start]
        while stack:
            node = stack[-1]
            nbrs = [v for v in adj[node] if v in visited or True]
            # try to find a visited neighbor (back edge) first
            back = None
            for v in adj[node]:
                if v in visited and v != parent.get(node):
                    back = v
                    break
            if back is not None:
                # extract cycle from back .. node
                cyc = [back]
                cur = node
                while cur != back:
                    cyc.append(cur)
                    cur = parent[cur]
                cyc.append(back)
                return cyc
            # else pick an unvisited neighbor
            unvisited = [v for v in adj[node] if v not in visited]
            if not unvisited:
                stack.pop()
                continue
            rng.shuffle(unvisited)
            nxt = unvisited[0]
            visited.add(nxt)
            parent[nxt] = node
            stack.append(nxt)
        return None

    # Strategy 2: DFS from a start, force return to start to get a start-based cycle
    def _start_cycle(start):
        visited = {start}
        parent = {start: None}
        path = [start]
        stack = [(start, iter(list(adj[start])))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt == start and len(path) >= 3:
                    return path[:]
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = node
                    path.append(nxt)
                    stack.append((nxt, iter(list(adj[nxt]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
                # mark back? keep visited to allow other branches (simple path constraint)
        return None

    # Generate a batch
    for _ in range(20000):
        if time.time() - start_time > time_limit:
            break
        # pick a start weighted by degree
        degs = [len(adj[v]) for v in range(n)]
        total = sum(degs)
        if total == 0:
            break
        r = rng.random() * total
        acc = 0
        start = n - 1
        for v in range(n):
            acc += degs[v]
            if r <= acc:
                start = v
                break

        if rng.random() < 0.5:
            cyc = _random_walk_cycle(start)
        else:
            cyc = _start_cycle(start)
        if cyc:
            _add(cyc)
        if len(pool) >= 3000:
            break

    # Strategy 3: enumerate all short cycles (length 3 and 4) deterministically
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # triangles
            for w in adj[v]:
                if w > v and w in adj[u]:
                    _add([u, v, w])
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w <= v:
                    continue
                if w in adj[u]:
                    continue
                for z in adj[w]:
                    if z <= w:
                        continue
                    if z in adj[u]:
                        _add([u, v, w, z])

    return pool


def _solve_sa(pool, edge_set, n, rng, time_limit):
    """
    Simulated Annealing to find a set of edge-disjoint cycles maximizing sum(|c|-1).
    """
    start_time = time.time()
    m = len(pool)
    if m == 0:
        return []

    # Precompute cycle edges and weights (|c|-1)
    cyc_edges = []
    cyc_weights = []
    for cyc in pool:
        L = len(cyc)
        es = []
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            es.append(tuple(sorted((a, b))))
        cyc_edges.append(es)
        cyc_weights.append(L - 1)

    # Initial solution: Greedy pick by weight descending
    used = set()
    current_set = []
    current_score = 0
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(m), key=lambda i: -cyc_weights[i])
    for i in sorted_indices:
        if time.time() - start_time > time_limit * 0.1: break
        es = cyc_edges[i]
        if any(e in used for e in es):
            continue
        for e in es:
            used.add(e)
        current_set.append(i)
        current_score += cyc_weights[i]

    best_set = current_set[:]
    best_score = current_score

    # SA Parameters
    T = 5.0
    T_min = 0.1
    
    def can_add(cyc_idx, used_set):
        es = cyc_edges[cyc_idx]
        for e in es:
            if e in used_set:
                return False
        return True

    def add_cycle(cyc_idx):
        nonlocal current_score
        for e in cyc_edges[cyc_idx]:
            used.add(e)
        current_set.append(cyc_idx)
        current_score += cyc_weights[cyc_idx]

    def remove_cycle(cyc_idx):
        nonlocal current_score
        for e in cyc_edges[cyc_idx]:
            used.remove(e)
        current_set.remove(cyc_idx)
        current_score -= cyc_weights[cyc_idx]

    while time.time() - start_time < time_limit:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random cycle
            i = rng.randint(0, m - 1)
            if i in current_set:
                continue
            if can_add(i, used):
                add_cycle(i)
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
        elif move_type < 0.7:
            # Remove a random cycle
            if not current_set:
                continue
            i = rng.choice(current_set)
            remove_cycle(i)
            delta = -cyc_weights[i]
            if rng.random() < pow(2.71828, delta / T):
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                add_cycle(i)
        else:
            # Swap
            if not current_set:
                continue
            i_out = rng.choice(current_set)
            i_in = rng.randint(0, m - 1)
            if i_in in current_set:
                continue
            
            e_out_set = set(cyc_edges[i_out])
            e_in_list = cyc_edges[i_in]
            
            valid = True
            for e in e_in_list:
                if e in e_out_set:
                    continue
                if e in used:
                    valid = False
                    break
            
            if valid:
                delta = cyc_weights[i_in] - cyc_weights[i_out]
                
                remove_cycle(i_out)
                add_cycle(i_in)
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    
                if delta < 0:
                    if rng.random() >= pow(2.71828, delta / T):
                        remove_cycle(i_in)
                        add_cycle(i_out)
                
        T *= 0.999
        if T < T_min:
            T = T_min

    return [pool[i] for i in best_set]


def _solve_one(edges, n, deadline):
    if not edges:
        return []

    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[b := v].add(u)

    # time budget split
    t0 = time.time()
    remaining = deadline - t0
    pool_time = remaining * 0.4
    sa_time = remaining * 0.6

    rng = random.Random(777)
    pool = _generate_cycle_pool(adj, n, edge_set, rng, pool_time)

    selected = _solve_sa(pool, edge_set, n, rng, sa_time)

    # Build pieces: selected cycles + leftover edges as singletons
    used = set()
    for cyc in selected:
        L = len(cyc)
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            used.add(tuple(sorted((a, b))))

    pieces = [list(c) for c in selected]
    for e in edge_set:
        if e not in used:
            pieces.append(list(e))

    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # give each graph a slice; ensure last finishes before budget
        remaining = total_budget - (time.time() - start_time)
        per = remaining / (len(specs) - i)
        graph_deadline = time.time() + per
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
