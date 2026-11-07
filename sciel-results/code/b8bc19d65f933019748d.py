import time
from collections import deque
from random import Random


def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
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


def _embed(tree_edges, k, free, rng, time_limit=1.0):
    """Find an injective host map for tree (vertices 0..k-1) using only free edges."""
    if k == 1:
        return [0]
    adj = [[] for _ in range(k)]
    for (u, v) in tree_edges:
        adj[u].append(v); adj[v].append(u)
    
    # BFS order from 0
    order = []
    parent = [-1] * k
    seen = [False] * k
    seen[0] = True
    dq = deque([0])
    while dq:
        x = dq.popleft()
        order.append(x)
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True; parent[y] = x; dq.append(y)
                
    start_time = time.time()
    
    # Try multiple times with different random shuffles within the time limit
    while time.time() - start_time < time_limit:
        host = [-1] * k
        used = set()
        counter = [0]
        max_nodes = 1000000 # High limit, rely on time check
        
        def dfs(idx):
            if time.time() - start_time > time_limit:
                return False
            if counter[0] > max_nodes:
                return False
            if idx == len(order):
                return True
            v = order[idx]
            p = parent[v]
            if p == -1:
                cands = [h for h in range(20) if h not in used]
            else:
                hp = host[p]
                cands = [h for h in range(20) if h != hp and h not in used and frozenset((hp, h)) in free]
            
            if not cands:
                return False
                
            rng.shuffle(cands)
            
            for h in cands:
                counter[0] += 1
                if counter[0] > max_nodes:
                    return False
                host[v] = h
                used.add(h)
                if dfs(idx + 1):
                    return True
                host[v] = -1
                used.remove(h)
            return False

        if dfs(0):
            return [int(x) for x in host]
        
        # If failed, loop restarts with new shuffle (rng state advances)
        
    return None


def construct():
    trees = instance_trees(20, 611)
    all_edges = set()
    for a in range(20):
        for b in range(a + 1, 20):
            all_edges.add(frozenset((a, b)))
    best = None
    best_val = -1
    start = time.time()
    deadline = start + 110.0
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rng = Random(attempt * 7919 + 13)
        free = set(all_edges)
        result = [None] * 20
        # largest (most constrained) trees first
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: -(i + 1))
        val = 0
        for i in order_idx:
            if time.time() > deadline:
                break
            k = i + 1
            te = trees[i]
            if k == 1:
                result[i] = [0]
                continue
            # Use a small time limit per tree to allow multiple attempts overall
            # and to prevent getting stuck on one tree
            emb = _embed(te, k, free, rng, time_limit=0.5)
            if emb is not None and len(set(emb)) == k:
                ok = True
                es = []
                for (u, v) in te:
                    e = frozenset((emb[u], emb[v]))
                    if e not in free:
                        ok = False
                        break
                    es.append(e)
                if ok:
                    for e in es:
                        free.remove(e)
                    result[i] = [int(x) for x in emb]
                    val += k - 1
                # If not ok, we just skip this tree (leave as None)
                # This allows partial packings
            # If emb is None, we also skip
        if val > best_val:
            best_val = val
            best = result
        if best_val >= 190:
            break
    if best is None:
        # Fallback: return a trivial packing if nothing was found
        # This shouldn't happen as even T1 and T2 are easy
        best = [None] * 20
        if trees[0]: best[0] = [0]
        if trees[1]: best[1] = [0, 1]
    out = []
    for e in best:
        if e is None:
            out.append(None)
        else:
            out.append([int(x) for x in e])
    return out
