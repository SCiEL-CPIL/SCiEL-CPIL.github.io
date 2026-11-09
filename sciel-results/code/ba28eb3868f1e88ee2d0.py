import time

def construct():
    n = 15
    start = time.time()
    timeout = 55.0
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v); adj[v].add(u)
        return adj
    
    def is_valid(edges):
        adj = make_adj(edges)
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    best = None
    best_count = -1
    
    # Base: K_{7,8}
    L = list(range(7)); R = list(range(7,15))
    base = set()
    for u in L:
        for v in R:
            base.add((u,v))
    
    def greedy_add(edges):
        edges = set(edges)
        changed = True
        while changed:
            changed = False
            for (u,v) in all_pairs:
                if (u,v) in edges: continue
                if is_valid(edges | {(u,v)}):
                    edges.add((u,v))
                    changed = True
        return edges
    
    # Try several random perturbations of base
    import random
    random.seed(42)
    candidates = [base]
    for _ in range(50):
        pert = set(base)
        for _ in range(random.randint(0,5)):
            e = random.choice(list(pert))
            pert.remove(e)
        candidates.append(pert)
    
    for cand in candidates:
        if time.time() - start > timeout: break
        res = greedy_add(cand)
        if len(res) > best_count:
            best_count = len(res)
            best = res
    
    # Local search on best
    if best is not None and time.time() - start < timeout:
        cur = set(best)
        while time.time() - start < timeout:
            improved = False
            # Try remove one, add greedily
            for e in list(cur):
                if time.time() - start > timeout: break
                trial = cur - {e}
                trial = greedy_add(trial)
                if len(trial) > len(cur):
                    cur = trial
                    improved = True
                    break
            if not improved:
                break
        if len(cur) > best_count:
            best_count = len(cur)
            best = cur
    
    return [[u,v] for u,v in best]
