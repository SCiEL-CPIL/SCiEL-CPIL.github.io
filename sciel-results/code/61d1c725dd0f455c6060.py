import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    trees = instance_trees()
    n_trees = 20
    
    # Pre-compute adjacency lists for each tree
    adj = []
    for t in trees:
        k = len(t) + 1
        g = [[] for _ in range(k)]
        for u, v in t:
            g[u].append(v)
            g[v].append(u)
        adj.append(g)
        
    def solve(order, seed):
        rng = random.Random(seed)
        used_edges = set()
        embeddings = [None] * n_trees
        
        def find_embedding(k, g, used_edges, rng):
            if k == 1:
                return [rng.randint(0, 19)]
            
            deg = [len(g[i]) for i in range(k)]
            verts = list(range(k))
            verts.sort(key=lambda x: -deg[x])
            
            phi = [-1] * k
            
            def backtrack(idx):
                if idx == len(verts):
                    return True
                
                v = verts[idx]
                # Try all hosts 0..19
                # We can filter out hosts already in phi
                used_hosts = set(phi)
                
                # Generate candidate hosts
                candidates = [h for h in range(20) if h not in used_hosts]
                rng.shuffle(candidates)
                
                for h in candidates:
                    # Check edges to previously assigned vertices
                    valid = True
                    for u in verts[:idx]:
                        if u in g[v]:
                            e = frozenset((phi[u], h))
                            if e in used_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    phi[v] = h
                    if backtrack(idx + 1):
                        return True
                    phi[v] = -1
                return False
            
            if backtrack(0):
                res = [0] * k
                for v in range(k):
                    res[v] = phi[v]
                return res
            return None

        for k in order:
            g = adj[k-1]
            emb = find_embedding(k, g, used_edges, rng)
            if emb is None:
                return None
            
            for u, v in trees[k-1]:
                e = frozenset((emb[u], emb[v]))
                used_edges.add(e)
            
            embeddings[k-1] = emb
        
        return embeddings

    # Try increasing order
    order_inc = list(range(1, 21))
    # Try decreasing order
    order_dec = list(range(20, 0, -1))
    
    # Interleave or try inc first?
    # Smaller trees are less constrained in terms of edge count, but T20 is very constrained (uses all vertices).
    # Actually, T20 being last means the remaining graph must be isomorphic to T20.
    # T20 being first means we just pick a random spanning tree embedding, and then try to fit the rest.
    # Fitting the rest into the complement of a random spanning tree might be harder?
    # Or easier?
    # Let's try both.
    
    for seed in range(50):
        res = solve(order_inc, seed)
        if res is not None:
            out = []
            for i in range(20):
                if res[i] is None:
                    out.append(None)
                else:
                    out.append([int(x) for x in res[i]])
            return out

    for seed in range(50):
        res = solve(order_dec, seed)
        if res is not None:
            out = []
            for i in range(20):
                if res[i] is None:
                    out.append(None)
                else:
                    out.append([int(x) for x in res[i]])
            return out

    # Fallback
    return [None] * 19 + [list(range(20))]
