import time
import random

N = 15
DEADLINE = 55.0  # Safety margin under the 90s limit

def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    # For every middle edge (u, v) with u < v, for every simple 3-edge path
    # s - u - v - t  (s in adj[u]\{v}, t in adj[v]\{u}, s != t),
    # require deg[s] != deg[t].
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
    # single pass over the shuffled order; add an edge if it keeps validity
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

    # A guaranteed-valid seed: the star centred at 0 has no 3-edge path at all.
    star = [(0, v) for v in range(1, N)]
    
    # Another strong seed: Complete bipartite K_{7,8}
    # Partition A: 0..6 (7 vertices), Partition B: 7..14 (8 vertices)
    # Edges: all (a, b) for a in A, b in B
    # Degrees: Vertices in A have degree 8, Vertices in B have degree 7.
    # Any 3-edge path s-u-v-t:
    # If u in A, v in B: s in B, t in A. deg(s)=7, deg(t)=8. 7!=8. Valid.
    # If u in B, v in A: s in A, t in B. deg(s)=8, deg(t)=7. 8!=7. Valid.
    # So K_{7,8} is valid and has 56 edges.
    k78 = [(a, b) for a in range(7) for b in range(7, 15)]
    
    best = k78
    best_len = len(k78)

    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand

    # Phase 1: many randomized greedy builds from K_{7,8} and star.
    # Since K_{7,8} is already 56 edges, we start there and try to add more.
    # We also try starting from empty or star to see if we find something else,
    # but K_{7,8} is a very strong baseline.
    
    # Let's try to add edges to K_{7,8} specifically.
    # Missing edges are within A and within B.
    # If we add an edge (a1, a2) within A:
    # New paths involving this edge:
    # 1. s-a1-a2-t. s in N(a1)\{a2}, t in N(a2)\{a1}.
    #    N(a1) contains all B and a2. So s can be in B or a2 (excluded). So s in B.
    #    N(a2) contains all B and a1. So t can be in B or a1 (excluded). So t in B.
    #    deg(s) = 7, deg(t) = 7. Equal! Invalid.
    # So we cannot add ANY edge within A or within B to K_{7,8} without creating a violation
    # immediately?
    # Let's re-verify.
    # Path s-a1-a2-t.
    # s is neighbor of a1. a1 is in A. Neighbors of a1 are all in B.
    # So s must be in B. deg(s) = 7.
    # t is neighbor of a2. a2 is in A. Neighbors of a2 are all in B.
    # So t must be in B. deg(t) = 7.
    # Yes, deg(s) == deg(t) == 7. Violation.
    # So K_{7,8} is "locally optimal" in the sense that no single edge can be added.
    # But maybe we can swap edges? Or start from a different structure?
    
    # What if we use a different bipartite split?
    # K_{a, b} with a+b=15.
    # Degrees: A has degree b, B has degree a.
    # Path s-u-v-t.
    # If u in A, v in B: s in B, t in A. deg(s)=a, deg(t)=b.
    # Need a != b.
    # Since a+b=15, a and b are different unless 15 is even (it's not).
    # So any complete bipartite graph K_{a, 15-a} is valid!
    # Edge count = a(15-a).
    # Maximize a(15-a). Max at a=7 or 8.
    # 7*8 = 56.
    # So 56 is the max for complete bipartite.
    
    # Can we do better than 56?
    # Total edges in K15 is 105.
    # Let's try to add edges to K_{7,8} by modifying the structure.
    
    # Strategy:
    # 1. Start with K_{7,8}.
    # 2. Try to remove an edge and add two others? (2-opt style)
    # 3. Or just use the greedy approach on a shuffled set of all edges, starting from K_{7,8}.
    #    Since greedy adds edges if valid, and K_{7,8} is valid, it will keep K_{7,8} and try to add more.
    #    As shown, adding any edge to K_{7,8} is invalid. So greedy will stop at 56.
    
    # We need a smarter search.
    # Maybe start from a denser graph and remove edges to fix violations?
    # Or use a different seed.
    
    # Let's try a seed that is not bipartite.
    # What if we have a triangle?
    # If we have a triangle 0-1-2-0.
    # Degrees might be balanced.
    
    # Let's stick to the greedy approach but with multiple diverse seeds.
    # Seeds:
    # 1. K_{7,8} (56 edges)
    # 2. K_{6,9} (54 edges)
    # 3. K_{5,10} (50 edges)
    # 4. Random dense graph, repaired.
    # 5. Star (14 edges)
    
    seeds = [k78]
    # Add other bipartite splits
    for a in [5, 6, 8, 9, 10]:
        b = 15 - a
        seed = [(u, v) for u in range(a) for v in range(a, 15)]
        seeds.append(seed)
    seeds.append(star)
    seeds.append([])
    
    # Phase 1: Greedy from seeds
    for seed in seeds:
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, seed)
        consider(cand)
        # Second pass
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 2: Local search on best
    # Try removing an edge and re-greedying.
    # Since K_{7,8} is a local optimum for single-edge addition, 
    # removing an edge might open up space for more edges elsewhere.
    
    for _ in range(20): # 20 iterations of local search
        if time.time() - t0 > DEADLINE:
            break
        
        base = best[:]
        # Remove a random edge
        if not base:
            continue
        idx = rng.randint(0, len(base) - 1)
        reduced = base[:idx] + base[idx + 1:]
        
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, reduced)
        consider(cand)
        
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 3: Random restarts with repair
    # Start with a random graph with high density, then remove edges to fix violations.
    # This might find non-bipartite structures.
    for _ in range(50):
        if time.time() - t0 > DEADLINE:
            break
        
        adj = [set() for _ in range(N)]
        # Start with K_{7,8} and add random edges within partitions
        # We know adding (a1, a2) is bad.
        # But maybe adding (a1, a2) and removing (b1, b2) is good?
        
        # Let's just do random dense graph
        for u in range(N):
            for v in range(u+1, N):
                if rng.random() < 0.8:
                    adj[u].add(v)
                    adj[v].add(u)
        
        edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        
        # Repair
        max_steps = 100
        steps = 0
        valid = False
        while not valid and steps < max_steps:
            deg = _degrees(adj)
            viol = None
            for u in range(N):
                au = adj[u]
                for v in au:
                    if v <= u: continue
                    av = adj[v]
                    for s in au:
                        if s == v: continue
                        ds = deg[s]
                        for t in av:
                            if t == u or t == s: continue
                            if ds == deg[t]:
                                viol = (s, u, v, t)
                                break
                        if viol: break
                    if viol: break
                if viol: break
            
            if viol is None:
                valid = True
                break
            
            s, u, v, t = viol
            # Remove one of the edges (s,u), (u,v), (v,t)
            # Randomly pick
            e_options = [(s, u), (u, v), (v, t)]
            a, b = rng.choice(e_options)
            adj[a].remove(b)
            adj[b].remove(a)
            steps += 1
        
        if valid:
            # Convert to list of lists
            cand = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        cand.append([u, v])
            consider(cand)
            
            # Greedy add
            if time.time() - t0 < DEADLINE:
                order = all_edges[:]
                rng.shuffle(order)
                cand2 = _greedy(order, cand)
                consider(cand2)

    result = [[int(u), int(v)] for (u, v) in best]
    return result
