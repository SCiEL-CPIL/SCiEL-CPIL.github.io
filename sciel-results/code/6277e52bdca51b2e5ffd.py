import random

def _greedy_expand(u, v, remaining, adj, n, strategy='size'):
    L = {u}
    R = {v}
    changed = True
    while changed:
        changed = False
        # Try L
        best_l_gain = 0
        best_l_v = None
        for w in range(n):
            if w in L or w in R: continue
            ok = all((min(w,r), max(w,r)) in remaining for r in R)
            if ok:
                if strategy == 'size':
                    gain = len(R)
                else: # degree
                    gain = len(adj[w]) * len(R)
                if gain > best_l_gain:
                    best_l_gain = gain
                    best_l_v = w
        if best_l_v is not None:
            L.add(best_l_v)
            changed = True
            continue
        # Try R
        best_r_gain = 0
        best_r_v = None
        for w in range(n):
            if w in L or w in R: continue
            ok = all((min(l,w), max(l,w)) in remaining for l in L)
            if ok:
                if strategy == 'size':
                    gain = len(L)
                else:
                    gain = len(adj[w]) * len(L)
                if gain > best_r_gain:
                    best_r_gain = gain
                    best_r_v = w
        if best_r_v is not None:
            R.add(best_r_v)
            changed = True
    return L, R

def _greedy_expand_random(u, v, remaining, adj, n, seed):
    random.seed(seed)
    L = {u}
    R = {v}
    changed = True
    while changed:
        changed = False
        candidates_l = []
        candidates_r = []
        for w in range(n):
            if w in L or w in R: continue
            if all((min(w,r), max(w,r)) in remaining for r in R):
                candidates_l.append(w)
            if all((min(l,w), max(l,w)) in remaining for l in L):
                candidates_r.append(w)
        
        if candidates_l:
            # Pick one randomly or by weight
            w = random.choice(candidates_l)
            L.add(w)
            changed = True
            continue
        if candidates_r:
            w = random.choice(candidates_r)
            R.add(w)
            changed = True
            continue
    return L, R

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set((min(a,b), max(a,b)) for a,b in edges)
    
    # Base adj
    base_adj = [set() for _ in range(n)]
    for a,b in edges:
        base_adj[a].add(b)
        base_adj[b].add(a)
        
    best_bicliques = None
    best_len = float('inf')
    
    # Run multiple strategies
    for i in range(20):
        # Copy adj because we modify it
        adj = [set(s) for s in base_adj]
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            # Pick pivot
            # Strategy 1: First edge
            # Strategy 2: Edge with max degree
            if i % 2 == 0:
                u, v = next(iter(remaining))
            else:
                # Find max degree edge in remaining
                best_e = None
                best_score = -1
                for e in remaining:
                    a, b = e
                    score = len(adj[a]) + len(adj[b])
                    if score > best_score:
                        best_score = score
                        best_e = e
                u, v = best_e
                
            # Expand
            if i < 5:
                L, R = _greedy_expand(u, v, remaining, adj, n, 'size')
            elif i < 10:
                L, R = _greedy_expand(u, v, remaining, adj, n, 'degree')
            else:
                L, R = _greedy_expand_random(u, v, remaining, adj, n, i*123 + hash(frozenset(remaining)))
                
            if not L or not R:
                L, R = {u}, {v}
                
            bicliques.append([sorted(L), sorted(R)])
            
            for lu in L:
                for rv in R:
                    e = (min(lu, rv), max(lu, rv))
                    remaining.discard(e)
                    adj[lu].discard(rv)
                    adj[rv].discard(lu)
                    
        if len(bicliques) < best_len:
            best_len = len(bicliques)
            best_bicliques = bicliques
            
    return [[list(l), list(r)] for l, r in best_bicliques]
