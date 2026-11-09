import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    total_edges = len(edge_set)
    
    def make_edge(u, v):
        return (min(u, v), max(u, v))
    
    def expand_biclique(L, R, remaining):
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(make_edge(w, r) in remaining for r in R):
                    L.add(w)
                    changed = True
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(make_edge(w, l) in remaining for l in L):
                    R.add(w)
                    changed = True
        return L, R
    
    def greedy_partition(remaining, deadline, K=100):
        bicliques = []
        remaining = set(remaining)
        
        while remaining and time.time() < deadline:
            deg = [0] * n
            for (u, v) in remaining:
                deg[u] += 1
                deg[v] += 1
            
            edge_list = list(remaining)
            edge_list.sort(key=lambda e: deg[e[0]] * deg[e[1]], reverse=True)
            
            best_L = None
            best_R = None
            best_size = 0
            
            K_eff = min(K, len(edge_list))
            for (u, v) in edge_list[:K_eff]:
                L, R = expand_biclique({u}, {v}, remaining)
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R
            
            if best_L is None:
                u, v = next(iter(remaining))
                bicliques.append((frozenset([u]), frozenset([v])))
                remaining.discard(make_edge(u, v))
                continue
            
            bicliques.append((frozenset(best_L), frozenset(best_R)))
            
            to_remove = set()
            for l in best_L:
                for r in best_R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove
        
        for (u, v) in remaining:
            bicliques.append((frozenset([u]), frozenset([v])))
        
        return bicliques
    
    def can_merge(L1, R1, L2, R2):
        candidates = []
        newL = L1 | L2
        newR = R1 | R2
        if newL and newR and not (newL & newR):
            candidates.append((newL, newR))
        newL = L1 | R2
        newR = R1 | L2
        if newL and newR and not (newL & newR):
            candidates.append((newL, newR))
        if R2 <= R1:
            newL = L1 | L2
            if newL and not (newL & R1):
                candidates.append((newL, R1))
        if L2 <= L1:
            newR = R1 | R2
            if newR and not (L1 & newR):
                candidates.append((L1, newR))
        if R1 <= R2:
            newL = L1 | L2
            if newL and not (newL & R2):
                candidates.append((newL, R2))
        if L1 <= L2:
            newR = R1 | R2
            if newR and not (L2 & newR):
                candidates.append((L2, newR))
        
        for newL, newR in candidates:
            valid = True
            for l in newL:
                for r in newR:
                    if l == r:
                        continue
                    if make_edge(l, r) not in edge_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return True, newL, newR
        return False, None, None
    
    def try_merge_all(bicliques, deadline):
        bicliques = list(bicliques)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    ok, newL, newR = can_merge(L1, R1, L2, R2)
                    if ok:
                        bicliques.pop(j)
                        bicliques.pop(i)
                        bicliques.append((frozenset(newL), frozenset(newR)))
                        improved = True
                        break
                if improved:
                    break
        return bicliques
    
    # Try multiple K values and keep the best
    best_bicliques = None
    best_count = total_edges + 1
    
    for K in [50, 100, 200]:
        if time.time() > deadline:
            break
        bics = greedy_partition(edge_set, deadline, K)
        if time.time() < deadline:
            bics = try_merge_all(bics, deadline)
        if len(bics) < best_count:
            best_count = len(bics)
            best_bicliques = bics
    
    if best_bicliques is None:
        best_bicliques = [(frozenset([u]), frozenset([v])) for (u, v) in edge_set]
    
    result = []
    for L, R in best_bicliques:
        result.append([list(L), list(R)])
    
    covered = set()
    valid = True
    for bc in result:
        left, right = bc
        if not left or not right or set(left) & set(right):
            valid = False
            break
        for l in left:
            if not isinstance(l, int) or isinstance(l, bool):
                valid = False
                break
            for r in right:
                if not isinstance(r, int) or isinstance(r, bool) or l == r:
                    valid = False
                    break
                e = make_edge(l, r)
                if e not in edge_set or e in covered:
                    valid = False
                    break
                covered.add(e)
        if not valid:
            break
    
    if not valid or covered != edge_set:
        result = [[ [u], [v] ] for (u, v) in edge_set]
    
    return result
