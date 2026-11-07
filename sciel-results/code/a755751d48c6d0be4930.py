import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_partition(strategy, seed=0):
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    adj = _build_adjacency()
    bicliques = []
    
    degrees = [len(adj[i]) for i in range(n)]
    
    while edge_set:
        if time.time() > 55:
            break
            
        if strategy == 'max_degree':
            best_edge = None
            best_score = -1
            for u, v in edge_set:
                score = degrees[u] + degrees[v]
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
        elif strategy == 'first':
            u, v = next(iter(edge_set))
        elif strategy == 'random':
            u, v = random.choice(list(edge_set))
        else:
            u, v = next(iter(edge_set))

        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for r in R:
                    if r not in adj[w]:
                        can_add = False
                        break
                    e = (min(w, r), max(w, r))
                    if e not in edge_set:
                        can_add = False
                        break
                if can_add:
                    L.add(w)
                    changed = True
            
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for l in L:
                    if l not in adj[w]:
                        can_add = False
                        break
                    e = (min(l, w), max(l, w))
                    if e not in edge_set:
                        can_add = False
                        break
                if can_add:
                    R.add(w)
                    changed = True
        
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in edge_set:
                    edge_set.remove(e)
        
        bicliques.append([sorted(L), sorted(R)])
        
    return bicliques

def construct():
    start_time = time.time()
    best_bicliques = None
    min_count = float('inf')
    
    strategies = ['max_degree', 'first', 'random']
    
    for strat in strategies:
        for seed in range(20):
            if time.time() - start_time > 50:
                break
            random.seed(seed)
            res = _greedy_partition(strat, seed)
            if len(res) < min_count:
                min_count = len(res)
                best_bicliques = res
        if time.time() - start_time > 50:
            break
            
    if best_bicliques is None:
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    return best_bicliques
