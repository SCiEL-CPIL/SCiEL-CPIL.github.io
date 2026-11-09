import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    initial_edges = set(edges)
    
    best_partition = None
    best_count = float('inf')
    
    # Run multiple random restarts
    for run_seed in range(20):
        if not _deadline_ok():
            break
            
        current_edges = set(initial_edges)
        partition = []
        rng = random.Random(seed + run_seed)
        
        while current_edges:
            if not _deadline_ok():
                # Fallback: just add remaining edges as K2
                for e in current_edges:
                    partition.append([e[0], e[1]])
                break
            
            # 1. Find edge with maximum common neighbors in the remaining graph
            # to prioritize forming larger cliques (especially triangles)
            
            max_common = -1
            candidates = []
            
            for u, v in current_edges:
                common = 0
                # Iterate over smaller adjacency list for efficiency
                if len(adj[u]) < len(adj[v]):
                    for w in adj[u]:
                        if w == v: continue
                        e_vw = _edge_key(v, w)
                        e_uw = _edge_key(u, w)
                        if e_vw in current_edges and e_uw in current_edges:
                            common += 1
                else:
                    for w in adj[v]:
                        if w == u: continue
                        e_uw = _edge_key(u, w)
                        e_vw = _edge_key(v, w)
                        if e_uw in current_edges and e_vw in current_edges:
                            common += 1
                
                if common > max_common:
                    max_common = common
                    candidates = [(u, v)]
                elif common == max_common:
                    candidates.append((u, v))
            
            if not candidates:
                break
                
            u, v = rng.choice(candidates)
            
            # 2. Expand to maximal clique
            clique = [u, v]
            
            # Find candidates for expansion
            candidates_w = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e_uw = _edge_key(u, w)
                e_vw = _edge_key(v, w)
                if e_uw in current_edges and e_vw in current_edges:
                    candidates_w.append(w)
            
            while candidates_w:
                if not _deadline_ok():
                    break
                
                # Pick w with max degree in the subgraph induced by candidates_w
                max_deg = -1
                best_w = -1
                tie_list = []
                
                for w in candidates_w:
                    deg = 0
                    for x in candidates_w:
                        if w == x:
                            continue
                        e_wx = _edge_key(w, x)
                        if e_wx in current_edges:
                            deg += 1
                    
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                        tie_list = [w]
                    elif deg == max_deg:
                        tie_list.append(w)
                
                if best_w == -1:
                    break
                    
                if len(tie_list) > 1:
                    best_w = rng.choice(tie_list)
                    
                clique.append(best_w)
                candidates_w.remove(best_w)
                
                # Filter candidates_w: must be connected to best_w
                new_candidates_w = []
                for w in candidates_w:
                    e_wbw = _edge_key(w, best_w)
                    if e_wbw in current_edges:
                        new_candidates_w.append(w)
                candidates_w = new_candidates_w
                
            partition.append(clique)
            
            # 3. Remove edges of the clique
            for a, b in combinations(clique, 2):
                e = _edge_key(a, b)
                if e in current_edges:
                    current_edges.remove(e)
                    
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = partition
            
    if best_partition is None:
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
