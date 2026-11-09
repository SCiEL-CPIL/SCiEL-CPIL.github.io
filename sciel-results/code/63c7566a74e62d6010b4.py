import random
import math

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_triangles(n, adj):
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
    return triangles

def _get_score_and_worst(n, adj):
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0, []
    max_sum = 0
    worst = []
    for t in triangles:
        s = len(adj[t[0]]) + len(adj[t[1]]) + len(adj[t[2]])
        if s > max_sum:
            max_sum = s
            worst = [t]
        elif s == max_sum:
            worst.append(t)
    return max_sum, worst

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def initial_graph():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add(frozenset((u, v)))
        edges.add(frozenset((0, 1)))
        return [list(e) for e in edges]
    
    edges = initial_graph()
    adj = _build_adjacency(n, edges)
    current_score, worst_tris = _get_score_and_worst(n, adj)
    
    best_score = current_score
    best_edges = edges[:]
    
    def edge_key(u, v):
        return frozenset((u, v))
        
    # Precompute a list of all non-edges? No, it changes.
    # But we can maintain a set of current edges.
    
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.01
    max_iters = 150000
    
    for it in range(max_iters):
        if temp < min_temp:
            break
            
        edges_set = set(frozenset(e) for e in edges)
        
        # Determine removal candidate
        remove_candidates = list(edges_set)
        if not remove_candidates:
            temp *= cooling
            continue
            
        # Biased removal: prefer edges incident to vertices in worst triangles
        if worst_tris and random.random() < 0.5:
            # Pick a vertex from a worst triangle
            tri = random.choice(worst_tris)
            v_crit = random.choice(tri)
            # Find edges incident to v_crit that are NOT in the triangle
            # Triangle edges are (u,v), (v,w), (u,w)
            tri_edges = set()
            for i in range(3):
                for j in range(i+1, 3):
                    tri_edges.add(frozenset((tri[i], tri[j])))
            
            incident = [e for e in remove_candidates if v_crit in e and e not in tri_edges]
            if incident:
                e_remove = random.choice(incident)
            else:
                e_remove = random.choice(remove_candidates)
        else:
            e_remove = random.choice(remove_candidates)
            
        # Determine addition candidate
        # Find all non-edges
        # Optimization: iterate all_edges and check if in edges_set
        # This is O(E_total) = 120 checks.
        non_edges = []
        for e in all_edges:
            k = frozenset(e)
            if k not in edges_set:
                non_edges.append(e)
                
        if not non_edges:
            temp *= cooling
            continue
            
        e_add = random.choice(non_edges)
        
        # Perform swap
        new_edges_set = set(edges_set)
        new_edges_set.remove(e_remove)
        new_edges_set.add(frozenset(e_add))
        
        if len(new_edges_set) != target_edges:
            temp *= cooling
            continue
            
        new_adj = _build_adjacency(n, [list(e) for e in new_edges_set])
        new_score, new_worst_tris = _get_score_and_worst(n, new_adj)
        
        if new_score <= current_score:
            current_score = new_score
            edges = [list(e) for e in new_edges_set]
            adj = new_adj
            worst_tris = new_worst_tris
            if current_score < best_score:
                best_score = current_score
                best_edges = edges[:]
        else:
            delta = new_score - current_score
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_score = new_score
                edges = [list(e) for e in new_edges_set]
                adj = new_adj
                worst_tris = new_worst_tris
                
        temp *= cooling
        
    return {'n': n, 'edges': best_edges}
