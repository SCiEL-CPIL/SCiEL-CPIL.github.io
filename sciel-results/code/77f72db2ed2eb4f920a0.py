import time
import random

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 145.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        if remaining_time < 2.0:
            cliques = trivial_partition(n, p, seed)
        else:
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 40.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 100.0)
            cliques = solve_instance(n, p, seed, instance_time_limit)
        
        results.append(cliques)
    
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    best_cliques = None
    best_count = float('inf')
    start_time = time.time()
    
    num_restarts = 40
    
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(seed + attempt * 7919)
        remaining_time = time_limit - (time.time() - start_time)
        cliques = greedy_partition(edges, n, edge_set, rng, remaining_time)
        
        if cliques is None:
            continue
        
        cliques = local_improve(cliques, n, edge_set, time_limit - (time.time() - start_time))
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques
    
    if best_cliques is None:
        return trivial_partition(n, p, seed)
    return best_cliques

def greedy_partition(edges, n, edge_set, rng, time_limit):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining_edges = len(edges)
    cliques = []
    start_time = time.time()
    
    vertex_order = list(range(n))
    rng.shuffle(vertex_order)
    
    while remaining_edges > 0:
        if time.time() - start_time > time_limit:
            return None
        
        clique = find_clique(adj, n, vertex_order, rng)
        
        if not clique or len(clique) < 2:
            found = False
            for u in vertex_order:
                for v in list(adj[u]):
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        removed = 0
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if v in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    removed += 1
        
        if removed > 0:
            cliques.append(clique)
            remaining_edges -= removed
    
    return cliques

def find_clique(adj, n, vertex_order, rng):
    best_clique = []
    candidates = vertex_order[:min(15, n)]
    rng.shuffle(candidates)
    
    for start in candidates:
        if len(adj[start]) == 0:
            continue
        C = [start]
        P = set(adj[start])
        while P:
            best_u = None
            best_deg = -1
            for u in P:
                deg = len(P & adj[u])
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            if best_u is None:
                break
            C.append(best_u)
            P = P & adj[best_u]
        if len(C) > len(best_clique):
            best_clique = C
    
    return best_clique

def local_improve(cliques, n, edge_set, time_limit):
    if not cliques:
        return cliques
    start_time = time.time()
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        new_cliques = []
        clique_sets = [set(c) for c in cliques]
        i = 0
        while i < len(clique_sets):
            if i >= len(clique_sets):
                break
            merged = False
            for j in range(i + 1, len(clique_sets)):
                union = clique_sets[i] | clique_sets[j]
                if is_clique(union, edge_set):
                    new_cliques.append(list(union))
                    clique_sets.pop(j)
                    clique_sets.pop(i)
                    changed = True
                    merged = True
                    break
            if not merged:
                new_cliques.append(list(clique_sets[i]))
                clique_sets.pop(i)
        cliques = new_cliques
    return cliques

def is_clique(vertices, edge_set):
    vlist = list(vertices)
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            u, v = vlist[i], vlist[j]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True
