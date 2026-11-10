import itertools
import random
import time

def construct():
    n = 16
    
    # Precompute all unique Q3 edge patterns on vertices 0..7
    q3_edges_base = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor and (xor & (xor - 1)) == 0:
                q3_edges_base.append((i, j))
    
    unique_q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        mapped_edges = set()
        for u, v in q3_edges_base:
            mapped_edges.add((perm[u], perm[v]))
        norm_edges = frozenset(tuple(sorted(e)) for e in mapped_edges)
        unique_q3_patterns.add(norm_edges)
        
    unique_q3_patterns = list(unique_q3_patterns)
    
    def find_q3(edge_set):
        for vert_subset in itertools.combinations(range(n), 8):
            for P in unique_q3_patterns:
                found = True
                for u, v in P:
                    e = (vert_subset[u], vert_subset[v])
                    if e[0] > e[1]:
                        e = (e[1], e[0])
                    if e not in edge_set:
                        found = False
                        break
                if found:
                    return set((vert_subset[u], vert_subset[v]) for u, v in P)
        return None

    random.seed(42)
    
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.8:
                edges.add((u, v))
                
    max_iter = 200
    while max_iter > 0:
        q3 = find_q3(edges)
        if q3 is None:
            break
        e_to_remove = random.choice(list(q3))
        edges.remove(e_to_remove)
        max_iter -= 1
        
    all_edges = set(itertools.combinations(range(n), 2))
    missing_edges = list(all_edges - edges)
    random.shuffle(missing_edges)
    
    for e in missing_edges:
        edges.add(e)
        q3 = find_q3(edges)
        if q3 is not None:
            edges.remove(e)
            
    # Local Search
    for _ in range(5):
        if not edges:
            break
        e_to_remove = random.choice(list(edges))
        edges.remove(e_to_remove)
        
        missing_edges = list(all_edges - edges)
        random.shuffle(missing_edges)
        for e in missing_edges:
            edges.add(e)
            q3 = find_q3(edges)
            if q3 is not None:
                edges.remove(e)
                
    return [[u, v] for u, v in edges]
