import time
import itertools

# Precompute Q3 edges
Q3_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]
Q3_EDGE_SET = set(frozenset(e) for e in Q3_EDGES)

def find_q3_in_subset(subset, graph_edges):
    """
    Returns a list of edges (u,v) forming a Q3 in the induced subgraph on subset, or None.
    """
    # Degree check: each vertex in Q3 has degree 3
    for u in subset:
        deg = 0
        for v in subset:
            if u != v:
                e = (min(u,v), max(u,v))
                if e in graph_edges:
                    deg += 1
        if deg < 3:
            return None

    mapping = {}
    used = set()
    
    def backtrack(q3_idx):
        if q3_idx == 8:
            # Verify all edges
            q3_edges_found = []
            for u, v in Q3_EDGES:
                mu = mapping[u]
                mv = mapping[v]
                e = (min(mu, mv), max(mu, mv))
                if e not in graph_edges:
                    return None
                q3_edges_found.append(e)
            return q3_edges_found
        
        for i, v in enumerate(subset):
            if i in used:
                continue
            
            # Pruning: Check edges to already mapped vertices
            valid = True
            for u in range(q3_idx):
                if (u, q3_idx) in Q3_EDGE_SET or (q3_idx, u) in Q3_EDGE_SET:
                    mu = mapping[u]
                    e = (min(mu, v), max(mu, v))
                    if e not in graph_edges:
                        valid = False
                        break
            if valid:
                mapping[q3_idx] = v
                used.add(i)
                res = backtrack(q3_idx + 1)
                if res is not None:
                    return res
                used.remove(i)
                del mapping[q3_idx]
        return None

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    edges_set = set()
    for i in range(n):
        for j in range(i+1, n):
            edges_set.add((i, j))
            
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # Greedy removal
    while time.time() < deadline:
        found = False
        for subset in all_subsets:
            if time.time() > deadline:
                break
            q3_edges = find_q3_in_subset(subset, edges_set)
            if q3_edges is not None:
                found = True
                # Remove the first edge found in the Q3
                e_to_remove = q3_edges[0]
                edges_set.remove(e_to_remove)
                break
        
        if not found:
            break
            
    result = [[u, v] for u, v in edges_set]
    return result
