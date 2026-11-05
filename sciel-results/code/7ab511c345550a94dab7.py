import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with K4,12
    L = list(range(4))
    R = list(range(4, 16))
    edges = [[u, v] for u in L for v in R]
    
    # We represent the graph as a set of frozensets for fast lookup
    edge_set = set(frozenset(e) for e in edges)
    
    # Q3 edges in canonical form (0..7)
    # Bipartition: {0,1,2,3} and {4,5,6,7}
    q3_canonical_edges = [
        (0, 4), (0, 5), (0, 6),
        (1, 4), (1, 5), (1, 7),
        (2, 4), (2, 6), (2, 7),
        (3, 5), (3, 6), (3, 7)
    ]

    def find_q3_in_subset(subset):
        """
        Check if the induced subgraph on 'subset' (list of 8 vertices) contains Q3.
        Returns list of Q3 edges in original vertex IDs if found, else None.
        """
        # Build adjacency set for the induced subgraph using local indices 0..7
        induced_edges = set()
        for i in range(8):
            for j in range(i + 1, 8):
                if frozenset((subset[i], subset[j])) in edge_set:
                    induced_edges.add((i, j))
                    
        if len(induced_edges) < 12:
            return None
            
        # Find all independent sets of size 4 in the induced graph
        # These are candidates for one partition of the bipartite Q3
        for I in itertools.combinations(range(8), 4):
            is_ind = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if (I[i], I[j]) in induced_edges or (I[j], I[i]) in induced_edges:
                        is_ind = False
                        break
                if not is_ind:
                    break
            if not is_ind:
                continue
                
            J = [v for v in range(8) if v not in I]
            
            # Check if the bipartite graph between I and J contains Q3
            # We try all permutations of I and J
            for perm_I in itertools.permutations(I):
                for perm_J in itertools.permutations(J):
                    valid = True
                    for u, v in q3_canonical_edges:
                        # u is in 0..3, v is in 4..7
                        mapped_u = perm_I[u]
                        mapped_v = perm_J[v - 4]
                        if (mapped_u, mapped_v) not in induced_edges and (mapped_v, mapped_u) not in induced_edges:
                            valid = False
                            break
                    if valid:
                        # Construct actual edges in original graph
                        q3_actual_edges = []
                        for u, v in q3_canonical_edges:
                            orig_u = subset[perm_I[u]]
                            orig_v = subset[perm_J[v - 4]]
                            q3_actual_edges.append((orig_u, orig_v))
                        return q3_actual_edges
        return None

    def find_q3():
        vertices = list(range(16))
        for subset in itertools.combinations(vertices, 8):
            res = find_q3_in_subset(subset)
            if res:
                return res
        return None

    # Main loop
    while time.time() < deadline:
        q3_edges = find_q3()
        if q3_edges is None:
            break
            
        # Remove one edge from q3_edges
        random_edge = random.choice(q3_edges)
        edge_set.remove(frozenset(random_edge))
        
    return [list(e) for e in edge_set]
