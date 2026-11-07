"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    all_results = []
    
    for n, k, p, seed in specs:
        rng = Random(seed)
        
        # 1. Add the main clique K = {0, ..., k-1}
        # This covers all edges within the first k vertices.
        cliques = [list(range(k))]
        
        # 2. For each vertex v in the independent set I = {k, ..., n-1},
        #    find its neighbors in K.
        #    The edge generation order is:
        #    for v in range(k, n):
        #        for u in range(k):
        #            if rng.random() < p: add edge (u, v)
        
        for v in range(k, n):
            neighbors = []
            for u in range(k):
                if rng.random() < p:
                    neighbors.append(u)
            
            if neighbors:
                # Create a clique consisting of v and all its neighbors in K.
                # This is a valid clique because neighbors are in K (a clique) 
                # and v is connected to all of them.
                # This covers all edges incident to v.
                cliques.append([v] + neighbors)
                
        all_results.append(cliques)
        
    return all_results
# EVOLVE-BLOCK-END
