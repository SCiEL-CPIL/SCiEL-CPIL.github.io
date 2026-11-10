"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency for the independent set part (vertices k..n-1)
        # We need to know for each i in [k, n) which vertices in [0, k) it is connected to.
        
        # 1. Create the main clique for the clique part C = {0, ..., k-1}
        # This covers all edges (u, v) where u, v < k.
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Identify cross edges (u, v) where u < k and v >= k
        # For each v in the independent set, collect its neighbors in C.
        # We can do this by iterating through the edge list.
        
        # Efficiently map vertex in I to its neighbors in C
        # Note: edges are sorted. 
        # We can just use a dictionary or list of sets.
        
        neighbors_in_c = {v: set() for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_c[v].add(u)
            elif v < k and u >= k:
                # Edges are generated as (u, v) with u < v in the first part, 
                # but cross edges are (u, v) where u in [0,k), v in [k,n).
                # The generator does: edges += [(u, v) for v in range(k, n) for u in range(k) if ...]
                # So u is always the smaller index (in C) and v is the larger index (in I).
                # So the first case covers all cross edges.
                pass

        # 3. For each vertex v in the independent set that has neighbors,
        #    create a clique consisting of v and all its neighbors in C.
        #    This clique covers all edges incident to v exactly once.
        #    It is a valid clique because all neighbors are in C (which is a clique) 
        #    and all are connected to v.
        
        for v in range(k, n):
            if time.time() > deadline:
                break # Safety guard, though this loop is small
            nbrs = neighbors_in_c[v]
            if len(nbrs) >= 1:
                # The clique is nbrs U {v}
                # Ensure it's a list of ints
                clique = list(nbrs) + [v]
                # Sort for consistency, though not strictly required by "list of int vertices"
                # but good practice.
                clique.sort()
                # Check size >= 2
                if len(clique) >= 2:
                    cliques.append(clique)
            # If len(nbrs) == 0, no edges to cover for this vertex, so we don't add a clique.
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        partition = partition_split_graph(n, k, edges)
        
        # Convert to ensure types are correct (Python ints, lists)
        # The generator returns ints, sets contain ints, list() makes them ints.
        # Just to be safe, verify types.
        clean_partition = []
        for c in partition:
            clean_c = [int(x) for x in c]
            clean_partition.append(clean_c)
            
        results.append(clean_partition)
        
        if time.time() > deadline:
            # If we run out of time, we return what we have. 
            # But we need 3 lists. If we didn't finish, this might be invalid.
            # Given the O(n) complexity of the partition step, this shouldn't happen.
            pass

    return results
# EVOLVE-BLOCK-END
