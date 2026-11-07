"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. Use greedy packing to form cliques of size 3: {v, u, w} where v in I, u,w in K.
       This consumes 2 cross edges and 1 internal edge.
    3. Use cliques of size 2 for remaining cross edges {v, u}.
    4. Use cliques of size 2 for remaining internal edges {u, w}.
    
    This ensures edge-disjointness and minimizes clique count by maximizing size-3 cliques.
    """
    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency lists
        # K vertices: 0 to k-1
        # I vertices: k to n-1
        # Cross edges: v in I, u in K
        neighbors_in_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_in_I[v] = nbrs
        
        # Track available internal edges in K
        # Internal edges are pairs (u, w) with u < w, u,w in [0, k)
        # We use a set of frozensets or tuples to track availability
        available_internal_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                available_internal_edges.add((u, w))
        
        cliques = []
        
        # Step 1: Greedy packing of size-3 cliques
        # For each v in I, try to form as many {v, u, w} as possible
        for v in range(k, n):
            nbrs = neighbors_in_I[v][:]  # Copy to modify
            i = 0
            while i < len(nbrs):
                u = nbrs[i]
                found_pair = False
                j = i + 1
                while j < len(nbrs):
                    w = nbrs[j]
                    # Check if internal edge (u, w) is available
                    edge_key = (u, w) if u < w else (w, u)
                    if edge_key in available_internal_edges:
                        # Form clique {v, u, w}
                        cliques.append([v, u, w])
                        # Consume the internal edge
                        available_internal_edges.remove(edge_key)
                        # Consume u and w from nbrs list
                        # Remove w from list, keep u? No, both are consumed for this v.
                        # We need to remove u and w from the list of available neighbors for v.
                        # Since we are iterating, it's easier to build a new list of remaining neighbors.
                        remaining = []
                        for x in nbrs:
                            if x != u and x != w:
                                remaining.append(x)
                        nbrs = remaining
                        i = 0
                        found_pair = True
                        break
                    j += 1
                
                if not found_pair:
                    # No pair for u. Move to next u.
                    # We will handle u in the next step (size-2 cliques)
                    i += 1
        
        # Step 2: Cover remaining cross edges with size-2 cliques
        # We need to know which cross edges were NOT covered.
        # The covered cross edges are those in the size-3 cliques.
        # It is easier to re-iterate and check, or track covered cross edges.
        # Let's track covered cross edges.
        covered_cross = set()
        for c in cliques:
            if len(c) == 3:
                v = c[0]
                u, w = c[1], c[2]
                covered_cross.add((v, u))
                covered_cross.add((v, w))
        
        for v in range(k, n):
            for u in neighbors_in_I[v]:
                if (v, u) not in covered_cross:
                    cliques.append([v, u])
        
        # Step 3: Cover remaining internal edges with size-2 cliques
        for edge in available_internal_edges:
            u, w = edge
            cliques.append([u, w])
            
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
    
    return result

# EVOLVE-BLOCK-END
