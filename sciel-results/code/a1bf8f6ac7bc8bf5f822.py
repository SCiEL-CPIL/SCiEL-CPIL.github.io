"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for quick lookup
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Represent edges as a set of frozensets for O(1) lookup and removal
    # We need to track which edges are "covered"
    # A biclique is defined by (L, R). An edge (u, v) is covered if u in L and v in R OR v in L and u in R.
    # Actually, the definition says "cross pair must be a host edge".
    # So for a biclique [L, R], all pairs (u, v) with u in L, v in R must be in the graph.
    # And the set of these pairs must be disjoint for different bicliques.
    
    remaining_edges = set()
    for u, v in edges:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
            
    bicliques = []
    
    # Greedy strategy:
    # 1. Find a large biclique in the remaining edges.
    # 2. Remove those edges from remaining_edges.
    # 3. Repeat until no edges remain.
    
    def find_max_biclique(rem_edges):
        """
        Finds a maximal biclique within the subgraph induced by rem_edges.
        Uses a greedy heuristic: pick a vertex with high degree in rem_edges,
        then expand.
        """
        if not rem_edges:
            return None, None
            
        # Build a temporary adjacency list for the remaining edges
        temp_adj = defaultdict(set)
        for e in rem_edges:
            u, v = tuple(e)
            temp_adj[u].add(v)
            temp_adj[v].add(u)
            
        # Pick a starting vertex with high degree
        if not temp_adj:
            return None, None
            
        start_u = max(temp_adj.keys(), key=lambda x: len(temp_adj[x]))
        
        # Initialize L and R
        L = {start_u}
        # R is the set of neighbors of all vertices in L within rem_edges
        # Start with neighbors of start_u
        R = set(temp_adj[start_u])
        
        # Expand iteratively
        changed = True
        while changed:
            changed = False
            # New vertices that can be added to L:
            # A vertex x can be added to L if it is connected to ALL vertices in R.
            # Let's find candidates for L: vertices not in L that are connected to all in R.
            # Actually, standard biclique expansion:
            # L_new = {x not in L | N(x) in rem_edges contains R}
            # R_new = {y not in R | N(y) in rem_edges contains L}
            
            # Efficiently:
            # Common neighbors of L in the remaining graph.
            # Let's just compute the common neighbors of current R in the remaining graph to add to L.
            # And common neighbors of current L to add to R.
            
            # Optimization: 
            # Candidate L vertices must be connected to every node in R.
            # Candidate R vertices must be connected to every node in L.
            
            # Let's try a simple iterative expansion.
            # Since n is small (30), we can afford some checks.
            
            # Update R: vertices connected to all in L
            new_R = set()
            for v in range(n):
                if v in L:
                    continue
                # Check if v is connected to all in L
                is_connected_to_all_L = True
                for u in L:
                    if frozenset((u, v)) not in rem_edges:
                        is_connected_to_all_L = False
                        break
                if is_connected_to_all_L:
                    new_R.add(v)
            if new_R:
                R.update(new_R)
                changed = True
                
            # Update L: vertices connected to all in R
            new_L = set()
            for u in range(n):
                if u in L:
                    continue
                is_connected_to_all_R = True
                for v in R:
                    if frozenset((u, v)) not in rem_edges:
                        is_connected_to_all_R = False
                        break
                if is_connected_to_all_R:
                    new_L.add(u)
            if new_L:
                L.update(new_L)
                changed = True
                
        # Ensure L and R are non-empty
        if not L or not R:
             # Fallback: pick a single edge
             e = next(iter(rem_edges))
             u, v = tuple(e)
             return {u}, {v}
             
        return L, R

    while remaining_edges:
        if time.time() > deadline:
            break
            
        L, R = find_max_biclique(remaining_edges)
        
        if L is None or R is None:
            # Should not happen if remaining_edges is not empty
            break
            
        # Remove edges covered by this biclique
        edges_to_remove = set()
        for u in L:
            for v in R:
                edges_to_remove.add(frozenset((u, v)))
                
        # Only remove edges that actually exist in remaining_edges
        remaining_edges.difference_update(edges_to_remove)
        
        bicliques.append([list(L), list(R)])
        
    # Local Search: Try to merge bicliques
    # Two bicliques B1=[L1, R1] and B2=[L2, R2] can be merged into B=[L1 U L2, R1 U R2]
    # IF:
    # 1. All cross edges between (L1 U L2) and (R1 U R2) are in the original graph.
    # 2. The union of their edge sets is exactly the set of all cross edges in the new biclique.
    #    Wait, the partition requirement means the edge sets must be disjoint.
    #    If we merge, the new edge set is E(L1 U L2, R1 U R2).
    #    This set must be equal to E(L1, R1) U E(L2, R2) U E(L1, R2) U E(L2, R1).
    #    For the partition to hold, E(L1, R2) and E(L2, R1) must be empty (no edges) OR
    #    they must have been covered by other bicliques? No, the partition covers ALL edges exactly once.
    #    So if we merge B1 and B2, the new biclique covers E(L1, R1) U E(L2, R2) U E(L1, R2) U E(L2, R1).
    #    The original bicliques covered E(L1, R1) and E(L2, R2).
    #    The edges in E(L1, R2) and E(L2, R1) were previously covered by OTHER bicliques (or not covered? No, all edges are covered).
    #    So if we merge, we are "stealing" edges from other bicliques. This breaks the partition unless those other bicliques are also merged or adjusted.
    #    Therefore, simple pairwise merging is dangerous and complex.
    #    
    #    Alternative Local Search: "Swap" or "Reassign" vertices?
    #    Given the complexity, let's stick to the greedy result but try a different greedy order?
    #    Or try to find a *smaller* number of bicliques by using a different heuristic for the initial decomposition.
    
    # Let's try a second pass with a different heuristic: 
    # Instead of max biclique, what if we try to cover edges with bicliques that are "dense" in terms of edge count?
    # The current heuristic already maximizes the size of L and R, which tends to maximize edges.
    
    # Let's try to optimize by removing redundant vertices from the bicliques?
    # If a vertex u in L has no edges to any vertex in R in the *remaining* graph (before removal), it's fine.
    # But wait, the biclique is defined on the *original* graph edges.
    # The condition "Every cross pair must be a host edge" refers to the original graph.
    # The condition "cross-edge sets must partition all pinned graph edges" refers to the sets of edges.
    
    # Is it possible to reduce the number of bicliques by merging?
    # Consider B1=[{u}, {v1, v2}] and B2=[{w}, {v1, v2}].
    # Edges: (u,v1), (u,v2), (w,v1), (w,v2).
    # Merged B=[{u,w}, {v1,v2}].
    # Cross edges: (u,v1), (u,v2), (w,v1), (w,v2).
    # If all these edges exist in the original graph, and no other biclique covered any of them,
    # then we can merge.
    # How to check if "no other biclique covered any of them"?
    # We can check if the edge sets of B1 and B2 are disjoint (they are, by construction of the greedy loop)
    # and if the "new" cross edges (u,v1), (u,v2) etc were NOT covered by other bicliques.
    # Actually, if we just constructed the partition, every edge is covered by exactly one biclique.
    # If we merge B1 and B2, the new biclique covers the union of their edges PLUS the cross edges between L1/R2 and L2/R1.
    # If any edge in (L1 x R2) or (L2 x R1) exists in the graph, it MUST have been covered by some biclique Bk (k != 1,2).
    # If Bk is not B1 or B2, we cannot simply merge B1 and B2 without touching Bk.
    # However, if (L1 x R2) and (L2 x R1) contain NO edges from the original graph, then the union of edge sets is exactly the cross edges of the merged biclique.
    # In that case, we can merge B1 and B2 into one biclique, reducing the count by 1.
    
    # Let's implement this merge check.
    # Precompute original edge set for O(1) lookup.
    orig_edges_set = set()
    for u, v in edges:
        orig_edges_set.add(frozenset((u, v)))
        
    # Build a map: edge -> index of biclique that covers it
    edge_to_biclique = {}
    for i, (L, R) in enumerate(bicliques):
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in orig_edges_set:
                    # It must be here because of partition property
                    edge_to_biclique[e] = i
                    
    # Try to merge bicliques
    merged = False
    while merged and time.time() < deadline:
        merged = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Check if cross edges between L1/R2 and L2/R1 exist in original graph
                # If they do, they are covered by other bicliques, so we can't merge.
                # If they don't exist, we can merge.
                can_merge = True
                for u in L1:
                    for v in R2:
                        if frozenset((u, v)) in orig_edges_set:
                            can_merge = False
                            break
                    if not can_merge:
                        break
                if not can_merge:
                    continue
                for u in L2:
                    for v in R1:
                        if frozenset((u, v)) in orig_edges_set:
                            can_merge = False
                            break
                    if not can_merge:
                        break
                        
                if can_merge:
                    # Merge j into i
                    new_L = list(set(L1) | set(L2))
                    new_R = list(set(R1) | set(R2))
                    bicliques[i] = [new_L, new_R]
                    bicliques.pop(j)
                    merged = True
                    break # Restart inner loop as indices changed
            if merged:
                break # Restart outer loop
                
    # Final validation and conversion
    # Ensure types are correct
    result = []
    for L, R in bicliques:
        # Convert to lists of ints
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result

# EVOLVE-BLOCK-END
