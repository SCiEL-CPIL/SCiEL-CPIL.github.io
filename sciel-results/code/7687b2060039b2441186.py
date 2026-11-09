import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Precompute bitmask for each vertex's potential neighbors
    # adj_mask[v] will be a bitmask of neighbors of v
    
    def contains_q3(adj_masks):
        """
        Exact check for Q3 subgraph using bitmasks.
        A Q3 is an 8-vertex bipartite graph with 12 edges.
        We can check for the existence of a 4-cycle in the bipartite representation 
        or more directly: check if there exist 4 vertices A and 4 vertices B 
        such that every vertex in A is connected to exactly 2 vertices in B 
        and every vertex in B is connected to exactly 2 vertices in A, 
        forming a cube? No, that's not sufficient.
        
        A better way: Q3 is the 3-dimensional hypercube.
        It can be characterized as:
        1. It has 8 vertices.
        2. It is 3-regular.
        3. It is bipartite.
        4. Its diameter is 3.
        5. It contains 4 squares (4-cycles) that cover all edges.
        
        Actually, a simpler structural check for a specific set of 8 vertices:
        If we pick 8 vertices, we can check if the induced subgraph is isomorphic to Q3.
        There are C(16,8) = 12870 subsets. This is small enough to check if done efficiently.
        However, we want to check the *entire* graph for *any* Q3 subgraph, not just induced.
        A subgraph isomorphic to Q3 means there exists a set of 8 vertices and 12 edges among them 
        that form a Q3.
        
        Efficient detection strategy:
        Since Q3 is bipartite, any Q3 subgraph must have a bipartition (S, T) with |S|=4, |T|=4.
        For each pair of disjoint 4-sets (S, T), check if the edges between S and T 
        contain a Q3.
        The number of ways to choose S and T is C(16,4) * C(12,4) / 2? No, we can just iterate 
        all partitions of 8 vertices into two sets of 4.
        Actually, we can iterate over all 8-vertex subsets, and for each, check if it contains a Q3.
        But C(16,8) is ~12k. For each, checking if it contains a Q3 subgraph (not necessarily induced) 
        is complex.
        
        Alternative: 
        A Q3 is uniquely determined by its 4-cycle structure?
        Let's use the property that Q3 is the only 3-regular bipartite graph on 8 vertices.
        So, if we find 8 vertices that form a 3-regular bipartite subgraph, it is a Q3.
        
        How to check if a graph contains a 3-regular bipartite subgraph on 8 vertices?
        This is still hard.
        
        Let's use a different approach for the detector:
        Iterate over all possible "centers" or structures?
        
        Actually, let's look at the inspirations. 
        "Bitmask adjacency replaces set lookups in contains_q3, making each edge check ~10-50x faster"
        "robust, exact bipartite-embedding Q3 detector"
        
        A bipartite embedding detector:
        A Q3 can be seen as two sets of 4 vertices, S and T.
        The edges between S and T must form a 3-regular bipartite graph.
        There are only 3 connected 3-regular bipartite graphs on 4+4 vertices?
        No, Q3 is the *only* 3-regular bipartite graph on 8 vertices (4+4).
        Proof: A 3-regular bipartite graph on 4+4 vertices must have 12 edges.
        The complement of a 3-regular bipartite graph on 4+4 vertices is a 1-regular bipartite graph (a perfect matching).
        So, a 4+4 bipartite graph is Q3 iff its complement is a perfect matching.
        
        So, the algorithm for contains_q3:
        1. Iterate over all pairs of disjoint 4-vertex subsets S and T.
           Number of such pairs: C(16,4) * C(12,4) = 1820 * 495 = ~900,000.
           This is too many to check for *every* edge addition in a tight loop.
           
        2. Optimization:
           We only need to check if adding an edge (u, v) *creates* a Q3.
           If the graph was Q3-free before, adding (u, v) can only create a Q3 that uses the edge (u, v).
           So we only need to check Q3s that contain the edge (u, v).
           
           If (u, v) is in a Q3, then u and v are in opposite bipartition sets of that Q3 (since Q3 is bipartite).
           Let S be the set of 4 vertices including u, and T be the set of 4 vertices including v.
           u is in S, v is in T.
           The other 3 vertices in S are chosen from V \ {u, v} \ T.
           The other 3 vertices in T are chosen from V \ {u, v} \ S.
           
           This is still complex.
           
        Let's try a simpler, slightly slower detector that is fast enough.
        Since we are doing local search, we can afford a detector that takes ~0.1ms.
        
        Let's precompute the list of all 8-vertex subsets that *could* be Q3s?
        No, the structure changes.
        
        Let's go with the "complement is perfect matching" check.
        For a candidate Q3 to be formed by adding edge (u, v):
        The Q3 must contain u and v.
        Let the bipartition be (S, T) with u in S, v in T.
        S = {u} ∪ S', T = {v} ∪ T', where S' and T' are 3-sets.
        S' ⊆ V \ {u, v, T}, T' ⊆ V \ {u, v, S}.
        
        Actually, we can just check all 4-sets S containing u and all 4-sets T containing v, 
        such that S ∩ T = ∅.
        Number of S: C(14, 3) = 364.
        Number of T: C(14, 3) = 364.
        But S and T must be disjoint.
        Total pairs (S, T) with u∈S, v∈T, S∩T=∅:
        We need to choose 3 more for S and 3 more for T from the remaining 14 vertices, disjoint.
        This is equivalent to choosing 6 vertices from 14, and partitioning them into 3 for S and 3 for T.
        C(14, 6) * C(6, 3) = 3003 * 20 = 60,060.
        For each pair (S, T), we check if the edges between S and T in the current graph 
        (plus the new edge u-v) form a Q3.
        i.e., check if the bipartite graph between S and T is 3-regular.
        This means every vertex in S has degree 3 to T, and every vertex in T has degree 3 to S.
        
        This check is very fast with bitmasks.
        For a given S, T:
        For each s in S:
            neighbors_in_T = adj_mask[s] & mask_T
            if bit_count(neighbors_in_T) != 3: fail
        For each t in T:
            neighbors_in_S = adj_mask[t] & mask_S
            if bit_count(neighbors_in_S) != 3: fail
        If all pass, it's a Q3.
        
        This is ~60k checks per edge addition. Each check is ~8 bit operations.
        60k * 8 = 480k ops. In Python, this might be ~0.1-0.5 ms.
        This is feasible for a few thousand iterations.
        
        Let's implement this.
        """
        # We are checking if adding edge (u, v) creates a Q3.
        # We assume the graph was Q3-free before.
        # So we only check Q3s containing (u, v).
        
        # u and v must be in opposite partitions.
        # Case 1: u in S, v in T
        # Case 2: v in S, u in T (symmetric, but we can just check one if we iterate all S,T)
        # Actually, if we iterate all S containing u and T containing v, we cover the case where u∈S, v∈T.
        # Do we need to check u∈T, v∈S?
        # Yes, because the bipartition is not fixed. But if we just check "is there a Q3 containing u and v",
        # and we check all S containing u and T containing v, we might miss the case where u is in T and v is in S.
        # However, if there is a Q3 containing u and v, either u is in S and v in T, OR u is in T and v in S.
        # So we need to check both?
        # Actually, if we swap S and T, it's the same set of edges.
        # So checking S containing u, T containing v covers all Q3s where u and v are in opposite partitions.
        # Wait, if u and v are in the *same* partition, they cannot be in a bipartite graph like Q3?
        # No, in a bipartite graph, edges only go between partitions. So if u and v are connected, they MUST be in opposite partitions.
        # So we only need to check pairs (S, T) where u∈S, v∈T (or vice versa).
        # Since the graph is undirected, the edge (u, v) is the same.
        # So we just need to check if there exists a partition (S, T) of some 8 vertices into 4+4, 
        # with u∈S, v∈T, such that the bipartite subgraph is 3-regular.
        
        # To be safe, let's just check both orientations? No, it's the same condition.
        # If (S, T) works with u∈S, v∈T, then (T, S) works with v∈T, u∈S.
        # The condition "bipartite subgraph is 3-regular" is symmetric.
        # So we only need to check one orientation: u∈S, v∈T.
        
        # Precompute masks for speed?
        # We will iterate over S' (3 vertices from V\{u,v}) and T' (3 vertices from V\{u,v}\S').
        
        # Optimization: Precompute list of all 3-subsets of V\{u,v}?
        # V\{u,v} has 14 vertices. C(14,3) = 364.
        # For each S', T' must be a 3-subset of V\{u,v}\S'.
        
        # Let's just iterate.
        
        remaining = [x for x in range(n) if x != u and x != v]
        
        # Generate all 3-subsets of remaining
        # There are 364.
        # For each S', generate all 3-subsets T' of remaining \ S'.
        
        # To make this fast, we can precompute the list of 3-subsets for any 14-set?
        # No, the set changes slightly.
        
        # Let's just use itertools.combinations.
        
        for s_prime in itertools.combinations(remaining, 3):
            S = frozenset([u] + list(s_prime))
            mask_S = 0
            for x in S:
                mask_S |= (1 << x)
            
            # T must be chosen from remaining \ S
            rem_for_t = [x for x in remaining if x not in S]
            
            for t_prime in itertools.combinations(rem_for_t, 3):
                T = frozenset([v] + list(t_prime))
                mask_T = 0
                for x in T:
                    mask_T |= (1 << x)
                
                # Check if bipartite graph between S and T is 3-regular
                # Note: We must include the edge (u, v) in the check.
                # But adj_masks does NOT include (u, v) yet.
                # So when checking degrees, we must account for (u, v).
                
                # For u in S:
                # deg(u, T) = bit_count(adj_mask[u] & mask_T) + 1 (for edge u-v)
                # For v in T:
                # deg(v, S) = bit_count(adj_mask[v] & mask_S) + 1 (for edge u-v)
                # For other s in S:
                # deg(s, T) = bit_count(adj_mask[s] & mask_T)
                # For other t in T:
                # deg(t, S) = bit_count(adj_mask[t] & mask_S)
                
                # Check u
                if (adj_masks[u] & mask_T).bit_count() + 1 != 3:
                    continue
                # Check v
                if (adj_masks[v] & mask_S).bit_count() + 1 != 3:
                    continue
                
                ok = True
                for s in S:
                    if s == u:
                        continue
                    if (adj_masks[s] & mask_T).bit_count() != 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                for t in T:
                    if t == v:
                        continue
                    if (adj_masks[t] & mask_S).bit_count() != 3:
                        ok = False
                        break
                
                if ok:
                    return True
        
        return False

    # State: adj_masks (list of 16 ints), edge_set (set of tuples)
    
    def init_graph():
        """Initialize a dense graph."""
        # Start with a random dense graph, or a specific structure.
        # Let's start with a random graph with p=0.6
        edges = set()
        masks = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.6:
                    edges.add((u, v))
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
        return edges, masks

    def add_edge(edges, masks, u, v):
        if (u, v) in edges:
            return False
        if u > v:
            u, v = v, u
        edges.add((u, v))
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        return True

    def remove_edge(edges, masks, u, v):
        if (u, v) not in edges:
            return False
        if u > v:
            u, v = v, u
        edges.remove((u, v))
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        return True

    # Local Search
    best_edges = set()
    best_masks = [0] * n
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 100
    max_iter_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        edges, masks = init_graph()
        
        # Remove all Q3s to get a valid starting point
        # This might be slow. Let's just start with a valid graph?
        # Or check and remove.
        # Since p=0.6, it's likely to have Q3s.
        
        # Fast removal:
        # While contains_q3_anywhere: remove an edge from a Q3.
        # But contains_q3_anywhere is expensive.
        
        # Alternative: Start with a known Q3-free graph?
        # K3,13 is Q3-free.
        # Let's start with K3,13 and add random edges?
        # Or start with a random sparse graph and add edges greedily?
        
        # Let's try: Start with empty graph, add edges greedily in random order.
        # This guarantees Q3-free.
        
        edges = set()
        masks = [0] * n
        
        # Randomize edge order
        edge_list = list(all_edges)
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if time.time() > deadline:
                break
            # Check if adding (u, v) creates a Q3
            if not contains_q3(masks): # Wait, contains_q3 above is for *adding* an edge.
                # We need a check that says "is (u, v) safe to add?"
                # My contains_q3(masks) above is named confusingly.
                # It checks if adding (u, v) creates a Q3, assuming (u, v) is NOT in masks.
                # So if it returns False, it is safe.
                add_edge(edges, masks, u, v)
        
        # Now we have a maximal Q3-free graph (with respect to this random order).
        # Now do local search: try to swap edges or add/remove to improve.
        # Since it's maximal, we can't add any edge.
        # We can only try to remove one edge and add two? Or just restart.
        
        # Actually, random greedy construction often gets stuck in local optima.
        # Let's try a different strategy:
        # Start with K3,13.
        # Try to add edges from the "missing" set (edges within the 3-set, within the 13-set, or between them if not already there).
        # K3,13 has edges between {0,1,2} and {3..15}.
        # Missing edges: within {0,1,2} (3 edges), within {3..15} (78 edges).
        
        # Let's use a hybrid:
        # 1. Start with K3,13.
        # 2. Randomly shuffle the missing edges.
        # 3. Try to add them one by one.
        # 4. If stuck, try to remove a random edge and retry?
        
        # Let's implement a simple hill-climbing on the number of edges.
        # Move: Remove edge e1, add edge e2. If count increases, accept.
        # But count is fixed if we swap.
        # So we need to find a way to add an edge by removing fewer edges.
        # E.g., remove 1 edge, add 2 edges.
        
        # This is complex. Let's stick to random restarts with greedy construction.
        # To improve, we can try different initial densities or orders.
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_masks = masks[:]
            
    # Return the best graph found
    return [[u, v] for u, v in best_edges]

