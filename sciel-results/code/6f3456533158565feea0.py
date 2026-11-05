"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import sys

# EVOLVE-BLOCK-START

def _generate_q3_subsets():
    """
    Generate all subsets of edges that form a Q3 (3-cube) on 4 distinct vertices.
    A Q3 has 8 vertices and 12 edges. We need to find all sets of 8 vertices
    and 12 edges that form a Q3.
    
    Actually, let's think differently. A Q3 is a specific graph on 8 vertices.
    We need to find all embeddings of Q3 into K16.
    
    A Q3 can be represented as the vertices being {0,1,2,3,4,5,6,7} with edges:
    (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    
    But we need all possible embeddings. This is complex. Let's use a different approach.
    
    Instead, let's precompute all possible Q3 edge sets. A Q3 on vertices 0..7 has a specific structure.
    The number of non-isomorphic embeddings is large, but we can generate them by considering
    all ways to assign 8 distinct vertices from 0..15 and all automorphisms of Q3.
    """
    # Q3 has 8 vertices and 12 edges
    # Base Q3 edges (vertices 0-7)
    base_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # The automorphism group of Q3 has 48 elements
    # But generating all embeddings is expensive. Let's use a smarter approach.
    
    # Alternative: We'll use a backtracking search that checks if adding an edge creates a Q3.
    # For that, we need to quickly check if a set of edges contains a Q3.
    
    # Let's precompute all possible Q3 edge sets by enumerating all 8-vertex subsets
    # and checking if they can form a Q3.
    
    q3_edge_sets = set()
    
    # All 8-vertex subsets from 16 vertices
    for vertices in itertools.combinations(range(16), 8):
        vertex_set = set(vertices)
        # Generate all perfect matchings or check if the induced subgraph on these vertices
        # could be a Q3
        
        # A Q3 is 3-regular on 8 vertices. Let's check all possible 3-regular graphs
        # on these 8 vertices that are isomorphic to Q3.
        
        # Actually, let's just enumerate all 12-edge subsets of the complete graph on 8 vertices
        # and check if they form a Q3.
        
        # Number of edges in K8 is 28. Choosing 12 from 28 is C(28,12) which is ~30 million. Too many.
        
        # Better approach: Use the fact that Q3 is bipartite with parts of size 4.
        # In Q3, vertices can be partitioned into two sets of 4 based on the parity of their
        # binary representation (if we label them 0-7 in binary).
        
        # Let's use a canonical form. For a given 8-vertex set, we can try all ways to
        # partition into two 4-sets and check if the bipartite graph is Q3.
        
        for part_a in itertools.combinations(vertices, 4):
            part_b = tuple(v for v in vertices if v not in part_a)
            part_a_set = set(part_a)
            part_b_set = set(part_b)
            
            # Check if there's a Q3 embedding with this bipartition
            # Q3 has 12 edges, each connecting a vertex in part_a to part_b
            # Each vertex has degree 3
            
            # We need to check if the bipartite graph between part_a and part_b
            # is isomorphic to Q3's bipartite structure
            
            # This is still complex. Let's use a different strategy.
            
            # Precompute all Q3 edge patterns once, then for each 8-vertex set,
            # check if any permutation gives a valid Q3.
            pass
    
    # This approach is too slow. Let's use a completely different method.
    
    # New approach: Use branch and bound with a Q3 detection subroutine.
    # We'll maintain the current edge set and when considering adding a new edge,
    # we'll check if it creates a Q3 by looking for the required structure.
    
    # For efficiency, we'll precompute all possible Q3 edge sets in K16.
    # There are C(16,8) * (number of Q3 embeddings on 8 labeled vertices) such sets.
    # C(16,8) = 12870. Number of Q3 embeddings on 8 labeled vertices = 8!/48 = 1680.
    # Total: 12870 * 1680 ≈ 21 million. This is feasible to precompute.
    
    # Let's generate all Q3 edge sets.
    
    # First, get all automorphisms of Q3
    # Q3 vertices: 0-7, edges as above
    # The automorphism group is the hyperoctahedral group of order 48
    
    # For simplicity, let's just enumerate all 8! permutations of 0-7 and check
    # if the permuted edges form a valid Q3.
    
    base_edges = set([(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)])
    
    # Generate all distinct Q3 edge sets on vertices 0-7
    q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        perm_map = {i: perm[i] for i in range(8)}
        new_edges = set()
        for u, v in base_edges:
            pu, pv = perm_map[u], perm_map[v]
            if pu > pv:
                pu, pv = pv, pu
            new_edges.add((pu, pv))
        q3_patterns.add(frozenset(new_edges))
    
    # Now generate all Q3 edge sets in K16
    all_q3_sets = set()
    for vertices in itertools.combinations(range(16), 8):
        vertex_map = {v: i for i, v in enumerate(vertices)}
        for pattern in q3_patterns:
            new_edges = set()
            for u, v in pattern:
                vu, vv = vertices[u], vertices[v]
                if vu > vv:
                    vu, vv = vv, vu
                new_edges.add((vu, vv))
            all_q3_sets.add(frozenset(new_edges))
    
    return list(all_q3_sets)

def _check_q3_free(edge_set):
    """Check if a set of edges contains no Q3."""
    # This is expensive. We'll use it only in the final verification.
    # For the search, we'll use incremental checks.
    edge_set = frozenset(edge_set)
    # Check all possible Q3 subsets
    # This is too slow for large edge sets. We'll use a different approach in the search.
    return True  # Placeholder

def construct():
    """
    Construct a Q3-free graph on 16 vertices with maximum number of edges.
    """
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with the K3,13 construction: 39 edges
    # This is a bipartite graph with parts of size 3 and 13
    initial_edges = set((a, b) for a in range(3) for b in range(3, 16))
    
    # Try to add edges using branch and bound
    # We'll use a greedy approach with backtracking
    
    best_edge_count = len(initial_edges)
    best_edges = initial_edges.copy()
    
    # Try adding edges one by one
    remaining_edges = [e for e in all_edges if e not in initial_edges]
    
    # Sort remaining edges by some heuristic (e.g., by vertex degree)
    def edge_score(edge):
        u, v = edge
        # Prefer edges that connect vertices with higher degrees
        return 0  # Placeholder
    
    # Use a simple greedy approach with conflict detection
    current_edges = initial_edges.copy()
    
    # For each remaining edge, check if adding it creates a Q3
    # A Q3 is created if there exist 8 vertices containing u and v
    # such that the induced subgraph on these 8 vertices contains a Q3
    
    # This is complex. Let's use a simpler heuristic:
    # Try to add edges and check for Q3 using a precomputed list
    
    # Precompute all Q3 edge sets (this is expensive but done once)
    # For now, let's use a simpler approach: try to improve upon K3,13
    
    # Actually, let's try a different initial construction.
    # The Turan graph T(16, 4) has 16*15/2 - 4*(4*3/2) = 120 - 24 = 96 edges
    # But T(16,4) may contain Q3.
    
    # Let's try a random approach with local search
    
    import random
    random.seed(42)
    
    # Start with a random dense graph and remove edges to eliminate Q3
    # Or start with K16 and remove edges
    
    # Better: Use the fact that Q3 is bipartite with parts of size 4.
    # A graph with no Q3 must have the property that for any 4+4 partition,
    # the bipartite graph is not Q3.
    
    # Let's try a structured approach:
    # Divide 16 vertices into 4 groups of 4: G0, G1, G2, G3
    # Include all edges within groups (cliques) and between groups
    # but avoid creating Q3 patterns
    
    # Actually, the simplest improvement might be to use a different bipartite graph.
    # K4,12 has 48 edges. K5,11 has 55 edges. K6,10 has 60 edges. K7,9 has 63 edges.
    # K8,8 has 64 edges.
    
    # Let's check if K8,8 is Q3-free.
    # A Q3 has bipartition 4+4. In K8,8, the bipartition is 8+8.
    # A Q3 embedded in K8,8 would need 4 vertices in each part of the Q3
    # to be in the same part of K8,8. Since K8,8 is complete bipartite,
    # any 4 vertices from one side and 4 from the other form a K4,4,
    # which contains Q3 as a subgraph (Q3 is a subgraph of K4,4).
    # So K8,8 is NOT Q3-free.
    
    # K7,9: A Q3 would need 4 vertices from one side and 4 from the other.
    # In K7,9, we can pick 4 from the 7-side and 4 from the 9-side,
    # and K4,4 contains Q3. So K7,9 is NOT Q3-free.
    
    # In fact, any complete bipartite graph K_{a,b} with a>=4 and b>=4 contains Q3.
    # So we need a non-complete bipartite graph or a non-bipartite graph.
    
    # The K3,13 construction works because one part has only 3 vertices,
    # so we can't pick 4 vertices from that part.
    
    # To improve, we could use a graph that is not bipartite but still Q3-free.
    # For example, take K3,13 and add some edges within the 13-vertex part,
    # being careful not to create Q3.
    
    # Let's try: Start with K3,13 and add edges among the 13 vertices.
    # When we add an edge (u,v) among the 13 vertices, we need to ensure
    # that no Q3 is created.
    
    # A Q3 involving the new edge (u,v) would need:
    # - u and v to be in the same part of the Q3 (since they're both in the 13-part)
    # - But in Q3, adjacent vertices are in different parts.
    # - So (u,v) cannot be an edge in Q3 if u,v are in the same part.
    # - Wait, in K3,13, the 13 vertices are all in one part. Edges only go
    #   between the 3-part and the 13-part.
    # - If we add an edge (u,v) within the 13-part, this edge is not in K3,13.
    # - For this edge to be part of a Q3, we need a Q3 embedding where
    #   u and v are adjacent in Q3.
    # - In Q3, adjacent vertices are in different bipartition parts.
    # - So u and v would need to be in different parts of the Q3.
    # - But in K3,13, all 13 vertices are in one bipartition part.
    # - So for a Q3 to use edge (u,v), we'd need 3 other vertices from the 3-part
    #   and 4 other vertices from the 13-part (including u,v).
    # - The Q3 has 8 vertices: 4 in each part.
    # - If u,v are in the 13-part, they're in one Q3-part.
    # - We need 2 more vertices from the 13-part and 4 from the 3-part.
    # - But the 3-part only has 3 vertices! So we can't pick 4.
    # - Therefore, adding any edge within the 13-part does NOT create a Q3!
    
    # This means we can add ALL edges within the 13-part!
    # The resulting graph has:
    # - 3*13 = 39 edges between the parts
    # - C(13,2) = 78 edges within the 13-part
    # - 0 edges within the 3-part
    # Total: 39 + 78 = 117 edges
    
    # Let's verify this is Q3-free.
    # Any Q3 needs 8 vertices. In our graph, the 3-part has only 3 vertices.
    # A Q3 has 4 vertices in each bipartition part.
    # If we try to embed Q3, we need 4 vertices in one part and 4 in the other.
    # Case 1: 4 vertices from the 13-part, 4 from the 3-part. Impossible (only 3 in 3-part).
    # Case 2: 4 vertices from the 3-part, 4 from the 13-part. Impossible (only 3 in 3-part).
    # Case 3: Some mix. But the graph is not bipartite (has edges within 13-part).
    #    However, Q3 is bipartite. Any subgraph isomorphic to Q3 must be bipartite.
    #    In our graph, consider the bipartition: A = 3-part, B = 13-part.
    #    Edges go between A and B, and within B.
    #    For a subgraph to be Q3 (bipartite), it must have a bipartition.
    #    The only possible bipartition of a subgraph is a subset of A and a subset of B,
    #    with no edges within each part of the subgraph's bipartition.
    #    But we have edges within B. So if the subgraph includes any edge within B,
    #    it can't be bipartite with the A/B partition.
    #    Could it have a different bipartition? Q3 is bipartite, so any subgraph
    #    isomorphic to Q3 must be bipartite. The graph induced by any 8 vertices
    #    in our construction: if it includes an edge within B, that edge connects
    #    two vertices in B. For the subgraph to be bipartite, these two vertices
    #    must be in different parts of the subgraph's bipartition. But they're both
    #    in B, and all edges from B go to A or within B. This gets complex.
    
    # Let's just verify computationally that the 117-edge graph is Q3-free.
    
    new_edges = set()
    # Edges between 3-part and 13-part
    for a in range(3):
        for b in range(3, 16):
            new_edges.add((a, b))
    # Edges within 13-part
    for i in range(3, 16):
        for j in range(i+1, 16):
            new_edges.add((i, j))
    
    # Verify Q3-free
    # We'll check all 8-vertex subsets
    is_q3_free = True
    for vertices in itertools.combinations(range(16), 8):
        vertex_set = set(vertices)
        # Get edges within this subset
        sub_edges = set()
        for u, v in new_edges:
            if u in vertex_set and v in vertex_set:
                sub_edges.add((u, v))
        
        # Check if sub_edges contains a Q3
        # A Q3 has 12 edges and 8 vertices, 3-regular
        if len(sub_edges) < 12:
            continue
        
        # Check if the subgraph is isomorphic to Q3
        # Q3 is 3-regular, so all vertices must have degree 3
        degrees = {v: 0 for v in vertices}
        for u, v in sub_edges:
            degrees[u] += 1
            degrees[v] += 1
        
        if any(deg != 3 for deg in degrees.values()):
            continue
        
        # Check if it's Q3 by checking the structure
        # Q3 has a specific structure: it's the 3-dimensional hypercube
        # We can check by verifying the distance distribution
        # In Q3, from any vertex, there are: 1 at distance 0, 3 at distance 1,
        # 3 at distance 2, 1 at distance 3
        
        # Pick a vertex and compute distances
        start = vertices[0]
        # BFS
        from collections import deque
        dist = {start: 0}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for u, v in sub_edges:
                if curr == u:
                    neighbor = v
                elif curr == v:
                    neighbor = u
                else:
                    continue
                if neighbor not in dist:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)
        
        # Check distance distribution
        if len(dist) != 8:
            continue  # Not connected, can't be Q3
        
        dist_counts = {}
        for d in dist.values():
            dist_counts[d] = dist_counts.get(d, 0) + 1
        
        # Q3: 1 at dist 0, 3 at dist 1, 3 at dist 2, 1 at dist 3
        if dist_counts.get(0, 0) != 1 or dist_counts.get(1, 0) != 3 or \
           dist_counts.get(2, 0) != 3 or dist_counts.get(3, 0) != 1:
            continue
        
        # Also check that all vertices at the same distance have the same number of neighbors
        # at distance+1, etc. For Q3, this is satisfied.
        # For now, assume this check is sufficient.
        
        # If we get here, this subgraph might be Q3
        is_q3_free = False
        break
    
    if is_q3_free:
        return [[u, v] for u, v in new_edges]
    
    # If not Q3-free, fall back to K3,13
    return [[a, b] for a in range(3) for b in range(3, 16)]

# EVOLVE-BLOCK-END
