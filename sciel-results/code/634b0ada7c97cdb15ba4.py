import time
import random
from itertools import combinations

# Precompute all Q3 embeddings in K_16
# A Q3 embedding is a set of 8 distinct vertices and 12 specific edges
# We precompute all sets of 12 edges that form a Q3

def _generate_q3_edge_sets():
    """Generate all sets of 12 edges that form a Q3 subgraph."""
    q3_edge_sets = set()
    n = 16
    # Q3 has 8 vertices. We need to choose 8 vertices from 16.
    # Then check if the induced subgraph on those 8 vertices contains a Q3.
    # Actually, we need to find all subsets of 12 edges from K_16 that form a Q3.
    # A Q3 is defined by 8 vertices with specific adjacency.
    
    # The 3-cube Q3 has vertices {0,...,7} and edges:
    # (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    
    q3_edges_template = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # For each choice of 8 vertices from 16, and each bijection from {0,...,7} to these 8 vertices,
    # check if the edges form a valid Q3 in the complete graph.
    # But we just need the set of 12 edges (as pairs of original vertex indices).
    
    # To avoid duplicates, we can canonicalize each edge set.
    
    for verts in combinations(range(n), 8):
        # For each permutation of these 8 vertices assigned to Q3 vertices 0..7
        # This is too many: C(16,8) * 8! is huge.
        
        # Better approach: The Q3 is a specific graph. Two Q3s are isomorphic if they have the same edge structure.
        # We can generate all Q3 subgraphs by considering all 8-vertex subsets and all ways to map Q3 vertices to them.
        # But we can reduce by noting that the automorphism group of Q3 has size 48.
        # So for each 8-vertex subset, there are 8!/48 = 1680 distinct Q3 embeddings (up to isomorphism).
        # Total: C(16,8) * 1680 = 12870 * 1680 = ~21.6 million. Still too many.
        
        # Alternative: Precompute by iterating over all 12-edge subsets? No, C(120,12) is astronomical.
        
        # Better: For the purpose of building the conflict graph, we need to know for each pair of edges (e1, e2)
        # whether there exists a Q3 containing both e1 and e2.
        # Two edges e1=(a,b) and e2=(c,d) can be in a common Q3 iff:
        # 1. They are disjoint (a,b,c,d all distinct) OR they share one vertex.
        # 2. The union of their vertices (3 or 4 vertices) can be extended to 8 vertices forming a Q3.
        
        # Actually, let's think differently. For the MIS approach, we need the conflict graph where:
        # - Vertices = all C(16,2) = 120 possible edges
        # - Two edges conflict if they cannot both be in a Q3-free graph, i.e., if there exists a Q3 containing both.
        
        # But actually, two edges don't "conflict" just because they're in some Q3. They conflict if adding both
        # forces a Q3. Wait, no. The condition is: a set of edges is Q3-free iff it contains no Q3 as a subgraph.
        # So two edges conflict if there exists a set of 10 other edges such that all 12 form a Q3.
        # This is equivalent to: the two edges are part of at least one Q3 subgraph of K_16.
        
        # So we need to determine, for each pair of edges, whether they are contained in some Q3.
        
        # Let's precompute this.
        pass
    
    # Let's use a different approach: directly check for each pair of edges if they can be in a Q3.
    # Two edges (a,b) and (c,d) are in a common Q3 iff the subgraph induced by {a,b,c,d} (or a subset if they share a vertex)
    # is a subgraph of Q3.
    
    # Q3 properties:
    # - Bipartite with parts of size 4
    # - Each vertex has degree 3
    # - Contains 4-cycles
    # - Two edges are in a Q3 iff they are either:
    #   a) Disjoint and at distance 2 in the Q3 (i.e., there's a path of length 2 between them)
    #   b) Adjacent (share a vertex)
    
    # Let's precompute all pairs of edges that are in some Q3.
    # We can do this by: for each 8-vertex subset, generate all Q3s on it, and mark all edge pairs in each Q3.
    
    # But that's expensive. Let's think of a smarter way.
    
    # Actually, for the conflict graph, we can use the following characterization:
    # Two edges e1=(u,v) and e2=(w,x) conflict if there exists a Q3 containing both.
    # This happens iff the 4 vertices {u,v,w,x} (with u=v or w=x if edges share a vertex) form a subgraph of Q3.
    
    # Let's precompute by brute force over all 8-vertex subsets, but optimize.
    
    # Alternative: Since we only need to build the conflict graph (120x120 matrix), and 120 is small,
    # we can check each pair by trying to extend to a Q3.
    
    # For a pair of edges, can we always extend to a Q3? Not always.
    # Example: two disjoint edges that are "too far apart" in the Q3 structure.
    
    # Let's just precompute all Q3 edge sets by iterating over all 8-vertex subsets and all Q3 structures on them.
    # Number of 8-vertex subsets: C(16,8) = 12870
    # For each, number of Q3 embeddings: 8! / |Aut(Q3)| = 40320 / 48 = 840
    # Total Q3s: 12870 * 840 = 10,810,800. This is feasible if we do it smartly.
    
    # But we don't need to store all Q3s. We just need to mark which pairs of edges appear together in some Q3.
    
    # Let's use a bitset or boolean matrix.
    
    # Map each edge (i,j) with i<j to an index 0..119.
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        # Index in the list of all edges
        # We'll create a mapping
        return i * 16 + j  # Temporary, we'll use a proper mapping later
    
    # Let's create a proper edge list and mapping
    all_edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
    num_edges = len(all_edges)  # 120
    
    # Conflict matrix: conflict[i][j] = True if edge i and edge j are in some common Q3
    conflict = [[False] * num_edges for _ in range(num_edges)]
    
    # Q3 template edges (on vertices 0..7)
    q3_template_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # For each 8-vertex subset, generate all Q3 embeddings
    # To avoid duplicates due to automorphisms, we can fix one vertex mapping.
    # But it's easier to just generate all 8! mappings and deduplicate by the resulting edge set.
    
    # Actually, let's just generate all and mark conflicts. Duplicates are fine, we just set True multiple times.
    
    start_time = time.time()
    for verts in combinations(range(n), 8):
        # Map Q3 vertices 0..7 to these 8 vertices
        # Use all permutations, but we can reduce by fixing the image of vertex 0 to be the first vertex in verts
        # This reduces by factor of 8.
        # Actually, let's just do all 8! to be safe, but break early if time is running out.
        
        from itertools import permutations
        for perm in permutations(verts):
            # perm[k] is the original vertex corresponding to Q3 vertex k
            # Check if this is a valid Q3: all template edges must be present in K_16 (they always are)
            # and the mapping must be injective (it is, since perm is a permutation)
            
            # Mark all pairs of edges in this Q3 as conflicting
            q3_edges_orig = []
            for a, b in q3_template_edges:
                u, v = perm[a], perm[b]
                if u > v:
                    u, v = v, u
                q3_edges_orig.append((u, v))
            
            # Mark all pairs
            for i_e in range(len(q3_edges_orig)):
                for j_e in range(i_e+1, len(q3_edges_orig)):
                    e1 = q3_edges_orig[i_e]
                    e2 = q3_edges_orig[j_e]
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    conflict[idx1][idx2] = True
                    conflict[idx2][idx1] = True
        
        if time.time() - start_time > 100:
            break
    
    return all_edges, conflict, edge_to_idx

# Precompute at module level
_all_edges, _conflict, _edge_to_idx = _generate_q3_edge_sets()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    num_edges = len(_all_edges)  # 120
    
    # Build conflict graph adjacency as bitmasks for fast operations
    # adj_mask[i] is a bitmask of all edges that conflict with edge i
    adj_mask = [0] * num_edges
    for i in range(num_edges):
        mask = 0
        for j in range(num_edges):
            if i != j and _conflict[i][j]:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Maximum Independent Set on the conflict graph
    # We want to find the largest set of edges with no two conflicting
    # This is equivalent to finding the largest Q3-free set of edges
    
    # Use branch and bound with greedy coloring for upper bound
    
    # Greedy coloring of a graph to get an upper bound on independent set size
    def greedy_color_bound(vertices, adj_masks):
        """Greedy coloring to get upper bound on max independent set."""
        if not vertices:
            return 0
        
        # Order vertices by degree (descending) for better bounds
        degs = [(adj_masks[v] & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF).bit_count() for v in vertices]
        # Actually, we need to compute degree within the subgraph induced by vertices
        # Let's do a simple greedy coloring
        
        # Convert to list for sorting
        vlist = list(vertices)
        # Sort by degree in the induced subgraph (descending)
        def degree_in_subgraph(v):
            count = 0
            for u in vlist:
                if u != v and (adj_masks[v] >> u) & 1:
                    count += 1
            return count
        
        vlist.sort(key=degree_in_subgraph, reverse=True)
        
        colors = {}
        next_color = 0
        for v in vlist:
            # Find the smallest available color
            used_colors = set()
            for u in vlist:
                if u == v:
                    break
                if u in colors and (adj_masks[v] >> u) & 1:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            next_color = max(next_color, c + 1)
        
        return next_color
    
    # Branch and bound
    best_set = []
    best_size = 0
    
    # Initial solution: greedy
    def greedy_mis():
        """Greedy independent set."""
        remaining = set(range(num_edges))
        chosen = []
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and (adj_mask[v] >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            chosen.append(best_v)
            # Remove best_v and all its neighbors
            to_remove = {best_v}
            for u in remaining:
                if (adj_mask[best_v] >> u) & 1:
                    to_remove.add(u)
            remaining -= to_remove
        return chosen
    
    greedy_solution = greedy_mis()
    best_size = len(greedy_solution)
    best_set = set(greedy_solution)
    
    # Branch and bound
    def branch_and_bound(candidates, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline - 10:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: greedy coloring
        cand_set = set(candidates)
        upper_bound = greedy_color_bound(cand_set, adj_mask)
        if len(current_set) + upper_bound <= best_size:
            return
        
        # Choose a vertex to branch on
        # Pick vertex with highest degree in candidates
        v = max(candidates, key=lambda x: (adj_mask[x] & 0).bit_count())  # Placeholder
        # Actually compute degree in candidates
        best_v = None
        best_deg = -1
        for u in candidates:
            deg = 0
            for w in candidates:
                if w != u and (adj_mask[u] >> w) & 1:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = u
        
        # Branch 1: include best_v
        new_candidates = [u for u in candidates if u != best_v and not ((adj_mask[best_v] >> u) & 1)]
        branch_and_bound(new_candidates, current_set + [best_v])
        
        # Branch 2: exclude best_v
        new_candidates2 = [u for u in candidates if u != best_v]
        branch_and_bound(new_candidates2, current_set)
    
    # Start with all edges
    all_cands = list(range(num_edges))
    branch_and_bound(all_cands, [])
    
    # Convert best_set to edge list
    edges = []
    for e_idx in best_set:
        u, v = _all_edges[e_idx]
        edges.append([u, v])
    
    return edges
