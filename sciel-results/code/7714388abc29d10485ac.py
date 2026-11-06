"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix (vectorized)."""
    common = adj @ adj
    # For each pair (u,w), C(u,w) = common[u,w] - adj[u,w] (subtract the edge itself)
    c = common - adj
    # Number of C4s = sum over u<w of C(u,w)*(C(u,w)-1)/2
    # Use upper triangle to avoid double counting
    iu = np.triu_indices(common.shape[0], k=1)
    vals = c[iu]
    return int(np.sum(vals * (vals - 1) // 2))

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5) with 90 edges.
    
    For q=5 (odd), the orthogonal polarity graph of PG(2,q) has:
    - n = q^2 + q + 1 = 31 vertices
    - 90 edges (C4-free)
    """
    q = 5
    # Enumerate all 1D subspaces of F_q^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    # Use the symmetric matrix M = [[1,0,0],[0,1,0],[0,0,-1]] mod 5
    # This gives a polarity where x^T M y = 0 defines perpendicularity
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges, adj

def _delta_c4_add_edge(adj, i, j):
    """
    Compute the change in C4 count when adding edge (i,j) to graph with adjacency adj.
    New C4s = number of length-2 paths from i to j = common_neighbors(i,j)
    But we need to be careful: a C4 is i-a-b-j-i where a~i, b~j, a~b, a!=b.
    This equals the number of edges in the bipartite subgraph between N(i) and N(j),
    minus any that go through i or j themselves (but i,j aren't in their own N).
    Actually: C4s created by adding (i,j) = number of pairs (a,b) with a in N(i), b in N(j),
    a != b, and a~b. This is exactly the number of edges between N(i) and N(j)
    in the induced subgraph, excluding self-loops.
    Vectorized: sum over a in N(i) of (adj[a, N(j)] sum) minus corrections.
    """
    # Number of C4s containing new edge (i,j) = number of paths of length 2 from i to j
    # that don't use edge (i,j) itself (it doesn't exist yet)
    # = (adj @ adj)[i, j]  ... wait, that counts paths i->a->j where a is common neighbor
    # But a C4 is i-a-b-j-i, which requires a~b too. So it's NOT just common neighbors.
    # 
    # A 4-cycle through edge (i,j): i-a-b-j-i
    # a must be neighbor of i, b must be neighbor of j, a must be neighbor of b, a != b
    # Count = sum over a in N(i) of |N(a) ∩ N(j)|, but excluding b=i (i not in N(j) since no edge)
    # and excluding a=b (a in N(i), b in N(j), if a==b then a is common neighbor, but a~a is false)
    # Actually if a==b, we need a~a which is false, so those are automatically excluded.
    # So: count = sum_{a in N(i)} |N(a) ∩ N(j)|
    # = sum_{a in N(i)} (adj[a] @ adj[j])  ... no, that's sum over b of adj[a,b]*adj[j,b]
    # = (adj[i] @ adj @ adj[j])  ... let me think again
    # 
    # Let's just compute directly with vectorization:
    ni = adj[i]  # boolean/int vector of neighbors of i
    nj = adj[j]  # neighbors of j
    # For each a in N(i), count neighbors of a that are in N(j)
    # = ni @ (adj @ nj)  where adj @ nj gives for each vertex v, |N(v) ∩ N(j)|
    common_with_j = adj @ nj  # for each v, number of common neighbors with j
    count = int(ni @ common_with_j)
    # But this counts a=j? No, j is not in N(i) (no edge i-j yet). 
    # And it counts b=i? i is not in N(j) (no edge). 
    # However, it also counts the case where a is a neighbor of i AND a is in N(j) AND a~a? No.
    # Let me verify: ni @ common_with_j = sum_a ni[a] * common_with_j[a]
    # = sum_a [a in N(i)] * |N(a) ∩ N(j)|
    # For a fixed a in N(i), |N(a) ∩ N(j)| counts b such that b~a and b~j.
    # If b==i: i~a (yes, a in N(i)) and i~j (no, no edge). So b=i not counted. Good.
    # If b==a: a~a (no). Not counted. Good.
    # So this is correct.
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Build the polarity graph (C4-free, 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    n = 31
    c4_base = _count_c4(adj)  # should be 0
    
    # Strategy 1: Add single edge with minimum C4 creation
    best_single = None
    best_single_c4 = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                delta = _delta_c4_add_edge(adj, i, j)
                if delta < best_single_c4:
                    best_single_c4 = delta
                    best_single = [i, j]
    
    # Strategy 2: Edge swap — remove edge e1 from polarity graph, add edge e2
    # This keeps 91 edges. Try to find swap that minimizes total C4s.
    # After removing e1 and adding e2, the graph has 91 edges.
    # C4s in new graph = C4s not involving either e1 or e2 
    #                   - C4s involving e1 in original (none, since C4-free)
    #                   + C4s involving e2 in new graph
    #                   + C4s that involve both e1 and e2 (these are new)
    # Since original is C4-free, removing e1 doesn't remove any C4s.
    # Adding e2 creates C4s. Some of those C4s might use e1, but e1 is removed.
    # So: C4s after swap = C4s created by adding e2 to (polarity - e1)
    # = C4s created by adding e2 to polarity, minus C4s that use e1.
    # A C4 created by adding e2 uses e2 and 3 other edges. If one of those is e1, it's invalid.
    
    # Precompute: for each non-edge e2, the C4s it creates in the full polarity graph
    # and which edges those C4s use.
    
    # For efficiency, let's do a limited search of swaps.
    # First, identify candidate edges to remove and add.
    
    # Get list of all edges in polarity graph
    polarity_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                polarity_edges.add((i, j))
    
    # Get list of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                non_edges.append((i, j))
    
    # For each non-edge e2, compute which C4s it creates and the edges involved
    # C4 through e2=(i,j): i-a-b-j-i, uses edges (i,a),(a,b),(b,j)
    # We need to track these for swap analysis.
    
    # Precompute for each non-edge the list of C4s it creates (as tuples of 3 other edges)
    # This is expensive, so let's be smart.
    
    # Alternative approach: for a swap (remove e1, add e2), the C4 count is:
    # (C4s created by adding e2 to full polarity) - (C4s created by e2 that use e1)
    # because removing e1 only kills C4s that use e1, and in the original C4-free graph,
    # no C4s exist, so we only need to subtract C4s that would have been created by e2 but use e1.
    
    # For non-edge e2=(i,j), C4s created are {i-a-b-j : a~i, b~j, a~b, a!=b}
    # A C4 uses e1=(u,v) if {i,a}=={u,v} or {a,b}=={u,v} or {b,j}=={u,v}
    
    # Let's precompute for each non-edge the count of C4s it creates,
    # and for each polarity edge, how many of those C4s use it.
    
    # This is O(|non_edges| * |N(i)| * |N(j)|) which could be large.
    # With n=31, max degree ~10, this is manageable.
    
    if time.time() < deadline - 30.0:
        # For each non-edge, compute C4s and track edge usage
        # c4_count[e2] = number of C4s created
        # edge_usage[e1][e2] = number of C4s created by e2 that use e1
        # But storing full matrix is 465*465 which is fine.
        
        # Actually, let's just do the swap search directly with a time budget.
        # For each pair (e1 to remove, e2 to add):
        #   c4s = c4s_created_by_e2 - c4s_using_e1_among_those
        # We want to minimize this.
        
        # Precompute c4s_created_by_e2 for all non-edges
        c4_created = {}
        # Also store the actual C4 edge sets for subtracting
        c4_edges = {}  # e2 -> list of frozensets of 3 edges
        
        for e2 in non_edges:
            i, j = e2
            ni = np.where(adj[i] == 1)[0]
            nj = np.where(adj[j] == 1)[0]
            c4_list = []
            for a in ni:
                for b in nj:
                    if a != b and adj[a, b] == 1:
                        # C4: i-a-b-j, edges: (i,a), (a,b), (b,j)
                        e_ia = (min(i,a), max(i,a))
                        e_ab = (min(a,b), max(a,b))
                        e_bj = (min(b,j), max(b,j))
                        c4_list.append((e_ia, e_ab, e_bj))
            c4_created[e2] = len(c4_list)
            c4_edges[e2] = c4_list
        
        # Now search for best swap
        best_swap_c4 = best_single_c4  # at worst, just add the best single edge
        best_swap = None
        
        # For each e2, for each e1 that appears in any C4 of e2:
        #   new_c4 = c4_created[e2] - (count of C4s of e2 that use e1)
        # We want to minimize new_c4.
        
        # Build reverse index: for each e1, which e2's C4s use it, and how many
        from collections import defaultdict
        # e1_to_count[e1][e2] = number of C4s of e2 that use e1
        e1_usage = defaultdict(lambda: defaultdict(int))
        
        for e2, c4_list in c4_edges.items():
            for (e_ia, e_ab, e_bj) in c4_list:
                for e1 in [e_ia, e_ab, e_bj]:
                    e1_usage[e1][e2] += 1
        
        # Now for each e2, the best e1 to remove is one that maximizes e1_usage[e1][e2]
        # (subject to e1 being in the polarity graph, which it is by construction)
        # new_c4 = c4_created[e2] - max_e1 e1_usage[e1][e2]
        
        for e2 in non_edges:
            if time.time() > deadline - 10.0:
                break
            base = c4_created[e2]
            if base == 0:
                # Adding e2 creates no C4s! Then removing any e1 gives 0 C4s.
                # But we still need exactly 91 edges, so we remove one and add e2.
                best_swap_c4 = 0
                best_swap = (None, e2)  # remove any edge, add e2
                break
            # Find e1 that appears in most C4s of e2
            max_overlap = 0
            best_e1 = None
            for e1, cnt in e1_usage[e2].items():
                if cnt > max_overlap:
                    max_overlap = cnt
                    best_e1 = e1
            new_c4 = base - max_overlap
            if new_c4 < best_swap_c4:
                best_swap_c4 = new_c4
                best_swap = (best_e1, e2)
        
        # If we found a swap with fewer C4s than single add, use it
        if best_swap is not None and best_swap_c4 < best_single_c4:
            e1_to_remove, e2_to_add = best_swap
            if e1_to_remove is None:
                # Just add e2_to_add and remove any edge
                # Find any edge to remove
                for (i, j) in polarity_edges:
                    e1_to_remove = (i, j)
                    break
            # Build final edge list
            edges_final = []
            for (i, j) in polarity_edges:
                if (i, j) != e1_to_remove:
                    edges_final.append([i, j])
            i2, j2 = e2_to_add
            edges_final.append([i2, j2])
            return edges_final
    
    # Fallback: use best single edge add
    if best_single is not None:
        edges_90.append(best_single)
        return edges_90
    
    # Ultimate fallback
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                edges_90.append([i, j])
                return edges_90
    
    return edges_90

# EVOLVE-BLOCK-END
