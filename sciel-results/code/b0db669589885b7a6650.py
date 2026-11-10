"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    total_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute Q3 subgraphs: a Q3 is a cube graph on 8 vertices.
    # Q3 is bipartite (4+4), 3-regular, 12 edges, girth 4.
    # A Q3 subgraph in G means 8 vertices induced (or a subset) that contains all 12 cube edges.
    # We'll precompute all subsets of 8 vertices that could host a Q3.
    
    # Actually, we'll use a different approach: precompute the 12 edge patterns of Q3
    # and for any set of 8 vertices, check if there's a bijection to Q3 vertices.
    
    # Q3 edges (in 3-bit representation):
    q3_verts = list(range(8))
    q3_edges_list = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:  # power of 2
                q3_edges_list.append((i, j))
    
    # For any 8-subset S of {0..15}, the number of Q3 embeddings into S
    # We'll precompute for each 8-subset whether it can contain a Q3.
    
    # Better approach: represent the graph as a 120-bit bitmask.
    # For each 8-subset S, determine if adding edges to form a Q3 on S would create a Q3.
    # A Q3 exists if there is a bijection f: {0..7} -> S such that for every (a,b) in Q3_edges, f(a)-f(b) is an edge in G.
    
    # Precompute: for each 8-subset S, list all 8! possible bijections? Too many.
    # Instead, for each 8-subset S, the induced subgraph on S has C(8,2)=28 possible edges.
    # A Q3 is a specific 12-edge subgraph. We need to check if the induced subgraph contains a Q3.
    
    # Key insight: Q3 is 3-regular. So in the induced subgraph on S, we need a 3-regular bipartite subgraph on 8 vertices with 12 edges that is isomorphic to Q3.
    
    # Let's precompute all 8-subsets and for each, the list of 12-edge subsets that form a Q3.
    from itertools import combinations
    
    all_8subsets = list(combinations(range(16), 8))
    
    # For each 8-subset, determine the set of 12-edge patterns (as frozensets of pairs in S) that form a Q3.
    # There are 8! / 6 = 1120 labelings of Q3 onto 8 labeled vertices (since Q3 has automorphism group of size 6).
    # Actually, Q3 has |Aut| = 48 (the symmetry group of the cube). So 8!/48 = 1120 distinct edge sets.
    
    # Precompute all 1120 edge sets for each 8-subset.
    # For a given 8-subset S = (s0, s1, ..., s7), a Q3 embedding is a permutation pi of {0..7}
    # such that edges are {(pi(a), pi(b)) for (a,b) in q3_edges_list}.
    # Two permutations give the same edge set if they differ by an automorphism of Q3.
    
    # We'll generate all unique 12-edge sets for a generic 8-vertex set.
    from itertools import permutations
    
    def get_q3_edge_sets_on_8labeled():
        """Return all unique 12-edge sets on labeled vertices 0..7 that form Q3."""
        seen = set()
        for perm in permutations(range(8)):
            edges = frozenset((min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in q3_edges_list)
            seen.add(edges)
        return seen
    
    # This is 40320 permutations, each producing 12 edges. Let's precompute.
    q3_patterns = get_q3_edge_sets_on_8labeled()
    # len(q3_patterns) should be 1120
    
    # Now for each 8-subset S, map the pattern to actual vertex pairs.
    # For S = (s0,...,s7), pattern {(a,b)} maps to {(S[a], S[b])} where a,b are indices in 0..7.
    
    # Actually, the pattern uses labeled vertices 0..7, and S maps index i to S[i].
    # So for subset S, the actual edges are {(S[a], S[b]) for (a,b) in pattern}.
    
    # Precompute: for each 8-subset, the list of 12-edge sets (as frozensets of actual vertex pairs).
    # Store as a list for fast lookup.
    
    # Edge index mapping: map (u,v) to an index 0..119
    edge_to_idx = {}
    idx_to_edge = []
    for u in range(16):
        for v in range(u+1, 16):
            idx = len(idx_to_edge)
            edge_to_idx[(u,v)] = idx
            idx_to_edge.append((u,v))
    
    # For each 8-subset, precompute the bitmask of edges that form a Q3.
    # q3_mask[i] = bitmask of the 12 edges (in the global 120-bit space) for the i-th Q3 pattern on subset i.
    # There are C(16,8)=12870 subsets, each with 1120 Q3 patterns. That's 14.4M bitmasks.
    # Each bitmask is a Python int (arbitrary precision). This might be memory heavy.
    # 12870 * 1120 = 14,414,400 bitmasks. Each is ~16 bytes for small ints, but 120-bit ints are ~20-40 bytes.
    # Total: ~500MB. Too much.
    
    # Alternative: store as a set of edge-index tuples, or use a different approach.
    # Better: don't precompute all. Instead, use a fast check function.
    
    # Let's use a bit-mask approach for the graph (120 bits).
    # For checking if adding an edge creates a Q3, we need to check if there exists an 8-subset S
    # containing both endpoints, such that S's induced subgraph (including the new edge) contains a Q3.
    
    # Simpler approach: use the existing creates_q3 logic but optimize it.
    # The current creates_q3 is O(n^4) or worse. Let's improve it.
    
    # Alternative: precompute for each pair of vertices (u,v), the list of 8-subsets containing u and v.
    # For each such subset S, check if S can contain a Q3 given the current edge mask.
    
    # Let's precompute: for each edge (u,v), the list of 8-subsets containing u and v.
    # There are C(14, 6) = 3003 such subsets per edge. 120 edges * 3003 = 360k subsets.
    # For each subset, we need to check if it contains a Q3.
    
    # Actually, let's use a different strategy: local search with fast Q3 detection.
    
    # Precompute the 1120 Q3 patterns as bitmasks over the 28 intra-8-subset edges.
    # For an 8-subset S, the 28 edges within S have indices. A Q3 pattern is a 12-bit mask over these 28 edges.
    # Precompute: q3_bitmasks_28 = [1120 28-bit integers]
    
    def get_q3_bitmasks_28():
        """Return list of 28-bit masks representing Q3 edge sets on 8 labeled vertices."""
        seen = set()
        for perm in permutations(range(8)):
            mask = 0
            for a, b in q3_edges_list:
                pa, pb = perm[a], perm[b]
                # Edge index within 0..7: use (min,max) -> index
                # There are C(8,2)=28 edges. Map (x,y) to x*(x-1)//2 + y for x<y? No.
                # Use: idx = x * 7 - x*(x-1)//2 + (y - x - 1) for x < y. Or just a dict.
                pass
            # This is getting complicated. Let's use a different mapping.
        return []
    
    # Let's use a simpler approach: edge index within 8 vertices.
    # For vertices 0..7, edge (i,j) with i<j has index = i*7 - i*(i+1)//2 + j - i - 1
    # Actually, standard: for i<j, index = sum_{k=0}^{i-1} (7-k) + (j-i-1) = i*(7) - i*(i+1)//2 ... 
    # Let's just use a lookup table.
    
    edge_idx_8 = {}
    idx_8 = []
    for i in range(8):
        for j in range(i+1, 8):
            idx = len(idx_8)
            edge_idx_8[(i,j)] = idx
            idx_8.append((i,j))
    # len(idx_8) = 28
    
    def get_q3_bitmasks_28():
        seen = set()
        for perm in permutations(range(8)):
            mask = 0
            for a, b in q3_edges_list:
                pa, pb = perm[a], perm[b]
                mn, mx = min(pa, pb), max(pa, pb)
                mask |= (1 << edge_idx_8[(mn, mx)])
            seen.add(mask)
        return list(seen)
    
    q3_masks_28 = get_q3_bitmasks_28()
    # len = 1120
    
    # Now for each 8-subset S, we can compute the bitmask of edges within S (28 bits) 
    # and check if it matches any Q3 pattern.
    # To do this fast, precompute for each 8-subset S a mapping from global edge index to local edge index (0..27).
    
    # For subset S = (s0,...,s7), the local edge index for (S[i], S[j]) is edge_idx_8[(i,j)].
    # We need a function: given global edge (u,v) and subset S, return local index or -1.
    
    # Precompute: for each 8-subset S (as a tuple), a list of 28 global edge indices.
    # subset_edges[S] = [global_idx for each of the 28 local edges]
    
    # But we have 12870 subsets. 12870 * 28 = 360k integers. Fine.
    
    subset_edges = {}
    for S in all_8subsets:
        edges = []
        for i in range(8):
            for j in range(i+1, 8):
                edges.append(edge_to_idx[(S[i], S[j])])
        subset_edges[S] = edges
    
    # For fast Q3 check: given a 120-bit mask (graph), check if it contains a Q3.
    # Iterate over all 8-subsets, extract the 28-bit submask, and check against q3_masks_28.
    
    # To speed up: precompute for each 8-subset a set of the 1120 Q3 masks (as 28-bit ints).
    # Then for a given graph mask, for each 8-subset, compute the 28-bit submask and check if it's in the set.
    # But 12870 * (cost of extracting 28 bits) per check is too slow if done for every edge addition.
    
    # Better: use the incremental approach. When adding edge (u,v), only check 8-subsets containing u and v.
    # Number of 8-subsets containing u and v: C(14,6) = 3003.
    # For each, extract 28-bit submask and check if it forms a Q3.
    
    # Precompute: for each edge (u,v), the list of 8-subsets containing u and v.
    # And for each such subset, the global edge indices for the 28 edges.
    
    # Let's precompute: for each edge index e, a list of (subset, [28 global indices]).
    # But 120 * 3003 = 360k entries, each with 28 ints. ~10M ints. ~80MB. OK.
    
    # Actually, let's be smarter. For a given edge (u,v), the 8-subsets containing u and v are:
    # {u, v} + 6 from the remaining 14 vertices.
    # For each such subset, we need the 28-bit submask.
    
    # Let's precompute for each pair (u,v):
    #   subsets_with_uv[(u,v)] = list of 8-subsets (tuples) containing u and v
    #   And for each such subset, the 28 global edge indices.
    
    # This is memory heavy. Let's instead compute on the fly with caching.
    
    # Alternative approach: use the fact that Q3 is bipartite (4+4).
    # Precompute all 4-4 partitions of 8-subsets and check for complete bipartite-like structure.
    
    # Let's just implement a fast creates_q3 that checks only 8-subsets containing the new edge.
    
    def creates_q3(u, v, edge_mask):
        """Check if adding edge (u,v) to the graph (edge_mask) creates a Q3."""
        # edge_mask is a 120-bit int
        uv_idx = edge_to_idx[(min(u,v), max(u,v))]
        new_mask = edge_mask | (1 << uv_idx)
        
        # Find all 8-subsets containing u and v
        remaining = [x for x in range(16) if x != u and x != v]
        for six in combinations(remaining, 6):
            S = tuple(sorted([u, v] + list(six)))
            if S not in subset_edges:
                continue
            # Extract 28-bit submask
            local_mask = 0
            edges = subset_edges[S]
            for i, gidx in enumerate(edges):
                if (new_mask >> gidx) & 1:
                    local_mask |= (1 << i)
            # Check if local_mask is a Q3
            # We need to check if local_mask matches any of the 1120 Q3 patterns
            # Precomputed: for each 8-subset, the set of Q3 local masks is the same (just relabeled)
            # Actually, the Q3 patterns are on labeled vertices 0..7, so they're the same for all subsets.
            # So we just check if local_mask is in q3_masks_28_set.
            if local_mask in q3_masks_28_set:
                return True
        return False
    
    q3_masks_28_set = set(q3_masks_28)
    
    # Now let's implement the main search.
    
    # Start with a good initial graph.
    # Use a random greedy approach, then local search.
    
    def popcount(x):
        return bin(x).count('1')
    
    # Initial graph: random dense graph, then remove Q3s.
    # Or start with empty and greedily add edges.
    
    best_mask = 0
    best_count = 0
    
    # Greedy initialization
    random.seed(42)
    
    # Try multiple random orderings
    all_edge_indices = list(range(total_edges))
    
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        random.shuffle(all_edge_indices)
        current_mask = 0
        current_count = 0
        
        for eidx in all_edge_indices:
            u, v = idx_to_edge[eidx]
            if not creates_q3(u, v, current_mask):
                current_mask |= (1 << eidx)
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_mask = current_mask
    
    # Local search: try to add edges, and if that fails, try edge swaps
    # Also try: for each missing edge, check if it can be added.
    
    # Phase 2: local improvement
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        current_mask = best_mask
        current_count = best_count
        improved = True
        
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Try adding each missing edge
            for eidx in all_edge_indices:
                if (current_mask >> eidx) & 1:
                    continue
                u, v = idx_to_edge[eidx]
                if not creates_q3(u, v, current_mask):
                    current_mask |= (1 << eidx)
                    current_count += 1
                    improved = True
            
            # If no edge could be added, try a random kick: remove a random edge and try to re-add others
            if not improved and current_count > 0:
                # Pick a random edge to remove
                present_edges = [i for i in range(total_edges) if (current_mask >> i) & 1]
                if present_edges:
                    rem_idx = random.choice(present_edges)
                    current_mask &= ~(1 << rem_idx)
                    current_count -= 1
                    
                    # Try to add edges again
                    for eidx in all_edge_indices:
                        if (current_mask >> eidx) & 1:
                            continue
                        u, v = idx_to_edge[eidx]
                        if not creates_q3(u, v, current_mask):
                            current_mask |= (1 << eidx)
                            current_count += 1
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_mask = current_mask
    
    # Convert best_mask to list of edges
    result_edges = []
    for eidx in range(total_edges):
        if (best_mask >> eidx) & 1:
            u, v = idx_to_edge[eidx]
            result_edges.append([u, v])
    
    return result_edges

# EVOLVE-BLOCK-END
