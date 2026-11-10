import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Map edge (u, v) to index
    edge_to_idx = {}
    idx_to_edge = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge[idx] = (u, v)
            idx += 1
    
    # Precompute Q3 masks
    # Q3 vertices: 0..7
    # Q3 edges: Hamming distance 1
    q3_edges_local = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                q3_edges_local.append((u, v))
                
    # Get all 8-vertex subsets
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute all Q3 masks in K16
    # This is memory intensive but fast to check later
    # 12870 subsets * 840 isomorphisms = 10.8M masks
    # We store them as Python ints (120 bits)
    
    # To save time, we might not precompute ALL, but just enough?
    # No, we need all for the check.
    
    # However, generating 10.8M masks might take too long in Python.
    # Let's optimize generation.
    
    # Alternative: Generate masks on the fly? No, too slow in check loop.
    
    # Let's try to generate them.
    q3_masks = []
    # Map local edge (u,v) in 0..7 to global edge index
    # For a specific subset S (tuple of 8 vertices), we need to map 0..7 to S[0]..S[7]
    
    # Precompute permutation of 8? No, we just iterate all mappings.
    # For each subset S, and each permutation of S?
    # There are 8! = 40320 permutations.
    # But Q3 has symmetries. We can just iterate all 40320 and add unique masks.
    
    # Actually, we can just iterate all 40320 mappings for each subset.
    # 12870 * 40320 = 519M iterations. Too slow.
    
    # Optimization:
    # For a fixed subset S, the set of edges in K16 induced by S is fixed.
    # We just need to check which of the 28 local edges are present in the global graph.
    # But we are precomputing the MASKS of the Q3 subgraphs.
    
    # Let's reduce the number of masks.
    # We only need to check if ANY Q3 is present.
    
    # Let's try a different check strategy for the "Add Edge" phase.
    # Instead of precomputing 10.8M masks, we can check locally.
    
    # Revised Plan:
    # 1. Greedy removal of K44 to get a base graph.
    # 2. For adding edges, check for Q3 by iterating over 8-vertex subsets containing the new edge?
    #    No, Q3 can be anywhere.
    
    # Let's stick to the K44 removal first.
    # Then, for adding edges, we will use a slower but correct Q3 check.
    
    # Fast Q3 Check Function:
    # Given a graph mask, check if it contains Q3.
    # We can iterate over all 12870 subsets.
    # For each subset, get the local edge mask (28 bits).
    # Check if it contains any of the 840 Q3 patterns.
    
    # Precompute the 840 Q3 patterns as 28-bit ints.
    q3_patterns = set()
    # Local edges: 0-1, 0-2, ... 6-7
    # Map local (u,v) to bit index in 0..27
    local_edge_to_bit = {}
    bit = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_bit[(u, v)] = bit
            bit += 1
            
    for perm in itertools.permutations(range(8)):
        # perm maps local 0..7 to subset 0..7
        # Wait, we need to map local Q3 vertices to subset vertices.
        # The pattern is defined on local vertices 0..7.
        # We just need the set of edges.
        # The set of edges is invariant under isomorphism.
        # So we only need to generate the 840 distinct edge sets?
        # No, Q3 is rigid. There are 48 automorphisms.
        # So there are 840 distinct embeddings into a fixed set of 8 vertices?
        # No, into a fixed set of 8 vertices, the number of Q3 subgraphs is 840.
        
        # Let's just generate all 40320 and add to set.
        mask = 0
        for u, v in q3_edges_local:
            # u, v are 0..7 in Q3
            # We need to map them to the subset?
            # No, the pattern is on the local indices 0..7.
            # The "pattern" is just the set of edges in Q3.
            # But Q3 is a specific graph.
            # The edges are (0,1), (0,2), etc.
            # The set of edges is FIXED.
            # So there is only ONE pattern for Q3?
            # No, because the vertices of the subset are not labeled 0..7 in a fixed way relative to Q3.
            # But the edges of Q3 are defined by Hamming distance.
            # If we permute the vertices of Q3, the edge set changes.
            
            # Actually, we can just take the standard Q3 edge set.
            # mask_std = 0
            # for u,v in q3_edges_local:
            #   mask_std |= 1 << local_edge_to_bit[(u,v)]
            
            # This mask_std represents Q3 on vertices 0..7.
            # Any other Q3 on vertices 0..7 is an isomorphic copy.
            # But in the context of a subset S, the vertices are S[0]..S[7].
            # The edges in G are determined by the global graph.
            # We just need to check if the induced subgraph on S contains Q3.
            # This is equivalent to checking if the 28-bit mask of G[S] contains the 28-bit mask of Q3?
            # No. G[S] might have extra edges.
            # We need to check if there exists a bijection f: {0..7} -> {0..7} such that
            # for all (u,v) in Q3, (f(u), f(v)) is an edge in G[S].
            
            # This is the subgraph isomorphism problem.
            # For 8 vertices, we can brute force.
            
            pass # This logic is for the check function, not precomputation.
            
    # Let's implement the check function using the 12870 subsets.
    
    def is_q3_free(mask):
        # mask: 120-bit int
        # For each subset of 8 vertices
        for S in all_8_subsets:
            # Build local mask (28 bits)
            local_mask = 0
            # Map local indices 0..7 to global vertices S[0]..S[7]
            # We need to check all 840 embeddings?
            # Or just check if the induced subgraph contains Q3.
            
            # Optimization:
            # Get the 28 edges of K8 induced by S in the global graph.
            # We can precompute for each S, the list of 28 global edge indices.
            
            # Let's precompute S_edge_indices: dict S -> list of 28 global edge indices
            # And S_local_map: dict S -> dict local_edge -> global_edge_idx
            
            # This is getting complex.
            
            # Simpler:
            # For each S, iterate over all 40320 permutations? No.
            # Iterate over all 840 Q3 embeddings?
            
            # Let's just use the 840 precomputed patterns on the 28 local edges.
            # We need to map the 28 local edges of S to the 28 local edges of the pattern.
            
            # Actually, the pattern is fixed on local indices 0..7.
            # The edges of G[S] are a subset of the 28 possible local edges.
            # We need to check if there is a permutation p of 0..7 such that
            # for all (u,v) in Q3_edges, (p(u), p(v)) is in G[S].
            
            # This is equivalent to:
            # Let M be the 28-bit mask of G[S].
            # For each permutation p, let M_p be the 28-bit mask of the edges (p(u), p(v)) for (u,v) in Q3.
            # Check if M_p is a subset of M.
            
            # There are 40320 permutations.
            # 12870 * 40320 = 519M checks. Too slow.
            
            # We can prune:
            # If the degree of any vertex in G[S] is < 3, skip.
            
            # Let's try to precompute the 840 unique masks M_p?
            # The number of distinct edge sets formed by permuting Q3 vertices is 840.
            # So we only need to check 840 masks.
            
            # Precompute all 840 masks.
            if not hasattr(is_q3_free, 'q3_masks_28'):
                masks = set()
                for perm in itertools.permutations(range(8)):
                    m = 0
                    for u, v in q3_edges_local:
                        # u, v are 0..7
                        # perm[u], perm[v] are the images
                        # We need the bit index for (perm[u], perm[v]) in the local 0..7 space
                        a, b = perm[u], perm[v]
                        if a > b: a, b = b, a
                        bit_idx = local_edge_to_bit[(a, b)]
                        m |= (1 << bit_idx)
                    masks.add(m)
                is_q3_free.q3_masks_28 = list(masks)
                
            q3_masks_28 = is_q3_free.q3_masks_28
            
            # Now, for this S, get the 28-bit mask of G[S]
            # We need to map global edges to local bits.
            # Precompute for each S, the 28 global edge indices.
            # s_edges[S] = list of 28 global edge indices corresponding to local (0,1)..(6,7)
            
            if not hasattr(is_q3_free, 's_edges'):
                is_q3_free.s_edges = {}
                for S in all_8_subsets:
                    edges = []
                    for u in range(8):
                        for v in range(u + 1, 8):
                            gu, gv = S[u], S[v]
                            if gu > gv: gu, gv = gv, gu
                            edges.append(edge_to_idx[(gu, gv)])
                    is_q3_free.s_edges[S] = edges
                    
            s_edges = is_q3_free.s_edges[S]
            
            # Build local_mask
            local_mask = 0
            for i, e_idx in enumerate(s_edges):
                if (mask >> e_idx) & 1:
                    local_mask |= (1 << i)
                    
            # Check if any q3_mask is subset of local_mask
            for qm in q3_masks_28:
                if (local_mask & qm) == qm:
                    return False
                    
        return True

    # Precompute K44 pairs
    # Disjoint 4-sets
    all_4_sets = list(itertools.combinations(range(n), 4))
    k44_pairs = []
    for i in range(len(all_4_sets)):
        for j in range(i + 1, len(all_4_sets)):
            A = all_4_sets[i]
            B = all_4_sets[j]
            if set(A).isdisjoint(set(B)):
                # Get 16 edge indices
                mask = 0
                for u in A:
                    for v in B:
                        a, b = u, v
                        if a > b: a, b = b, a
                        mask |= (1 << edge_to_idx[(a, b)])
                k44_pairs.append(mask)
                
    # Start with K16
    current_mask = (1 << total_edges) - 1
    
    # Greedy removal of K44
    while time.time() < deadline - 20.0:
        found = False
        for k44_mask in k44_pairs:
            if (current_mask & k44_mask) == k44_mask:
                # Found K44
                # Remove a random edge from this K44
                # Get the 16 edge indices
                # We can just pick a random bit from k44_mask
                # But we need to ensure we pick an edge that is in current_mask
                # Since k44_mask is subset of current_mask, any bit in k44_mask is in current_mask
                
                # Pick random edge
                edges_in_k44 = [i for i in range(total_edges) if (k44_mask >> i) & 1]
                random.shuffle(edges_in_k44)
                e_idx = edges_in_k44[0]
                current_mask &= ~(1 << e_idx)
                found = True
                break
        if not found:
            break
            
    # Now try to add edges
    # Iterate through all missing edges
    missing_edges = [i for i in range(total_edges) if not (current_mask >> i) & 1]
    random.shuffle(missing_edges)
    
    for e_idx in missing_edges:
        if time.time() > deadline - 5.0:
            break
        # Try adding
        new_mask = current_mask | (1 << e_idx)
        if is_q3_free(new_mask):
            current_mask = new_mask
            
    # Convert to edge list
    edges = []
    for i in range(total_edges):
        if (current_mask >> i) & 1:
            u, v = idx_to_edge[i]
            edges.append([u, v])
            
    return edges
