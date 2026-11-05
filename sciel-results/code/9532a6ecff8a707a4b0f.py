import random
from itertools import combinations

def construct():
    # Vertices 0..7 in U, 8..15 in V
    U = list(range(8))
    V = list(range(8, 16))
    
    # Initialize with all edges between U and V
    edges = set()
    for u in U:
        for v in V:
            edges.add((u, v))
            
    # Precompute Q3 edge masks for 4x4 bipartite grid
    # Q3 vertices: 0,1,2,3 in U_part, 4,5,6,7 in V_part
    # Edges of Q3 (standard labeling):
    # 0-4, 1-4, 1-5, 2-4, 2-6, 3-4, 3-5, 3-7, 0-5? No.
    # Let's generate Q3 edges properly.
    # Q3 is hypercube. Vertices are 3-bit strings.
    # U_part: even parity? No, just split.
    # Standard Q3:
    # 000, 001, 010, 011, 100, 101, 110, 111
    # Bipartition by parity of sum of bits.
    # Even: 000, 011, 101, 110 (4 vertices) -> Let's call these U_idx 0..3
    # Odd: 001, 010, 100, 111 (4 vertices) -> Let's call these V_idx 0..3
    
    # Map standard Q3 vertices to indices
    # U_idx: 000->0, 011->1, 101->2, 110->3
    # V_idx: 001->0, 010->1, 100->2, 111->3
    
    # Edges in Q3 connect vertices differing by 1 bit.
    # 000 (U0) connected to 001 (V0), 010 (V1), 100 (V2)
    # 011 (U1) connected to 001 (V0), 010 (V1), 111 (V3)
    # 101 (U2) connected to 001 (V0), 100 (V2), 111 (V3)
    # 110 (U3) connected to 010 (V1), 100 (V2), 111 (V3)
    
    # Let's verify degrees:
    # U0: V0, V1, V2 (3)
    # U1: V0, V1, V3 (3)
    # U2: V0, V2, V3 (3)
    # U3: V1, V2, V3 (3)
    # Total 12 edges. Correct.
    
    # We need all 840 permutations of these edges?
    # No, we just need to check if the current 4x4 mask contains ANY isomorphism of this edge set.
    # Since the graph is small, we can precompute all distinct edge masks that form a Q3.
    # The edge set is a subset of the 16 possible (u,v) pairs.
    # The base mask:
    base_edges = [
        (0,0), (0,1), (0,2), # U0
        (1,0), (1,1), (1,3), # U1
        (2,0), (2,2), (2,3), # U2
        (3,1), (3,2), (3,3)  # U3
    ]
    
    # Generate all permutations of U indices and V indices?
    # Actually, we can just generate all 4! * 4! = 576 mappings?
    # No, the automorphism group of Q3 has size 48.
    # So there are 8! / 48 = 840 distinct embeddings of Q3 into K4,4?
    # Wait, the vertices of Q3 are labeled 0..7.
    # We are mapping them to the 4 U-indices and 4 V-indices.
    # The bipartition is fixed (U vs V).
    # So we map the 4 U-vertices of Q3 to the 4 U-indices of the subgraph.
    # And the 4 V-vertices of Q3 to the 4 V-indices.
    # There are 4! * 4! = 576 such mappings.
    # However, many produce the same edge mask.
    # We can just iterate over the 576 mappings and collect unique masks.
    
    q3_masks = set()
    import itertools
    for perm_u in itertools.permutations(range(4)):
        for perm_v in itertools.permutations(range(4)):
            mask = 0
            # Map base edges to new indices
            # base_edges uses indices 0..3 for U and V
            # perm_u[i] is the new index for old U index i
            # perm_v[j] is the new index for old V index j
            # Wait, base_edges are (u_old, v_old).
            # We want to map u_old -> perm_u[u_old]? No.
            # The permutation perm_u is a bijection from {0,1,2,3} to {0,1,2,3}.
            # Let's say new_u = perm_u[old_u].
            for (u_old, v_old) in base_edges:
                new_u = perm_u[u_old]
                new_v = perm_v[v_old]
                # Bit position: new_u * 4 + new_v
                bit_pos = new_u * 4 + new_v
                mask |= (1 << bit_pos)
            q3_masks.add(mask)
            
    q3_masks_list = list(q3_masks)
    
    # Greedy removal
    # To speed up, we can randomize the order of subsets
    while True:
        found = False
        # Iterate through all 4x4 subsets
        # We can shuffle the list of subsets to randomize which edge is removed
        subsets_u = list(itertools.combinations(range(4), 4)) # Just [ (0,1,2,3) ]
        # Actually, the subgraph is defined by choosing 4 indices from 0..3?
        # No, the subgraph is defined by the actual vertices in U and V.
        # U has 8 vertices. We choose 4.
        # So we iterate over combinations of U indices (0..7) and V indices (8..15).
        
        # To make it faster, we can just check all 70 * 70 = 4900 pairs.
        u_combos = list(itertools.combinations(range(8), 4))
        v_combos = list(itertools.combinations(range(8, 16), 4))
        
        # Shuffle to avoid bias
        random.shuffle(u_combos)
        random.shuffle(v_combos)
        
        for uc in u_combos:
            for vc in v_combos:
                # Build mask of edges present between uc and vc
                mask = 0
                # uc[0..3], vc[0..3]
                for i in range(4):
                    for j in range(4):
                        if (uc[i], vc[j]) in edges:
                            bit_pos = i * 4 + j
                            mask |= (1 << bit_pos)
                
                # Check if any q3_mask is subset of mask
                for qm in q3_masks_list:
                    if (mask & qm) == qm:
                        # Found Q3
                        # Remove a random edge from the Q3
                        # The edges in the Q3 are those where mask has 1 and qm has 1
                        # We can just pick a random bit from qm
                        # But we need to map bit index back to (u,v)
                        # bit_pos = i*4+j corresponds to (uc[i], vc[j])
                        
                        # Get all bits in qm
                        bits = [b for b in range(16) if (qm >> b) & 1]
                        random.shuffle(bits)
                        removed = False
                        for b in bits:
                            i = b // 4
                            j = b % 4
                            e = (uc[i], vc[j])
                            if e in edges:
                                edges.remove(e)
                                removed = True
                                break
                        if removed:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
            
    return [list(e) for e in edges]
