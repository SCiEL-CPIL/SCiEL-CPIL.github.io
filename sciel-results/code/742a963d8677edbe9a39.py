import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 16
    all_vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute all 8-sets
    all_8sets = list(itertools.combinations(all_vertices, 8))
    
    # Precompute for each 8-set: list of (A, B) partitions
    # A is a tuple of 4 vertices, B is the complement in the 8-set
    # We can optimize by fixing the first vertex of the 8-set in A to break symmetry?
    # No, we need to check all. But we can precompute the masks.
    
    # For each 8-set, precompute:
    # 1. List of partitions (A, B)
    # 2. For each partition, the bitmask of A and B?
    # Actually, we need to check if the missing edges in G[A,B] form a matching.
    
    # Let's precompute for each 8-set S:
    # partitions[S] = list of (mask_A, mask_B, list_of_pairs_in_AxB)
    # Wait, list_of_pairs_in_AxB is fixed for the partition.
    # We need to check if the set of missing edges is a matching.
    
    # Better:
    # For a partition (A, B), the edges in G[A,B] are determined by adj_mask.
    # Missing edges are (a, b) in A x B such that NOT (adj_mask[a] >> b) & 1.
    # We need these to form a matching.
    
    # Optimization:
    # For a fixed 8-set S, and a fixed partition (A, B):
    # Let M be the set of missing edges.
    # M is a matching iff for every a in A, it has at most 1 missing neighbor in B,
    # AND for every b in B, it has at most 1 missing neighbor in A.
    
    # We can precompute for each 8-set S and partition (A,B):
    # a_mask[a] = bitmask of B vertices connected to a
    # b_mask[b] = bitmask of A vertices connected to b
    # Then missing_a[a] = b_full_mask ^ a_mask[a]
    # missing_b[b] = a_full_mask ^ b_mask[b]
    # Check if missing_a[a] has popcount <= 1 for all a in A
    # AND missing_b[b] has popcount <= 1 for all b in B
    
    # This is very fast.
    
    # Precomputation structure:
    # For each 8-set index idx:
    #   partitions[idx] = list of (list_A, list_B, mask_A, mask_B)
    #   Actually, we just need the lists A and B to iterate.
    
    partition_data = []
    for S in all_8sets:
        parts = []
        S_set = set(S)
        # Generate all subsets A of size 4 from S
        # To avoid duplicates, we can fix the smallest element of S to be in A?
        # No, A and B are interchangeable in the definition of Q3 (bipartite).
        # But (A, B) and (B, A) are the same partition.
        # So we can fix the smallest vertex of S to be in A.
        # This reduces partitions by half.
        
        s_sorted = sorted(S)
        min_v = s_sorted[0]
        rest = [v for v in s_sorted if v != min_v]
        
        # Choose 3 more from rest to be in A
        for A_rest in itertools.combinations(rest, 3):
            A = [min_v] + list(A_rest)
            B = [v for v in s_sorted if v not in A]
            parts.append((A, B))
            
        partition_data.append(parts)

    # Main state
    adj_mask = [0] * 16
    edges = set()
    
    def add_edge(u, v):
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    def remove_edge(u, v):
        edges.remove((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
    def is_q3_free():
        # Check all 8-sets
        for idx, S in enumerate(all_8sets):
            parts = partition_data[idx]
            for A, B in parts:
                # Check if missing edges form a matching
                valid = True
                # For each a in A, count missing neighbors in B
                b_mask_full = 0
                for b in B:
                    b_mask_full |= (1 << b)
                    
                a_mask_full = 0
                for a in A:
                    a_mask_full |= (1 << a)
                
                # Check A side
                for a in A:
                    missing_in_b = b_mask_full & ~(adj_mask[a])
                    # Count bits
                    if missing_in_b & (missing_in_b - 1): # More than 1 bit set
                        valid = False
                        break
                
                if not valid:
                    continue # Try next partition? No, if this partition fails, it's not a Q3 via this partition.
                             # But we need to check if ANY partition makes it a Q3.
                             # Wait. The condition is: Q3 exists iff there EXISTS a partition (A,B) such that missing edges form a matching.
                             # So if we find ONE partition where it works, it's a Q3.
                             # If we finish all partitions and none work, it's not a Q3.
                             
                # Check B side
                for b in B:
                    missing_in_a = a_mask_full & ~(adj_mask[b])
                    if missing_in_a & (missing_in_a - 1):
                        valid = False
                        break
                
                if valid:
                    return False # Found a Q3
        
        return True

    # Greedy construction
    # Start with empty graph? Or random?
    # Random dense start and prune is usually better for maximization.
    # But pruning is expensive if we use is_q3_free() which scans all.
    
    # Alternative: Start empty, add edges greedily.
    # For each edge in random order:
    #   Try to add it.
    #   If adding it creates a Q3, skip.
    #   Else keep it.
    
    # To check if adding (u,v) creates a Q3 efficiently:
    # We only need to check 8-sets containing u and v.
    
    # Precompute: for each pair (u,v), list of 8-set indices containing both.
    # C(14, 6) = 3003 sets per pair.
    # 120 pairs * 3003 = 360k entries. Manageable.
    
    sets_containing = {pair: [] for pair in all_pairs}
    for idx, S in enumerate(all_8sets):
        S_set = set(S)
        for u, v in itertools.combinations(S, 2):
            sets_containing[(u, v)].append(idx)
            sets_containing[(v, u)].append(idx) # Just in case, though we use (min, max)

    def creates_q3_if_add(u, v):
        # Check only relevant 8-sets
        for idx in sets_containing[(u, v)]:
            S = all_8sets[idx]
            parts = partition_data[idx]
            for A, B in parts:
                # u and v must be in opposite partitions for them to be part of the Q3 edges?
                # Yes, Q3 is bipartite. All edges are between A and B.
                # So if u and v are both in A, they cannot be connected in Q3.
                # Since we are adding edge (u,v), they MUST be in opposite partitions.
                
                u_in_A = u in A
                v_in_A = v in A
                if u_in_A == v_in_A:
                    continue # Skip this partition, u and v are on same side, so (u,v) is not a Q3 edge
                
                # Now check if missing edges in G[A,B] form a matching
                # Note: We are checking the graph WITH the edge (u,v) added.
                # So for the specific pair (u,v), the edge is present.
                # For other pairs, use current adj_mask.
                
                b_mask_full = 0
                for b in B:
                    b_mask_full |= (1 << b)
                a_mask_full = 0
                for a in A:
                    a_mask_full |= (1 << a)
                
                valid = True
                
                # Check A side
                for a in A:
                    # Get neighbors in B from current adj_mask
                    # But if a == u, we must include v as neighbor.
                    # If a != u, use adj_mask.
                    # Actually, we can just temporarily add the edge, check, then remove?
                    # Or simulate.
                    
                    # Simulate:
                    # missing_in_b = b_mask_full & ~(neighbors_of_a_in_B)
                    
                    # Let's just temporarily add the edge to adj_mask?
                    # It's faster to simulate.
                    
                    if a == u:
                        # Neighbors in B are (adj_mask[u] & b_mask_full) | (1 << v)
                        neigh = (adj_mask[u] & b_mask_full) | (1 << v)
                    else:
                        neigh = adj_mask[a] & b_mask_full
                    
                    missing = b_mask_full & ~neigh
                    if missing & (missing - 1):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check B side
                for b in B:
                    if b == v:
                        neigh = (adj_mask[v] & a_mask_full) | (1 << u)
                    else:
                        neigh = adj_mask[b] & a_mask_full
                    
                    missing = a_mask_full & ~neigh
                    if missing & (missing - 1):
                        valid = False
                        break
                
                if valid:
                    return True
        
        return False

    # Greedy loop
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if (u, v) in edges:
            continue
        if not creates_q3_if_add(u, v):
            add_edge(u, v)
    
    # Second pass: try to add more?
    # The greedy order matters. Let's try multiple random orders and keep the best.
    # But 110s is tight.
    
    # Let's do a few restarts if time permits.
    best_edges = set(edges)
    best_count = len(edges)
    
    # Reset
    edges.clear()
    adj_mask = [0] * 16
    
    # Restart 2
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if time.time() > deadline - 10:
            break
        if (u, v) in edges:
            continue
        if not creates_q3_if_add(u, v):
            add_edge(u, v)
    
    if len(edges) > best_count:
        best_edges = set(edges)
        best_count = len(edges)
        
    # Restore best
    edges = best_edges
    # Rebuild adj_mask
    adj_mask = [0] * 16
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    return [list(e) for e in edges]

if __name__ == "__main__":
    import sys
    edges = construct()
    print(len(edges))
