import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    
    # Precompute all edges for the final output
    all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Q3 structure: 8 vertices, 12 edges.
    # Canonical Q3 edges (u, v) where u xor v is a power of 2
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            if (u ^ v) & (u ^ v - 1) == 0: # Power of 2
                q3_edges.append((u, v))
    # q3_edges has 12 pairs.
    
    # Precompute all 8-subsets of 16 vertices
    # C(16,8) = 12870
    from itertools import combinations
    subsets_8 = list(combinations(range(n), 8))
    
    # For each subset, we need to check if it can be mapped to Q3.
    # There are 8! / 48 = 840 distinct subgraphs (isomorphism classes of induced subgraphs? No, just edge sets).
    # Actually, we just need to know if the graph induced by the 8 vertices *contains* a Q3.
    # A Q3 is a specific set of 12 edges.
    # The induced subgraph on 8 vertices has at most 28 edges.
    # We need to check if there is a bijection f: {0..7} -> Subset such that for all (u,v) in q3_edges, 
    # the edge (f(u), f(v)) exists in the graph.
    
    # To speed this up, we can precompute the "edge masks" for the 840 Q3 variants on 8 vertices.
    # Let's represent the edges of the induced subgraph on 8 vertices as a bitmask of length 28.
    # For a subset S = (v0, ..., v7), the edges are (vi, vj).
    # We map (vi, vj) to an index 0..27.
    
    # Precompute the 840 Q3 edge masks on 8 vertices.
    # The 840 masks correspond to the distinct edge sets of Q3 on 8 labeled vertices.
    # We can generate them by taking all permutations of 0..7, checking if the edge set is unique, and storing the mask.
    
    q3_masks = set()
    for perm in range(math.factorial(8)):
        # Unpack permutation
        # This is slow to do at runtime. Let's hardcode or generate once.
        pass
        
    # Actually, generating 40320 perms and extracting masks is fast enough if done once.
    # But we can be smarter.
    # The Q3 is the hypercube.
    # Let's just use the bitmask check on the full graph adj matrix during SA.
    
    def build_graph(pos, R):
        """
        pos: 16x3 array
        R: threshold
        Returns: adj list of bitmasks
        """
        adj = [0] * n
        # Compute pairwise distances
        # For speed, we can use numpy
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                dists[i][j] = d
                dists[j][i] = d
                if d < R:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def is_q3_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is Q3-free.
        Reuses logic from current program but optimized.
        """
        # Iterate over all 8-subsets
        # For each subset, check if it contains a Q3.
        # To check if a specific 8-vertex induced subgraph contains a Q3:
        # We can extract the 28 edges into a 28-bit mask.
        # Then check if any of the 840 precomputed Q3 masks is a subset of this mask.
        
        # Precompute the 840 masks once outside this function?
        # Yes, let's do that.
        
        return True # Placeholder, will implement below

    # Let's precompute the 840 Q3 masks for 8 vertices.
    # Vertices 0..7. Edges (u,v) where u xor v is power of 2.
    # Map edge (u,v) to index: 
    # There are 28 edges. 
    # Index = u*7 - u*(u+1)//2 + (v-u-1) ? No, simpler:
    edge_idx_map = {}
    idx = 0
    for u in range(8):
        for v in range(u+1, 8):
            edge_idx_map[(u, v)] = idx
            idx += 1
            
    q3_edge_indices = [edge_idx_map[(u, v)] for u, v in q3_edges]
    
    # Generate all unique edge masks for Q3 on 8 vertices
    # We iterate over all 40320 permutations.
    # For each perm, the edge (u,v) in canonical Q3 becomes (perm[u], perm[v]) in the labeled graph.
    # We need to map (perm[u], perm[v]) to an index in the 0..27 range.
    # Note: perm[u] and perm[v] are the labels in the subset.
    
    unique_q3_masks = set()
    # Use a faster way to generate perms? 
    # 40320 is small.
    from itertools import permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges:
            pu, pv = p[u], p[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << edge_idx_map[(pu, pv)])
        unique_q3_masks.add(mask)
        
    q3_masks_list = list(unique_q3_masks)
    # Should be 840 masks.

    def is_q3_free_fast(adj):
        """
        Fast check using precomputed masks.
        """
        for subset in subsets_8:
            # Extract induced subgraph mask
            mask = 0
            for i in range(8):
                ui = subset[i]
                for j in range(i+1, 8):
                    uj = subset[j]
                    if adj[ui] & (1 << uj):
                        # Edge exists
                        # Map (i, j) to index in 0..27
                        # The index in edge_idx_map corresponds to the sorted pair (i, j)
                        # Since i < j, we can use the same logic
                        # But edge_idx_map is for 0..7.
                        # We need to map the local indices i, j to the global edge index in the 28-bit mask.
                        # Let's precompute a local index map for i,j in 0..7.
                        pass
            
            # This is getting complicated to map local i,j to the 28-bit mask index.
            # Let's just use the local indices directly.
            # The edge_idx_map was built for 0..7.
            # So for local indices i, j, the bit position is edge_idx_map[(i, j)].
            
            local_mask = 0
            for i in range(8):
                ui = subset[i]
                for j in range(i+1, 8):
                    uj = subset[j]
                    if adj[ui] & (1 << uj):
                        local_mask |= (1 << edge_idx_map[(i, j)])
            
            # Check if local_mask contains any Q3 mask
            # local_mask is a superset of a Q3 mask if (local_mask & q3_mask) == q3_mask
            for q3m in q3_masks_list:
                if (local_mask & q3m) == q3m:
                    return False
        return True

    # Optimize: is_q3_free_fast is called many times.
    # 12870 subsets * 840 masks = ~10 million checks per evaluation.
    # This is too slow for SA.
    
    # Alternative: 
    # Only check subsets that are "likely" to be Q3s.
    # Or, use the `creates_q3_fast` logic from the current program which is much faster on average.
    # The current program's `is_q3_free` iterates over edges in G and calls `creates_q3_fast`.
    # Let's use that. It's proven to be fast enough for the current program (score 1.38).
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u, v) to graph adj creates a Q3.
        Assumes (u, v) might or might not be in adj, but checks the structure.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def is_q3_free(adj):
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    if creates_q3_fast(u, v, adj):
                        return False
                mask -= bit
        return True

    # SA Strategy
    # State: pos (16x3), R
    # Init: Random points on sphere, R such that ~50 edges.
    
    pos = np.random.randn(n, 3)
    # Normalize to sphere radius 1
    norms = np.linalg.norm(pos, axis=1)
    pos = pos / norms[:, np.newaxis]
    
    # Find R for ~50 edges
    # Total pairs 120. 50 edges is 40%.
    # Distances on unit sphere: mean dist ~1.4.
    # Let's try R = 1.0 initially.
    R = 1.0
    
    best_edges = 0
    best_pos = pos.copy()
    best_R = R
    
    current_temp = 1.0
    cooling = 0.99
    
    for step in range(20000):
        if time.time() > deadline - 5:
            break
            
        # Evaluate current state
        adj = build_graph(pos, R)
        edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        
        if edge_count > best_edges and is_q3_free(adj):
            best_edges = edge_count
            best_pos = pos.copy()
            best_R = R
        
        # Propose Move
        move_type = random.random()
        new_pos = pos.copy()
        new_R = R
        
        if move_type < 0.5:
            # Jitter one point
            idx = random.randint(0, n-1)
            new_pos[idx] += np.random.randn(3) * current_temp
            # Project back to sphere? Or allow free space?
            # Free space might allow denser packing?
            # Let's keep on sphere to maintain structure.
            norm = np.linalg.norm(new_pos[idx])
            if norm > 0:
                new_pos[idx] = new_pos[idx] / norm
        elif move_type < 0.8:
            # Jitter R
            new_R = R + np.random.randn() * current_temp * 0.1
            if new_R < 0.1: new_R = 0.1
        else:
            # Swap two points (symmetry breaking)
            i, j = random.sample(range(n), 2)
            new_pos[i], new_pos[j] = new_pos[j].copy(), new_pos[i].copy()
            
        # Evaluate New State
        new_adj = build_graph(new_pos, new_R)
        new_edge_count = sum(bin(new_adj[i]).count('1') for i in range(n)) // 2
        
        # Check validity
        # If invalid, penalty is huge
        if is_q3_free(new_adj):
            new_score = new_edge_count
        else:
            new_score = new_edge_count - 1000 # Penalty
        
        current_score = edge_count
        if not is_q3_free(adj):
            current_score = edge_count - 1000
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / current_temp):
            pos = new_pos
            R = new_R
            
        current_temp *= cooling
        
    # Final Polish on Best
    # Try to add edges to best graph by decreasing R or moving points closer
    # We can do a local search on R only for the best geometry.
    
    # Reconstruct best adj
    best_adj = build_graph(best_pos, best_R)
    
    # Try to decrease R slightly to add edges? No, decreasing R removes edges.
    # We want to MAXIMIZE edges.
    # So we want to INCREASE R or DECREASE distances.
    
    # Let's try to increase R.
    # But increasing R might create Q3s.
    
    # Local Search on R
    r = best_R
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        new_r = r + 0.01
        new_adj = build_graph(best_pos, new_r)
        if is_q3_free(new_adj):
            r = new_r
            best_adj = new_adj
        else:
            break
            
    # Try to move points closer to each other (scale down)
    # If we scale down, distances decrease, so we can increase R to get more edges.
    # Or just decrease R to get more edges at same R? No.
    # If we scale down points, we can keep R same and get more edges.
    
    scale = 1.0
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        new_scale = scale * 0.99
        new_pos = best_pos * new_scale
        new_adj = build_graph(new_pos, best_R)
        if is_q3_free(new_adj):
            best_pos = new_pos
            scale = new_scale
            best_adj = new_adj
        else:
            break

    # Final Verification and Output
    if not is_q3_free(best_adj):
        # Fallback
        adj = [0] * n
        for _ in range(5):
            u, v = random.sample(range(n), 2)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        best_adj = adj

    result_edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            bit = mask & (-mask)
            v = bit.bit_length() - 1
            if v > u:
                result_edges.append([u, v])
            mask -= bit
            
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
