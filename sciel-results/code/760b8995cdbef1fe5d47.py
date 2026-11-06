import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = set((u, v) for u in range(n) for v in range(u+1, n))
    all_edges_list = list(all_edges)
    
    # Precompute edge index for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges_list)}
    
    # Q3 template: 8 vertices, 12 edges
    # Vertex i connected to i^1, i^2, i^4
    q3_template_edges = set()
    for i in range(8):
        for bit in [1, 2, 4]:
            j = i ^ bit
            if i < j:
                q3_template_edges.add((i, j))
    
    # Generate all 48 automorphisms of Q3
    def generate_q3_autos():
        autos = set()
        for perm in itertools.permutations([0, 1, 2]):
            for flips in itertools.product([0, 1], repeat=3):
                def transform(x):
                    new = 0
                    for i in range(3):
                        bit = (x >> perm[i]) & 1
                        bit ^= flips[i]
                        new |= (bit << i)
                    return new
                auto = tuple(transform(i) for i in range(8))
                autos.add(auto)
        return list(autos)
    
    q3_autos = generate_q3_autos()
    
    # Precompute all distinct Q3 edge sets (as frozensets of edge tuples)
    # Two embeddings give the same edge set iff they differ by an automorphism
    # So we enumerate all bijections from {0..7} to a canonical 8-set, modulo automorphisms
    
    # For each 8-subset of {0..15}, compute all distinct Q3 edge masks
    # Represent as 120-bit integers
    
    # First, precompute for the canonical subset (0,1,...,7)
    canonical_subset = tuple(range(8))
    
    # For each permutation of {0..7}, compute the Q3 edge mask on the canonical subset
    # Then deduplicate using automorphisms
    canonical_q3_masks = set()
    for perm in itertools.permutations(range(8)):
        # perm maps q3 vertex i to local vertex perm[i]
        mask = 0
        for i in range(8):
            for bit in [1, 2, 4]:
                j = i ^ bit
                u = perm[i]
                v = perm[j]
                if u < v:
                    edge = (u, v)
                else:
                    edge = (v, u)
                idx = edge_to_idx[edge]
                mask |= (1 << idx)
        canonical_q3_masks.add(mask)
    
    # Now for each 8-subset of {0..15}, compute all Q3 edge masks
    # This is expensive: C(16,8) * len(canonical_q3_masks)
    # But we can be smarter: precompute all at once
    
    subsets_8 = list(itertools.combinations(range(16), 8))
    
    # Precompute all Q3 masks
    all_q3_masks = []
    for s in subsets_8:
        # For each canonical mask, apply the permutation from canonical_subset to s
        for cmask in canonical_q3_masks:
            # Extract the 28-bit pattern (edges of K8)
            # Map each edge in cmask to the corresponding edge in s
            new_mask = 0
            # cmask has bits set for edges in the canonical K8
            # We need to map these to edges in s
            # For each bit position in cmask, get the corresponding edge in K8,
            # then map to the edge in s
            for bit in range(28):
                if cmask & (1 << bit):
                    # Get the edge in K8 corresponding to this bit
                    # Edge index in K8: we need a mapping
                    # Let's precompute this
                    pass
            # This approach is getting complex. Let me use a different method.
            break
        break
    
    # Let me use a simpler approach: direct Q3 check with backtracking
    # but optimized with bitmask operations
    
    def has_q3(adj_mask):
        """Check if graph contains Q3 as a subgraph."""
        # Quick check: need at least 8 vertices with degree >= 3
        high_deg = [v for v in range(16) if bin(adj_mask[v]).count('1') >= 3]
        if len(high_deg) < 8:
            return False
        
        # Try all 8-subsets of high_deg vertices
        for s in itertools.combinations(high_deg, 8):
            s_set = set(s)
            # Build local adjacency
            local_adj = [0] * 8
            for i, vi in enumerate(s):
                for j, vj in enumerate(s):
                    if i != j and (adj_mask[vi] >> vj) & 1:
                        local_adj[i] |= (1 << j)
            
            # Check if this 8-vertex graph contains Q3
            # Q3 is 3-regular, so each vertex must have degree >= 3 in the local graph
            local_deg = [bin(local_adj[i]).count('1') for i in range(8)]
            if min(local_deg) < 3:
                continue
            
            # Backtracking: map Q3 vertices to local vertices
            q3_adj = [0] * 8
            for i, j in q3_template_edges:
                q3_adj[i] |= (1 << j)
                q3_adj[j] |= (1 << i)
            
            mapping = [-1] * 8  # local vertex -> q3 vertex, or -1
            
            def backtrack(k):
                if k == 8:
                    return True
                # Choose the local vertex to assign next
                # Use most-constrained-first: pick the local vertex with fewest available q3 vertices
                best_li = -1
                best_options = 0
                for li in range(8):
                    if mapping[li] != -1:
                        continue
                    # Count how many q3 vertices can be assigned to li
                    options = 0
                    for q in range(8):
                        # Check if q3 vertex q can be assigned to local vertex li
                        ok = True
                        for q2 in range(8):
                            if mapping[li] == -1 and (q3_adj[q] >> q2) & 1:
                                # q3 vertices q and q2 are adjacent
                                # If q2 is already assigned, check adjacency
                                if mapping[q2] != -1:
                                    if not (local_adj[li] >> mapping[q2]) & 1:
                                        ok = False
                                        break
                        if ok:
                            options += 1
                    if options > best_options:
                        best_options = options
                        best_li = li
                    if options == 0:
                        return False
                
                if best_li == -1:
                    return True
                
                # Try assigning each q3 vertex to best_li
                for q in range(8):
                    # Check if q3 vertex q can be assigned to local vertex best_li
                    ok = True
                    for q2 in range(8):
                        if (q3_adj[q] >> q2) & 1:
                            # q3 vertices q and q2 are adjacent
                            if mapping[q2] != -1:
                                if not (local_adj[best_li] >> mapping[q2]) & 1:
                                    ok = False
                                    break
                    if not ok:
                        continue
                    
                    # Also check: for each q2 not yet assigned, if q and q2 are not adjacent,
                    # no constraint. If they are adjacent, we'll check when q2 is assigned.
                    
                    mapping[best_li] = q
                    if backtrack(k + 1):
                        return True
                    mapping[best_li] = -1
                
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    # Build adjacency mask from edge set
    def build_adj_mask(edge_set):
        adj_mask = [0] * 16
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Start with a good initial graph
    # Try the complement of the Shrikhande graph
    def shrikhande_adj(u, v):
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        da = (a - c) % 4
        db = (b - d) % 4
        if da == 0 and (db == 1 or db == 3):
            return True
        if db == 0 and (da == 1 or da == 3):
            return True
        if da == 2 and db == 2:
            return True
        return False
    
    shrikhande_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            if shrikhande_adj(u, v):
                shrikhande_edges.add((u, v))
    
    complement_edges = all_edges - shrikhande_edges
    
    adj_mask = build_adj_mask(complement_edges)
    if not has_q3(adj_mask):
        best_edges = set(complement_edges)
        best_count = len(best_edges)
    else:
        # Try random dense graphs
        best_edges = None
        best_count = 0
        
        for attempt in range(30):
            if time.time() > deadline - 40:
                break
            random.shuffle(all_edges_list)
            for frac in [0.7, 0.65, 0.6, 0.55, 0.5, 0.45]:
                num_edges = int(len(all_edges_list) * frac)
                candidate = set(all_edges_list[:num_edges])
                adj_m = build_adj_mask(candidate)
                if not has_q3(adj_m):
                    if len(candidate) > best_count:
                        best_count = len(candidate)
                        best_edges = set(candidate)
                    break
        
        if best_edges is None:
            # Fallback: start with a sparse graph and build up
            best_edges = set()
            best_count = 0
    
    # Local search: greedy addition + edge swaps
    candidate_edges = list(all_edges - best_edges)
    
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(candidate_edges)
        current_edges = set(best_edges)
        adj_mask = build_adj_mask(current_edges)
        
        # Phase 1: greedy addition
        for u, v in candidate_edges:
            if time.time() > deadline - 10:
                break
            if (u, v) in current_edges:
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            current_edges.add((u, v))
            if has_q3(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                current_edges.discard((u, v))
        
        # Phase 2: edge swaps
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_list = list(current_edges)
            random.shuffle(current_list)
            for rem_u, rem_v in current_list:
                if time.time() > deadline - 10:
                    break
                adj_mask[rem_u] &= ~(1 << rem_v)
                adj_mask[rem_v] &= ~(1 << rem_u)
                current_edges.discard((rem_u, rem_v))
                
                added = []
                for u2, v2 in candidate_edges:
                    if time.time() > deadline - 10:
                        break
                    if (u2, v2) in current_edges:
                        continue
                    if u2 == rem_u and v2 == rem_v:
                        continue
                    adj_mask[u2] |= (1 << v2)
                    adj_mask[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    if has_q3(adj_mask):
                        adj_mask[u2] &= ~(1 << v2)
                        adj_mask[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                    else:
                        added.append((u2, v2))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    improved = True
                else:
                    for u2, v2 in added:
                        adj_mask[u2] &= ~(1 << u2)
                        adj_mask[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                    adj_mask[rem_u] |= (1 << rem_v)
                    adj_mask[rem_v] |= (1 << rem_u)
                    current_edges.add((rem_u, rem_v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            candidate_edges = list(all_edges - best_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
