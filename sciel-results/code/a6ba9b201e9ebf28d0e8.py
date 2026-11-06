import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Precomputation
    # All 8-subsets
    all_8_subsets = list(combinations(range(16), 8))
    
    # Partitions of 8 items into 4 and 4
    # Represented as tuple of 4 indices for A, B is complement
    partitions = []
    # Indices 0..7
    # Unique partitions: fix 0 in A, choose 3 from 1..7
    for a in combinations(range(1, 8), 3):
        A = tuple(sorted((0,) + a))
        B = tuple(x for x in range(8) if x not in A)
        partitions.append((A, B))
        
    # Precompute for each pair (u,v), the list of 8-subsets containing them
    # This saves filtering time in the inner loop
    subsets_for_pair = {pair: [] for pair in combinations(range(16), 2)}
    for sub in all_8_subsets:
        for pair in combinations(sub, 2):
            subsets_for_pair[pair].append(sub)
            
    # Helper to check if a specific 8-subset contains Q3
    # edges: set of frozensets of size 2, or bitmask
    # Using bitmask for speed: 16 bits.
    # But checking specific edges in a subset requires mapping.
    
    def check_subset_q3(sub, edge_mask):
        # sub: tuple of 8 vertices
        # edge_mask: int, 16 bits
        
        # Map sub vertices to 0..7
        # We need to check 35 partitions
        # For each partition, check internal edges and missing edges matching
        
        # Optimization: 
        # 1. Check if the induced subgraph is bipartite.
        #    If not, return False.
        #    If yes, the bipartition is unique (if connected) or one of few.
        #    But we can just iterate the 35 partitions.
        
        for A, B in partitions:
            # Check internal edges in A
            # A is tuple of 4 indices in 0..7
            # Map to original vertices
            # a_orig = [sub[i] for i in A]
            # Check if any pair in a_orig is in edge_mask
            
            # To speed up, precompute masks for internal edges of A and B?
            # The subset changes, so the mapping changes.
            
            # Let's just check directly.
            # Internal edges in A:
            has_internal_A = False
            for i in range(4):
                u = sub[A[i]]
                for j in range(i+1, 4):
                    v = sub[A[j]]
                    # Check edge (u,v)
                    if (edge_mask >> (u*16 + v)) & 1 or (edge_mask >> (v*16 + u)) & 1:
                        has_internal_A = True
                        break
                if has_internal_A: break
            
            if has_internal_A: continue
            
            # Internal edges in B
            has_internal_B = False
            for i in range(4):
                u = sub[B[i]]
                for j in range(i+1, 4):
                    v = sub[B[j]]
                    if (edge_mask >> (u*16 + v)) & 1 or (edge_mask >> (v*16 + u)) & 1:
                        has_internal_B = True
                        break
                if has_internal_B: break
            
            if has_internal_B: continue
            
            # Bipartite w.r.t A, B
            # Check if missing edges between A and B contain a perfect matching
            # Missing edge (a, b) if not in edge_mask
            
            # Construct bitmask of missing edges between A and B
            # 4x4 matrix
            # Check if it has perfect matching
            
            # Permutations of B for A
            # 24 permutations
            found_matching = False
            for perm in permutations(range(4)):
                # perm maps i in A to perm[i] in B
                ok = True
                for i in range(4):
                    u = sub[A[i]]
                    v = sub[B[perm[i]]]
                    # If edge (u,v) exists, it's not missing.
                    # We need the edge to be MISSING for it to be in the matching of missing edges?
                    # Wait.
                    # Condition: Missing edges contain perfect matching.
                    # So for the matching M, all edges in M must be MISSING from G.
                    # So (u, v) must NOT be in edge_mask.
                    if (edge_mask >> (u*16 + v)) & 1 or (edge_mask >> (v*16 + u)) & 1:
                        ok = False
                        break
                if ok:
                    found_matching = True
                    break
            
            if found_matching:
                return True
                
        return False

    # Main Loop
    # Initialize
    edge_mask = 0
    for _ in range(80): # Random start
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u != v:
            edge_mask |= (1 << (u*16 + v)) | (1 << (v*16 + u))
            
    best_mask = edge_mask
    best_edges = bin(edge_mask).count('1') // 2
    
    # Repair
    while time.time() < deadline - 5.0:
        # Find a Q3
        q3_found = None
        for sub in all_8_subsets:
            if check_subset_q3(sub, edge_mask):
                q3_found = sub
                break
        
        if q3_found is None:
            break
            
        # Remove a random edge from q3_found
        # Get all edges in q3_found
        edges_in_sub = []
        for i in range(8):
            for j in range(i+1, 8):
                u = q3_found[i]
                v = q3_found[j]
                if (edge_mask >> (u*16 + v)) & 1:
                    edges_in_sub.append((u, v))
                    
        if edges_in_sub:
            u, v = random.choice(edges_in_sub)
            edge_mask &= ~((1 << (u*16 + v)) | (1 << (v*16 + u)))
            
    # Update best
    cur_edges = bin(edge_mask).count('1') // 2
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_mask = edge_mask
        
    # Improve
    while time.time() < deadline:
        # Pick random non-edge
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u == v: continue
        if (edge_mask >> (u*16 + v)) & 1: continue
        
        # Tentative add
        new_mask = edge_mask | (1 << (u*16 + v)) | (1 << (v*16 + u))
        
        # Check if new Q3 created
        # Only check subsets containing u, v
        is_valid = True
        for sub in subsets_for_pair[(u, v)]:
            if check_subset_q3(sub, new_mask):
                is_valid = False
                break
        
        if is_valid:
            edge_mask = new_mask
            cur_edges = bin(edge_mask).count('1') // 2
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_mask = edge_mask
        else:
            # Optional: SA acceptance? 
            # For now, just reject.
            pass
            
    # Convert best_mask to list of edges
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if (best_mask >> (u*16 + v)) & 1:
                edges.append([u, v])
                
    return edges
