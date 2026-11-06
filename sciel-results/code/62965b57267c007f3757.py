import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Precompute all Q3 masks in the 4x4 bipartite representation
    # Q3 as bipartite: 3-regular bipartite graph on 4+4 vertices
    # There are exactly 6 such graphs (up to isomorphism), but we need all labelings
    # Actually, we need all 12-edge subsets of the 16 possible edges that form Q3
    # Q3 is the unique 3-regular bipartite graph on 4+4 (up to isomorphism)
    # Number of 3-regular bipartite graphs on 4+4: we can enumerate
    
    def is_q3_mask(mask):
        # Check if the bipartite graph defined by mask is Q3
        # Q3 is 3-regular on both sides
        deg_a = [0]*4
        deg_b = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    deg_a[i] += 1
                    deg_b[j] += 1
        if all(d == 3 for d in deg_a) and all(d == 3 for d in deg_b):
            return True
        return False

    q3_masks = []
    all_edges_16 = list(range(16))
    for combo in itertools.combinations(all_edges_16, 12):
        mask = 0
        for idx in combo:
            mask |= (1 << idx)
        if is_q3_mask(mask):
            q3_masks.append(mask)
    
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    
    # Initialize with Complement of Shrikhande graph
    def idx(i, j):
        return i * 4 + j
        
    # Build full grid graph K_4 square K_4
    for i in range(4):
        for j in range(4):
            u = idx(i, j)
            for j2 in range(4):
                if j2 != j:
                    v = idx(i, j2)
                    if u < v:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            for i2 in range(4):
                if i2 != i:
                    v = idx(i2, j)
                    if u < v:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
    # Remove a perfect matching to get Shrikhande (5-regular)
    matching = [
        (idx(0,0), idx(1,1)),
        (idx(0,1), idx(1,0)),
        (idx(2,0), idx(3,1)),
        (idx(2,1), idx(3,0)),
        (idx(0,2), idx(1,3)),
        (idx(0,3), idx(1,2)),
        (idx(2,2), idx(3,3)),
        (idx(2,3), idx(3,2))
    ]
    for u, v in matching:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Take the COMPLEMENT
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Fast Q3 detection: check if graph contains Q3
    # Q3 has 8 vertices. We check all 8-subsets, and for each, all ways to split into A(4) and B(4)
    # But we can be smarter: only check splits where the bipartite subgraph is 3-regular
    
    def has_q3(adj):
        """Check if adj contains Q3. Returns (True, edge_in_q3) or (False, None)."""
        # For each 8-subset S
        for S in itertools.combinations(range(n), 8):
            S_set = set(S)
            # For each 4-subset A of S, B = S \ A
            for A in itertools.combinations(S, 4):
                A_set = set(A)
                B = tuple(v for v in S if v not in A_set)
                
                # Build the 16-bit mask of edges between A and B
                # A[0]..A[3] are a0..a3, B[0]..B[3] are b0..b3
                mask = 0
                for i in range(4):
                    a = A[i]
                    for j in range(4):
                        b = B[j]
                        if adj[a] & (1 << b):
                            mask |= (1 << (i*4 + j))
                
                # Quick degree check
                # a-degrees: bits i, i+4, i+8, i+12 for a_i
                if bin(mask & 0x1111).count('1') < 3: continue
                if bin(mask & 0x2222).count('1') < 3: continue
                if bin(mask & 0x4444).count('1') < 3: continue
                if bin(mask & 0x8888).count('1') < 3: continue
                # b-degrees: bits 0-3, 4-7, 8-11, 12-15
                if bin(mask & 0x000F).count('1') < 3: continue
                if bin(mask & 0x00F0).count('1') < 3: continue
                if bin(mask & 0x0F00).count('1') < 3: continue
                if bin(mask & 0xF000).count('1') < 3: continue
                
                # Check if mask contains any Q3 pattern
                for qm in q3_masks:
                    if (mask & qm) == qm:
                        # Found Q3 - return an edge from it
                        for i in range(4):
                            for j in range(4):
                                if qm & (1 << (i*4 + j)):
                                    return (True, (A[i], B[j]))
                        return (True, (A[0], B[0]))
        return (False, None)
    
    # Phase 1: Remove edges to make Q3-free
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        found, edge = has_q3(adj)
        if not found:
            break
        u, v = edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Addition
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                missing_edges.append((u, v))
                
    while time.time() < deadline - 5:
        random.shuffle(missing_edges)
        added_any = False
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if Q3 exists
            found, _ = has_q3(adj)
            
            if found:
                # Q3 found. Remove the new edge.
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep the edge
                added_any = True
        if not added_any:
            break

    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
