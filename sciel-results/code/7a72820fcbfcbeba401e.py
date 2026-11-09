import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    N = 16
    all_pairs = list(itertools.combinations(range(N), 2))
    all_8_subsets = list(itertools.combinations(range(N), 8))
    
    # Precompute: for each pair (u,v), the list of 8-sets containing both
    pair_to_8sets = {}
    for pair in all_pairs:
        u, v = pair
        pair_to_8sets[pair] = [s for s in all_8_subsets if u in s and v in s]
    
    # Precompute 24 permutations of range(4)
    perms24 = list(itertools.permutations(range(4)))
    
    def check_8_set_q3(v8, adj_mask):
        """Check if 8 vertices contain a Q3 subgraph. Returns True if Q3 found."""
        # Quick check: need at least 12 edges
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_mask[v8[i]] >> v8[j]) & 1:
                    edge_count += 1
        if edge_count < 12:
            return False
        
        # Try all C(8,4)=70 ways to split into A (4 vertices) and B (4 vertices)
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A = [v8[i], v8[j], v8[k], v8[l]]
                        mask_A = (1 << A[0]) | (1 << A[1]) | (1 << A[2]) | (1 << A[3])
                        B = [v8[m] for m in range(8) if not (mask_A >> v8[m]) & 1]
                        
                        a0, a1, a2, a3 = A
                        m0, m1, m2, m3 = adj_mask[a0], adj_mask[a1], adj_mask[a2], adj_mask[a3]
                        
                        # For each permutation of B, check if it forms a Q3
                        for p in perms24:
                            b0, b1, b2, b3 = B[p[0]], B[p[1]], B[p[2]], B[p[3]]
                            
                            # Q3 edges: A[i] connects to B[j] where i and j differ in exactly one bit
                            # A[0]=000: connects to B[001],B[010],B[100] = B[p[1]],B[p[2]],B[p[4]]... 
                            # Actually, let me use the standard Q3 adjacency:
                            # Vertices 0-7 in binary, edge between x,y if Hamming distance 1
                            # After relabeling: A = {a0,a1,a2,a3} correspond to bits {0,1,2,3}
                            # B = {b0,b1,b2,b3} correspond to the other 4 positions
                            # 
                            # Standard Q3: label A as 000,001,010,011 and B as 100,101,110,111
                            # A[0]=000 connects to B[0]=100, B[1]=101, B[2]=110
                            # A[1]=001 connects to B[0]=100, B[1]=101, B[3]=111
                            # A[2]=010 connects to B[2]=110, B[3]=111, B[0]=100  -- wait
                            # Let me be more careful.
                            # 
                            # Q3 vertices: 000,001,010,011,100,101,110,111
                            # Edges: between vertices at Hamming distance 1
                            # 
                            # If A = {000,001,010,011} (last bit varies) and B = {100,101,110,111}:
                            # A[0]=000: neighbors in B: 100 (B[0])
                            # A[1]=001: neighbors in B: 101 (B[1])  
                            # A[2]=010: neighbors in B: 110 (B[2])
                            # A[3]=011: neighbors in B: 111 (B[3])
                            # Plus within A: 000-001, 000-010, 001-011, 010-011 (4 edges)
                            # Within B: 100-101, 100-110, 101-111, 110-111 (4 edges)
                            # Total: 4+4+4 = 12 edges
                            #
                            # But we also need cross edges:
                            # 000-100 (A[0]-B[0]), 001-101 (A[1]-B[1]), 010-110 (A[2]-B[2]), 011-111 (A[3]-B[3])
                            # That's only 4 cross edges, total 12. But Q3 has 12 edges total.
                            # Wait, Q3 has 3*2^2 = 12 edges. Let me recount.
                            # Each vertex has degree 3, 8 vertices, so 8*3/2 = 12 edges. Correct.
                            #
                            # So the cross edges are: A[i] to B[i] for i=0,1,2,3
                            # And within A: 000-001, 000-010, 001-011, 010-011
                            # Within B: 100-101, 100-110, 101-111, 110-111
                            #
                            # But this is just ONE specific labeling. We need to check all 24 permutations
                            # of B to account for different labelings.
                            #
                            # Actually, the bipartition of Q3 is unique (it's the only bipartition into two 4-sets
                            # where the graph is 3-regular on each side with the right structure).
                            # 
                            # Let me use a different approach: for the given A and B, check if there exists
                            # a bijection f: A->B such that the induced subgraph is Q3.
                            # 
                            # Q3 structure: each vertex in A has degree 3 in the subgraph,
                            # 2 neighbors in A and 1 neighbor in B. Similarly for B.
                            #
                            # For A = {a0,a1,a2,a3}:
                            #   a0 should connect to a1,a2 in A and to one B vertex
                            #   a1 should connect to a0,a3 in A and to one B vertex
                            #   a2 should connect to a0,a3 in A and to one B vertex
                            #   a3 should connect to a1,a2 in A and to one B vertex
                            #
                            # Wait, that's not right either. Let me just check all 24 permutations.
                            
                            # The key: in Q3, if we fix the labeling of A as 000,001,010,011,
                            # then B must be labeled as 100,101,110,111 in some order that makes
                            # the edge pattern work.
                            #
                            # Edges in Q3 (with A=000,001,010,011 and B=100,101,110,111):
                            # A-A: (000,001),(000,010),(001,011),(010,011)
                            # B-B: (100,101),(100,110),(101,111),(110,111)
                            # A-B: (000,100),(001,101),(010,110),(011,111)
                            #
                            # So for a permutation p of B, we check:
                            # A-A edges: (a0,a1),(a0,a2),(a1,a3),(a2,a3) must exist
                            # B-B edges: (B[p[0]],B[p[1]]),(B[p[0]],B[p[2]]),(B[p[1]],B[p[3]]),(B[p[2]],B[p[3]]) must exist
                            # A-B edges: (a0,B[p[0]]),(a1,B[p[1]]),(a2,B[p[2]]),(a3,B[p[3]]) must exist
                            
                            # Check A-A edges
                            if not ((m0 >> a1) & 1): continue
                            if not ((m0 >> a2) & 1): continue
                            if not ((m1 >> a3) & 1): continue
                            if not ((m2 >> a3) & 1): continue
                            
                            # Check A-B edges
                            if not ((m0 >> B[p[0]]) & 1): continue
                            if not ((m1 >> B[p[1]]) & 1): continue
                            if not ((m2 >> B[p[2]]) & 1): continue
                            if not ((m3 >> B[p[3]]) & 1): continue
                            
                            # Check B-B edges
                            bp0, bp1, bp2, bp3 = B[p[0]], B[p[1]], B[p[2]], B[p[3]]
                            mb0, mb1, mb2, mb3 = adj_mask[bp0], adj_mask[bp1], adj_mask[bp2], adj_mask[bp3]
                            if not ((mb0 >> bp1) & 1): continue
                            if not ((mb0 >> bp2) & 1): continue
                            if not ((mb1 >> bp3) & 1): continue
                            if not ((mb2 >> bp3) & 1): continue
                            
                            return True
        return False
    
    def is_q3_free(adj_mask):
        """Check if the entire graph is Q3-free."""
        for subset in all_8_subsets:
            if check_8_set_q3(subset, adj_mask):
                return False
        return True
    
    def try_add_edge(u, v, edges, adj_mask):
        """Try to add edge (u,v). Returns True if added successfully."""
        if u == v:
            return False
        if (u, v) in edges or (v, u) in edges:
            return False
        
        # Temporarily add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check only 8-sets containing both u and v
        for s in pair_to_8sets[(min(u,v), max(u,v))]:
            if check_8_set_q3(s, adj_mask):
                edges.discard((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        
        return True
    
    def remove_edge(u, v, edges, adj_mask):
        """Remove edge (u,v) if it exists."""
        if (u, v) in edges:
            edges.discard((u, v))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            return True
        elif (v, u) in edges:
            edges.discard((v, u))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            return True
        return False
    
    def greedy_fill(edges, adj_mask, time_limit):
        """Greedy edge addition until no more safe edges can be added."""
        non_edges = [e for e in all_pairs if e not in edges]
        
        while time.time() < time_limit:
            progress = False
            # Sort by degree sum for better heuristic
            def degree(v):
                return bin(adj_mask[v]).count('1')
            
            non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
            
            new_non_edges = []
            for e in non_edges:
                if time.time() > time_limit:
                    break
                u, v = e
                if try_add_edge(u, v, edges, adj_mask):
                    progress = True
                else:
                    new_non_edges.append(e)
            
            non_edges = new_non_edges
            if not progress:
                break
    
    def local_search_2opt(edges, adj_mask, time_limit):
        """Try 2-opt moves: remove one edge, add another."""
        best_count = len(edges)
        
        while time.time() < time_limit:
            improved = False
            all_edges_list = list(edges)
            random.shuffle(all_edges_list)
            
            for e_remove in all_edges_list:
                if time.time() > time_limit:
                    break
                
                u, v = e_remove
                # Temporarily remove
                edges.discard(e_remove)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                # Try to add a different edge
                non_edges = [e for e in all_pairs if e not in edges]
                random.shuffle(non_edges)
                
                added = False
                for e_add in non_edges[:20]:
                    a, b = e_add
                    if try_add_edge(a, b, edges, adj_mask):
                        added = True
                        break
                
                if added and len(edges) > best_count:
                    best_count = len(edges)
                    improved = True
                else:
                    # Revert
                    if added:
                        # Remove the added edge and put back the original
                        a, b = e_add
                        edges.discard((a, b))
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                    edges.add(e_remove)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            
            if not improved:
                break
    
    # Initial structures
    def make_complete():
        e = set()
        for u in range(N):
            for v in range(u+1, N):
                e.add((u, v))
        return e
    
    def make_bipartite(k):
        e = set()
        for u in range(k):
            for v in range(k, N):
                e.add((u, v))
        return e
    
    best_edges = set()
    best_count = -1
    
    # Try multiple starting points
    initial_structures = [
        make_bipartite(4),  # K4,12
        make_bipartite(3),  # K3,13
        make_bipartite(5),  # K5,11
        make_complete(),    # K16 (will need many removals)
    ]
    
    # Add some random structures
    for seed in [1, 2, 3]:
        random.seed(seed)
        e = set()
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < 0.6:
                    e.add((u, v))
        initial_structures.append(e)
    
    random.seed(42)
    
    for init in initial_structures:
        if time.time() > deadline - 15:
            break
        
        edges = set(init)
        adj_mask = [0] * N
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # First, make sure the initial structure is Q3-free
        # If not, remove problematic edges
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for s in all_8_subsets:
                if time.time() > deadline - 10:
                    break
                if check_8_set_q3(s, adj_mask):
                    # Find an edge in this Q3 to remove
                    # Remove a random edge from the 8-set that's in our graph
                    verts = list(s)
                    edges_in_set = [(verts[i], verts[j]) for i in range(8) for j in range(i+1, 8) 
                                     if (verts[i], verts[j]) in edges or (verts[j], verts[i]) in edges]
                    if edges_in_set:
                        e_to_remove = random.choice(edges_in_set)
                        u, v = e_to_remove
                        remove_edge(u, v, edges, adj_mask)
                        changed = True
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
        
        # Greedy fill
        greedy_fill(edges, adj_mask, deadline - 20)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
        
        # Local search
        local_search_2opt(edges, adj_mask, deadline - 10)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
    
    # Final validation
    adj_mask = [0] * N
    for (u, v) in best_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    if not is_q3_free(adj_mask):
        # Shouldn't happen, but remove edges until Q3-free
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for s in all_8_subsets:
                if time.time() > deadline - 2:
                    break
                if check_8_set_q3(s, adj_mask):
                    verts = list(s)
                    edges_in_set = [(verts[i], verts[j]) for i in range(8) for j in range(i+1, 8) 
                                     if (verts[i], verts[j]) in best_edges or (verts[j], verts[i]) in best_edges]
                    if edges_in_set:
                        e_to_remove = random.choice(edges_in_set)
                        u, v = e_to_remove
                        remove_edge(u, v, best_edges, adj_mask)
                        changed = True
    
    return [list(e) for e in best_edges]
