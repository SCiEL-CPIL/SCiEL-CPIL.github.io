import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 10000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            # Precompute conflict sets: for each vertex v, conflict[v] is the set of 
            # vertices u such that adding edge (u,v) would create a C4.
            # conflict[v] = {u : ∃w∈N(u) such that N(w)∩N(v)≠∅}
            # We maintain this incrementally.
            # Initially all zeros.
            # When we add edge (a,b):
            #   For each neighbor w of a: all neighbors of w (except a,b) gain conflict with b
            #   For each neighbor w of b: all neighbors of w (except a,b) gain conflict with a
            #   Also a gains conflict with all neighbors of neighbors of b (excluding a)
            #   And b gains conflict with all neighbors of neighbors of a (excluding b)
            # Simpler: just recompute from scratch each time? No, too slow.
            # Let's use a different approach: maintain "forbidden" sets.
            # forbidden[v] = set of u such that edge (u,v) cannot be added (would create C4)
            # Initially empty.
            # When adding edge (a,b):
            #   For any w ~ a: for any x ~ w, x != a, x != b: forbidden[x] |= {b} and forbidden[b] |= {x}
            #   For any w ~ b: for any x ~ w, x != b, x != a: forbidden[x] |= {a} and forbidden[a] |= {x}
            #   Also: for any w ~ a: forbidden[w] |= {b} is already handled? No.
            #   Actually: adding (a,b) means that for any path a-w-x-b of length 3, we need to prevent it.
            #   The C4 would be a-w-x-b-a. So we need: for any w~a and x~w and x~b (x!=a), 
            #   the edge (a,b) creates C4. But we're adding (a,b), so we check BEFORE adding.
            #   
            # Let me just use the O(degree) check but optimize with bitmasks.
            
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False
                # Check if ∃w∈N(u) such that N(w)∩N(v)≠∅
                # i.e., ∃w with (adj[u]>>w)&1 and (adj[w] & adj[v]) != 0
                # We can compute: for all w in N(u), check adj[w] & adj[v]
                # Precompute: let S = OR of adj[w] for all w in N(u), then check S & adj[v]
                # But that's not right because we need adj[w] & adj[v] != 0 for SOME w
                # which is equivalent to (OR of adj[w] for w in N(u)) & adj[v] != 0
                # Wait no! That's actually correct: (∃w: adj[w]&adj[v]!=0) iff (OR_w adj[w]) & adj[v] != 0
                # But OR_w adj[w] includes adj[u] itself potentially... no, w ranges over N(u)
                # Actually this is correct. Let me precompute neighbor_union[u] = OR of adj[w] for w in N(u)
                # Then can_add(u,v) = (adj[u]>>v)==0 and (neighbor_union[u] & adj[v]) == 0
                # But neighbor_union[u] changes when we add edges.
                # When we add edge (a,b):
                #   neighbor_union[a] |= adj[b]  (because b is now in N(a), so adj[b] contributes)
                #   neighbor_union[b] |= adj[a]
                #   For each w in N(a) (old neighbors, before adding b): 
                #       neighbor_union[w] |= adj[b]  (because a is in N(w), and now b is in N(a), 
                #       but wait, neighbor_union[w] = OR of adj[x] for x in N(w). 
                #       a is in N(w), so adj[a] is already in neighbor_union[w]. 
                #       After adding (a,b), adj[a] gains bit b, so neighbor_union[w] should gain bit b.
                #       So: neighbor_union[w] |= (1<<b) for all w in N(a)
                #   Similarly for all w in N(b): neighbor_union[w] |= (1<<a)
                #   Also: a's new neighbor is b, so for any w in N(b), w's neighbor_union should include adj[a]
                #       which now has bit b... this is getting complex.
                #
                # Simpler approach: just do the O(degree) check but with early termination.
                # For n<=47, degree is at most 46, so this is at most 46 iterations.
                m = adj[u]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if adj[w] & adj[v]:
                        return False
                    m ^= lsb
                return True
            
            # Greedy construction: prioritize vertices with lowest degree
            # Use a priority approach: always pick the most deficient vertex
            for iteration in range(n * min_deg * 3):
                if time.time() > deadline:
                    break
                
                # Find the most deficient vertex
                min_deg_val = min(degrees)
                if min_deg_val >= min_deg:
                    break
                
                deficient = [i for i in range(n) if degrees[i] == min_deg_val]
                u = random.choice(deficient)
                
                # Get non-neighbors
                non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                if not non_neighbors:
                    break
                
                random.shuffle(non_neighbors)
                
                added = False
                for v in non_neighbors:
                    if can_add(u, v):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added = True
                        break
                
                if not added:
                    # Try a different deficient vertex or break
                    break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}
            
            # Phase 2: Local search - try edge swaps to fix deficient vertices
            if time.time() < deadline:
                for ls_iter in range(200):
                    if time.time() > deadline:
                        break
                    
                    deficient = [i for i in range(n) if degrees[i] < min_deg]
                    if not deficient:
                        break
                    
                    u = random.choice(deficient)
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    if not non_neighbors:
                        break
                    
                    random.shuffle(non_neighbors)
                    
                    improved = False
                    for v in non_neighbors:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            improved = True
                            break
                    
                    if not improved:
                        break
            
            if min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    return {'n': 39, 'edges': []}
