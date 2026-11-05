import time
import random
import itertools

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has a 5-cycle."""
    # A 5-cycle is a set of 5 distinct vertices v0,v1,v2,v3,v4 with edges
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0
    # We can check by looking for paths of length 4 that close into a cycle
    # For efficiency, use the fact that C5 has 5 vertices and 5 edges
    
    # Count 5-cycles: for each ordered tuple (v0,v1,v2,v3,v4) with all distinct
    # check adj[v0][v1] & adj[v1][v2] & adj[v2][v3] & adj[v3][v4] & adj[v4][v0]
    # But this is O(n^5) which is too slow for n=35.
    
    # Better approach: use the fact that in a C5-free graph, the number of 
    # 5-cycles is 0. We can use a randomized or structured check.
    
    # Actually, let's use a more efficient method:
    # For each pair (u,v), count common neighbors, then look for paths of length 4.
    # A 5-cycle exists if there's a path v0-v1-v2-v3-v4 where v4 is adjacent to v0
    # and all vertices are distinct.
    
    # For n up to 35, O(n^5) = 35^5 ≈ 52 million, which might be borderline.
    # Let's use a smarter approach.
    
    # Check all 5-subsets for cycle structure - O(C(n,5) * 5!) is too much.
    
    # Better: iterate over all paths of length 4 and check if they close.
    # Path: v0-v1-v2-v3-v4, check if v4~v0 and all distinct.
    # Number of such paths: n * (n-1) * (n-2) * (n-3) * (n-4) ≈ n^5
    # For n=35: 35*34*33*32*31 ≈ 38 million. This is feasible in optimized code.
    
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 == v0 or v2 == v1 or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2 or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _independence_number(adj, n):
    """Compute the independence number of the graph exactly."""
    # Independence number = maximum set of vertices with no edges between them
    # Equivalent to clique number of the complement graph
    # For n up to 35, we can use Bron-Kerbosch or similar
    
    # Build complement adjacency
    comp = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                comp[i][j] = 1
                comp[j][i] = 1
    
    # Use maximum clique algorithm on complement
    # Bron-Kerbosch with pivot
    best = [0]
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        # Pivot selection
        if P:
            pivot = max(P, key=lambda v: sum(comp[v][u] for u in P) + sum(comp[v][u] for u in X))
            candidates = [v for v in P if not comp[v][pivot]]
        else:
            candidates = []
        
        for v in candidates:
            new_R = R + [v]
            new_P = [u for u in P if comp[v][u]]
            new_X = [u for u in X if comp[v][u]]
            bronk(new_R, new_P, new_X)
            P.remove(v)
            X.append(v)
    
    bronk([], list(range(n)), [])
    return best[0]

def _check_valid(adj, n):
    """Check if graph is C5-free and has independence number < 9."""
    if _has_c5(adj, n):
        return False
    if _independence_number(adj, n) >= 9:
        return False
    return True

def _greedy_random(n, adj_init, max_iter=500):
    """
    Random greedy: start from adj_init, try to add random edges that don't create C5
    and don't reduce independence number below 9.
    Returns (adj, valid) where valid indicates if final graph passes exact check.
    """
    adj = [row[:] for row in adj_init]
    
    # Precompute all possible edges not yet present
    possible_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                possible_edges.append((i, j))
    
    random.shuffle(possible_edges)
    
    for _ in range(max_iter):
        random.shuffle(possible_edges)
        for (i, j) in possible_edges:
            if adj[i][j]:
                continue
            # Try adding edge (i, j)
            adj[i][j] = 1
            adj[j][i] = 1
            
            # Quick check: does this create a C5?
            # Adding edge (i,j) creates a C5 if there's a path of length 4 from i to j
            # that doesn't use edge (i,j) and has all distinct vertices.
            # Path: i - a - b - c - d - j where a,b,c,d distinct from i,j and each other.
            # Actually, a C5 using new edge (i,j) is: i-j-c-d-e-i or i-j-a-b-c-i etc.
            # More precisely: a C5 containing edge (i,j) has form i-j-v3-v4-v5-i
            # where v3,v4,v5 are distinct and not i,j, and edges j-v3, v3-v4, v4-v5, v5-i exist.
            
            c5_found = False
            # Look for paths j-v3-v4-v5-i of length 3 from j to i (not using direct edge)
            # v3 adjacent to j, v4 adjacent to v3, v5 adjacent to v4 and i, all distinct from i,j
            for v3 in range(n):
                if v3 == i or v3 == j or not adj[j][v3]:
                    continue
                for v4 in range(n):
                    if v4 == i or v4 == j or v4 == v3 or not adj[v3][v4]:
                        continue
                    for v5 in range(n):
                        if v5 == i or v5 == j or v5 == v3 or v5 == v4:
                            continue
                        if adj[v4][v5] and adj[v5][i]:
                            c5_found = True
                            break
                    if c5_found:
                        break
                if c5_found:
                    break
            
            if c5_found:
                adj[i][j] = 0
                adj[j][i] = 0
                continue
            
            # Check independence number
            # Adding an edge can only decrease or maintain independence number
            # We need alpha < 9
            # Quick check: if before adding, alpha was already < 9, it stays < 9
            # But we need to track it. Let's compute it periodically.
            # For speed, let's just check every few iterations
            
            # For now, let's check alpha every 10 edge additions
            # This is approximate but should be fine for greedy
            pass
        
        # Periodic exact check
        if _check_valid(adj, n):
            return adj, True
    
    # Final check
    valid = _check_valid(adj, n)
    return adj, valid

def _build_k4_union(n_blocks, block_size=4):
    """Build disjoint union of K_{block_size} blocks."""
    n = n_blocks * block_size
    adj = [[0]*n for _ in range(n)]
    for b in range(n_blocks):
        base = b * block_size
        for i in range(block_size):
            for j in range(i+1, block_size):
                adj[base+i][base+j] = 1
                adj[base+j][base+i] = 1
    return n, adj

def construct():
    """
    Constructs a simple graph on vertices 0..n-1 with no 5-cycle and no independent set of size 9.
    Returns a dict with 'n' and 'edges'.
    Maximize n.
    
    Known best: n=32 via 8 disjoint copies of K4.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Default: n=32 via 8 disjoint K4
    n_best, adj_best = _build_k4_union(8, 4)
    
    # Try to improve by searching for n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Start from a base structure
        # For n=33: 8 K4 + 1 extra vertex
        # For n=34: 8 K4 + 2 extra vertices  
        # For n=35: 8 K4 + 3 extra vertices
        
        # Initialize: take 8 K4 blocks and add extra vertices isolated
        n_base, adj_base = _build_k4_union(8, 4)
        adj_init = [[0]*target_n for _ in range(target_n)]
        for i in range(n_base):
            for j in range(n_base):
                adj_init[i][j] = adj_base[i][j]
        
        # Extra vertices start isolated - they need to connect to reduce independence number
        # Current independence number: 8 (one from each K4) + (target_n - 32) isolated = 8 + extras
        # We need to connect extras to reduce alpha below 9
        
        # Greedy approach: add edges to connect extra vertices to existing structure
        # and between extras, while maintaining C5-freeness
        
        adj_current = [row[:] for row in adj_init]
        
        # First, connect extra vertices to the K4 blocks
        # Each extra vertex should be connected to at least one vertex in each K4 block
        # to prevent it from being in an independent set with one vertex from each K4
        
        extras = list(range(n_base, target_n))
        
        # For each extra vertex, add edges to vertices in each K4 block
        # to ensure no independent set of size 9
        for ev in extras:
            for b in range(8):
                base = b * 4
                # Connect to at least one vertex in each block
                # To be safe, connect to all 4 vertices in some blocks
                # But we must avoid creating C5s
                
                # Try connecting to one vertex per block first
                connected = False
                for v in range(base, base+4):
                    if time.time() > deadline:
                        break
                    # Check if adding edge (ev, v) creates a C5
                    adj_current[ev][v] = 1
                    adj_current[v][ev] = 1
                    
                    # Quick C5 check involving new edge (ev, v)
                    c5 = False
                    # Path: ev-v-a-b-c-ev or v-ev-a-b-c-v
                    # Look for paths of length 3 from v to ev not using direct edge
                    for a in range(target_n):
                        if a == ev or a == v or not adj_current[v][a]:
                            continue
                        for b in range(target_n):
                            if b == ev or b == v or b == a or not adj_current[a][b]:
                                continue
                            for c in range(target_n):
                                if c == ev or c == v or c == a or c == b:
                                    continue
                                if adj_current[b][c] and adj_current[c][ev]:
                                    c5 = True
                                    break
                            if c5:
                                break
                        if c5:
                            break
                    
                    if c5:
                        adj_current[ev][v] = 0
                        adj_current[v][ev] = 0
                    else:
                        connected = True
                        break
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # Now try to add more edges to further reduce independence number
        # and ensure C5-freeness
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Try adding random edges
            added_any = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j or adj_current[i][j]:
                    continue
                
                # Try adding edge
                adj_current[i][j] = 1
                adj_current[j][i] = 1
                
                # Check for C5 involving this edge
                c5 = False
                for a in range(target_n):
                    if a == i or a == j or not adj_current[j][a]:
                        continue
                    for b in range(target_n):
                        if b == i or b == j or b == a or not adj_current[a][b]:
                            continue
                        for c in range(target_n):
                            if c == i or c == j or c == a or c == b:
                                continue
                            if adj_current[b][c] and adj_current[c][i]:
                                c5 = True
                                break
                        if c5:
                            break
                    if c5:
                        break
                
                if c5:
                    adj_current[i][j] = 0
                    adj_current[j][i] = 0
                else:
                    added_any = True
            
            if not added_any:
                break
        
        if time.time() > deadline:
            break
        
        # Exact verification
        if _check_valid(adj_current, target_n):
            n_best = target_n
            adj_best = adj_current
            break  # Found valid graph for this n, try next
    
    # Convert to edges list
    edges = []
    for i in range(n_best):
        for j in range(i+1, n_best):
            if adj_best[i][j]:
                edges.append([i, j])
    
    return {'n': n_best, 'edges': edges}
