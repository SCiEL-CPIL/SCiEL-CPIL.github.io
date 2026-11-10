import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_groetzsch_adj():
    """
    Returns adjacency bitmasks for the Grötzsch graph (11 vertices).
    Vertices 0-4 form a C5.
    Vertices 5-9 are copies of 0-4.
    Vertex 10 is connected to 5-9.
    """
    n = 11
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # C5: 0-1-2-3-4-0
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        add_edge(u, v)
        
    # Mycielski copies
    # 5 (copy of 0) connects to N(0) = {1, 4}
    add_edge(5, 1)
    add_edge(5, 4)
    
    # 6 (copy of 1) connects to N(1) = {0, 2}
    add_edge(6, 0)
    add_edge(6, 2)
    
    # 7 (copy of 2) connects to N(2) = {1, 3}
    add_edge(7, 1)
    add_edge(7, 3)
    
    # 8 (copy of 3) connects to N(3) = {2, 4}
    add_edge(8, 2)
    add_edge(8, 4)
    
    # 9 (copy of 4) connects to N(4) = {3, 0}
    add_edge(9, 3)
    add_edge(9, 0)
    
    # 10 connects to 5, 6, 7, 8, 9
    for i in range(5, 10):
        add_edge(10, i)
        
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_exact(adj, n=16):
    """Exact 3-colorability check using backtracking with strong heuristics."""
    colors = [-1] * n
    uncolored = (1 << n) - 1
    
    # Precompute degrees for initial ordering
    # Dynamic ordering is used in DFS
    
    def dfs():
        nonlocal uncolored
        if uncolored == 0:
            return False # Fully colored means 3-colorable, so return False for "not 3-colorable"
        
        # Pick vertex with maximum degree in remaining uncolored subgraph
        best_v = -1
        best_deg = -1
        m = uncolored
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (adj[v] & uncolored).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        # Try colors
        for c in range(3):
            if forbidden & (1 << c):
                continue
            
            # Heuristic: Check if this choice leads to an immediate contradiction
            # (e.g., neighbor has only 1 color left and it's c? No, just recurse)
            
            colors[v] = c
            uncolored &= ~(1 << v)
            
            # Forward checking: If any uncolored neighbor has no available colors, backtrack
            fail = False
            nn = adj[v] & uncolored
            while nn:
                lsb_nn = nn & -nn
                nb2 = lsb_nn.bit_length() - 1
                nn ^= lsb_nn
                if colors[nb2] == -1:
                    nb_forbidden = 0
                    nnn = adj[nb2]
                    while nnn:
                        lsb_nnn = nnn & -nnn
                        nb3 = lsb_nnn.bit_length() - 1
                        nnn ^= lsb_nnn
                        if colors[nb3] != -1:
                            nb_forbidden |= 1 << colors[nb3]
                    # If all 3 colors are forbidden for nb2, fail
                    if nb_forbidden == 0b111:
                        fail = True
                        break
            
            if not fail:
                if not dfs():
                    return False
            else:
                # Just backtrack, don't return False immediately unless we want to prune harder
                pass
            
            uncolored |= 1 << v
            colors[v] = -1
        
        return True
    
    return dfs()

def is_not_3_colorable_fast(adj, n=16):
    """Fast check: try greedy first, then exact."""
    # Greedy attempt
    for _ in range(3):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        success = True
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                success = False
                break
        if success:
            return False # It IS 3-colorable
    
    # If greedy failed, it might still be 3-colorable. Check exactly.
    return is_not_3_colorable_exact(adj, n)

def local_search_add_edges(adj, edges, n=16, deadline=None):
    """Try to add edges while maintaining triangle-free and not 3-colorable."""
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Generate candidates: non-edges that don't create triangles
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                # Score: prefer adding edges between high-degree vertices?
                # Or low-degree to allow more future additions?
                # Let's try high degree first as it increases density immediately.
                score = adj[u].bit_count() + adj[v].bit_count()
                candidates.append((score, u, v))
        
        candidates.sort(reverse=True)
        
        for score, u, v in candidates:
            if deadline and time.time() > deadline:
                break
            
            # Add edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edges.append([u, v])
            
            # Check 3-colorability
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                # Remove edge
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edges.pop()
                
    return edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    deadline = START_TIME + TIME_LIMIT
    
    g_adj = build_groetzsch_adj()
    n_base = 11
    
    # Precompute saturable pairs in base graph
    # can_saturate[i][j] is True if we can add ALL edges between bag i and bag j
    # Condition: i, j not adjacent in base, and N(i) intersect N(j) is empty
    can_saturate = [[False] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if not (g_adj[i] & (1 << j)):
                if (g_adj[i] & g_adj[j]) == 0:
                    can_saturate[i][j] = True
                    can_saturate[j][i] = True

    best_edges = []
    best_count = 0
    
    # Phase 1: Exhaustive Search over Compositions of Grötzsch Blow-up
    # Number of compositions of 16 into 11 parts is C(15, 10) = 3003
    # This is small enough to do fully.
    
    def gen_compositions(remaining, num_parts):
        if num_parts == 1:
            yield [remaining]
            return
        for first in range(1, remaining - num_parts + 2):
            for rest in gen_compositions(remaining - first, num_parts - 1):
                yield [first] + rest

    # Store top candidates for refinement
    top_candidates = []
    
    for comp in gen_compositions(n, n_base):
        if time.time() > deadline - 10: # Leave time for refinement
            break
            
        # Calculate edge count
        edge_count = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if (g_adj[i] & (1 << j)) or can_saturate[i][j]:
                    edge_count += comp[i] * comp[j]
                    
        # Keep track of top 100 candidates by edge count
        top_candidates.append((edge_count, comp))
        
    # Sort candidates by edge count descending
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates
    # We only need to check the top few because they have the highest potential.
    # However, a slightly lower edge count candidate might be more "flexible" for local search.
    # Let's take top 20.
    
    for edge_count, comp in top_candidates[:20]:
        if time.time() > deadline - 5:
            break
            
        # Construct graph
        bag_map = [0] * n
        current_idx = 0
        for i in range(n_base):
            for _ in range(comp[i]):
                bag_map[current_idx] = i
                current_idx += 1
                
        adj = [0] * n
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                bi = bag_map[i]
                bj = bag_map[j]
                if bi == bj:
                    continue
                if (g_adj[bi] & (1 << bj)) or can_saturate[bi][bj]:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
                    edges.append([i, j])
        
        # Verify triangle-free (should be true by construction, but safe)
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        # Verify not 3-colorable
        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
            # Local search to improve
            current_edges = local_search_add_edges(adj, list(edges), n, deadline)
            if current_edges and len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
    
    # Phase 2: Random Maximal Triangle-Free Search
    # This catches structures that are not blow-ups of Grötzsch.
    if time.time() < deadline - 5:
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        
        for _ in range(50): # 50 random trials
            if time.time() > deadline - 5:
                break
            
            adj = [0] * n
            edges = []
            
            # Greedy construction of maximal triangle-free graph
            # We shuffle the edge list to get different maximal graphs
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
            
            for u, v in shuffled_edges:
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        edges.append([u, v])
            
            # Check if 4-chromatic
            if is_not_3_colorable_fast(adj, n):
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
                    
                # Local search on this random graph
                current_edges = local_search_add_edges(adj, list(edges), n, deadline)
                if current_edges and len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges

    # Fallback
    if not best_edges:
        # Return a known valid graph (Grötzsch blow-up with uniform distribution as much as possible)
        # 16 = 2*5 + 1*6
        comp = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        bag_map = [0] * n
        current_idx = 0
        for i in range(n_base):
            for _ in range(comp[i]):
                bag_map[current_idx] = i
                current_idx += 1
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                bi = bag_map[i]
                bj = bag_map[j]
                if bi == bj: continue
                if (g_adj[bi] & (1 << bj)) or can_saturate[bi][bj]:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
                    edges.append([i, j])
        best_edges = edges

    return best_edges
