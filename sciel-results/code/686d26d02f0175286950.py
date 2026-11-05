import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety
    
    n = 18
    best_score = 18
    best_edges = []
    
    # Helper: Get edges from adjacency matrix
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Helper: Check if graph is K4-free using bitmask
    # adj_mask[i] is an integer where bit j is 1 if edge (i,j) exists
    def is_k4_free_mask(adj_mask):
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # If there is a triangle u-v-w, w is in common.
                # A K4 exists if any two vertices in common are adjacent.
                # Iterate through vertices w in common
                w = common
                while w:
                    # Get lowest set bit
                    lsb = w & -w
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    # Check if any other vertex in common is adjacent to wi
                    # common_without_wi = common & ~(1<<wi)
                    # adj[wi] & common_without_wi
                    if adj_mask[wi] & (common & ~(1 << wi)):
                        return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # Uses bitmask DP / Branch and Bound
    def max_triangle_free_size_mask(adj_mask):
        # We want to find the largest subset S such that G[S] has no triangles.
        # This is equivalent to finding the maximum independent set in the "triangle graph"?
        # No, it's just Max Induced Subgraph with no triangles.
        
        # Since n=18 is small, we can use a recursive branch and bound.
        # State: current_set (bitmask), candidates (bitmask of vertices that can be added)
        # A vertex v can be added to S if adding it doesn't create a triangle.
        # This means v must not form a triangle with any two vertices in S.
        # Equivalently, v must not be connected to two adjacent vertices in S.
        
        # Precompute for each vertex v, the set of vertices that form a triangle with v and some other vertex?
        # Actually, let's just use a simple B&B on subsets.
        # Upper bound: size of current set + size of candidates.
        
        best = 0
        
        def dfs(current_set, candidates):
            nonlocal best
            if not candidates:
                if current_set.bit_count() > best:
                    best = current_set.bit_count()
                return
            
            # Upper bound
            if current_set.bit_count() + candidates.bit_count() <= best:
                return
            
            # Pick a vertex from candidates to branch on
            # Heuristic: pick the vertex with the highest degree in the induced subgraph of candidates?
            # Or just the first one.
            v = (candidates & -candidates).bit_length() - 1
            v_mask = 1 << v
            
            # Branch 1: Exclude v
            dfs(current_set, candidates & ~v_mask)
            
            # Branch 2: Include v
            # Check if v can be added to current_set without creating a triangle.
            # A triangle is formed if v is connected to u1, u2 in current_set and u1-u2 is an edge.
            # Let's check all pairs in current_set? That's slow.
            # Alternative: Precompute for each v, the set of "forbidden" vertices?
            # If we add v, we must remove any vertex u in current_set such that u and v are connected, 
            # AND u is connected to some w in current_set? No, that's not right.
            # If we add v, we check if there exists u, w in current_set such that (u,v) in E, (w,v) in E, (u,w) in E.
            
            valid = True
            # Iterate through vertices in current_set
            curr = current_set
            # Optimization: If current_set is small, this is fast.
            # We can optimize by checking if v has any neighbors in current_set that are themselves adjacent to other neighbors of v in current_set.
            neighbors_in_current = adj_mask[v] & current_set
            # For each neighbor u in current_set, if u has any neighbor w in (neighbors_in_current \ {u}), then triangle.
            # So, if any u in neighbors_in_current has (adj_mask[u] & neighbors_in_current & ~(1<<u)) != 0, then invalid.
            
            check = neighbors_in_current
            while check:
                lsb = check & -check
                check ^= lsb
                ui = lsb.bit_length() - 1
                if adj_mask[ui] & (neighbors_in_current & ~lsb):
                    valid = False
                    break
            
            if valid:
                new_current = current_set | v_mask
                # Update candidates: any vertex that forms a triangle with v and some vertex in new_current cannot be added?
                # Actually, the condition "no triangle in induced subgraph" is global.
                # If we include v, we must ensure that no future vertex w creates a triangle with v and some u in new_current.
                # This is complex to update candidates dynamically.
                # Given n=18, and best is likely small (e.g., 7-9), the search space is pruned heavily by the upper bound.
                # We just pass the same candidates minus v? No, candidates are vertices NOT in current_set.
                # If we include v, v is no longer in candidates.
                # Do we need to remove other vertices from candidates?
                # If a vertex w in candidates forms a triangle with v and some u in current_set, then w cannot be added LATER?
                # Yes, if w is added later, the set {v, u, w} would be in the final set, forming a triangle.
                # So we should remove from candidates any vertex w that forms a triangle with v and any u in current_set.
                
                # Forbidden vertices for candidates:
                # w is forbidden if (w,v) in E and (w,u) in E and (v,u) in E for some u in current_set.
                # This is equivalent to: w is in adj_mask[v] AND w is in (union of adj_mask[u] for u in current_set) ... wait.
                # Triangle (v, u, w) requires edges (v,u), (v,w), (u,w).
                # So w must be neighbor of v, and neighbor of some u in current_set (where u is also neighbor of v).
                # Let N_v_current = adj_mask[v] & current_set.
                # For any u in N_v_current, w must not be in adj_mask[u].
                # So w is forbidden if w is in adj_mask[v] AND w is in (adj_mask[u] for some u in N_v_current).
                # Let Forbidden = adj_mask[v]
                # For each u in N_v_current:
                #    Forbidden &= ~adj_mask[u]  # Wait, if w is in adj_mask[u], it's forbidden.
                #    So Forbidden = Forbidden & ~adj_mask[u] ? No.
                #    We want to REMOVE w from candidates if w forms a triangle.
                #    So we calculate the set of vertices that DO form a triangle, and remove them.
                
                triangle_forming_with_v = 0
                # w forms triangle with v and u (u in current_set) if:
                # (v,w) in E AND (u,w) in E AND (v,u) in E.
                # We know (v,u) in E for u in N_v_current.
                # So for each u in N_v_current, any w in adj_mask[v] & adj_mask[u] forms a triangle.
                
                # Union of adj_mask[u] for u in N_v_current, intersected with adj_mask[v].
                # Let's compute this.
                # Actually, we can just check each candidate? No, n is small.
                
                # Let's just keep candidates as is, and rely on the check in the next step?
                # If we don't prune candidates, we might try to add w later, and the check "valid" will fail.
                # This is correct but less efficient.
                # Given the time limit and n=18, the simple check is probably fast enough if the best score is found early.
                
                dfs(new_current, candidates & ~v_mask)
                
        dfs(0, (1 << n) - 1)
        return best

    # Initial Construction: Clebsch Graph on 16 vertices + 2 extra
    def generate_clebsch_adj_mask():
        adj_mask = [0]*n
        # Clebsch on 0..15
        for u in range(16):
            for v in range(u+1, 16):
                diff = u ^ v
                if bin(diff).count('1') == 2:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        # Vertices 16, 17
        # Connect them to break up large TF sets.
        # In Clebsch, max TF set is 16 (the whole graph is TF? No, Clebsch is TF).
        # Wait, Clebsch is triangle-free. So the induced subgraph on 16 vertices is triangle-free.
        # Score is at least 16.
        # We need to add edges to 16, 17 to reduce this.
        # If we connect 16 to some vertices, and 17 to some vertices, we might break the TF property of subsets including 16 or 17.
        # But the subset {0..15} is still TF.
        # To reduce the max TF size, we MUST add edges within {0..15} or ensure that any large TF set involving 16/17 is broken.
        # But {0..15} is a TF set of size 16.
        # So we MUST add edges within the Clebsch part to break it up.
        # The Clebsch graph is the complement of the Shrikhande graph? No.
        # Clebsch is (16, 5, 0, 2).
        # We can add edges to make it non-TF.
        
        # Let's add a few edges to the Clebsch part.
        # Randomly add 2-3 edges to {0..15} to break the large TF sets.
        # We must ensure K4-free.
        
        # Add edge (0, 1)? In Clebsch, 0 and 1 differ by 1 bit, so NOT adjacent.
        # Adding (0,1) creates triangles?
        # Neighbors of 0 in Clebsch: 3, 5, 6, 9, 10 (Hamming dist 2).
        # Neighbors of 1 in Clebsch: 2, 3, 7, 8, 11.
        # Common neighbors of 0 and 1: 3.
        # So adding (0,1) creates triangle (0,1,3).
        # Does it create K4?
        # Triangle (0,1,3). Neighbors of 3 in Clebsch: 0, 1, 10, 11, 12, 13, 14, 15?
        # 3 = 0011. Dist 2 from: 0000(0), 0001(1), 0110(6), 0111(7), 1010(10), 1011(11), 1100(12), 1101(13).
        # Wait, 3's neighbors:
        # 0 (0000), 1 (0001), 6 (0110), 7 (0111), 10 (1010), 11 (1011), 12 (1100), 13 (1101).
        # Common neighbors of 0, 1, 3?
        # 0's neighbors: 3, 5, 6, 9, 10.
        # 1's neighbors: 2, 3, 7, 8, 11.
        # 3's neighbors: 0, 1, 6, 7, 10, 11, 12, 13.
        # Intersection of N(0), N(1), N(3):
        # N(0) ∩ N(1) = {3}.
        # So 3 is the only common neighbor.
        # For K4, we need a vertex x connected to 0, 1, 3.
        # x must be in N(0) ∩ N(1) ∩ N(3).
        # N(0) ∩ N(1) = {3}. But x != 3.
        # So no K4 created by adding (0,1).
        
        # Let's add a few such edges.
        edges_to_add = [(0, 1), (2, 5), (8, 12)]
        for u, v in edges_to_add:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Connect 16, 17
        # Connect 16 to 0, 1, 2
        # Connect 17 to 3, 4, 5
        for i in range(3):
            adj_mask[16] |= (1 << i)
            adj_mask[i] |= (1 << 16)
        for i in range(3, 6):
            adj_mask[17] |= (1 << i)
            adj_mask[i] |= (1 << 17)
            
        # Edge 16-17?
        adj_mask[16] |= (1 << 17)
        adj_mask[17] |= (1 << 16)
        
        return adj_mask

    # Local Search
    def flip_edge_mask(adj_mask, u, v):
        new_mask = adj_mask[:]
        if (new_mask[u] >> v) & 1:
            new_mask[u] &= ~(1 << v)
            new_mask[v] &= ~(1 << u)
        else:
            new_mask[u] |= (1 << v)
            new_mask[v] |= (1 << u)
        return new_mask

    def score_graph_mask(adj_mask):
        if not is_k4_free_mask(adj_mask):
            return None
        return max_triangle_free_size_mask(adj_mask)

    # Start with Clebsch + modifications
    adj_mask = generate_clebsch_adj_mask()
    current_score = score_graph_mask(adj_mask)
    if current_score is not None:
        best_score = current_score
        best_edges = get_edges_from_adj([0]*n) # Placeholder, will fill below
        # Reconstruct edges for best_edges
        temp_adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    temp_adj[i][j] = 1
                    temp_adj[j][i] = 1
        best_edges = get_edges_from_adj(temp_adj)
    else:
        # Fallback to random
        adj_mask = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Remove K4s
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for quad in itertools.combinations(range(n), 4):
                u,v,w,x = quad
                if (adj_mask[u]>>v)&1 and (adj_mask[u]>>w)&1 and (adj_mask[u]>>x)&1 and \
                   (adj_mask[v]>>w)&1 and (adj_mask[v]>>x)&1 and (adj_mask[w]>>x)&1:
                    e = random.choice([(u,v),(u,w),(u,x),(v,w),(v,x),(w,x)])
                    adj_mask[e[0]] &= ~(1 << e[1])
                    adj_mask[e[1]] &= ~(1 << e[0])
                    changed = True
                    break
        current_score = score_graph_mask(adj_mask)
        if current_score is not None:
            best_score = current_score
            temp_adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_mask[i] >> j) & 1:
                        temp_adj[i][j] = 1
                        temp_adj[j][i] = 1
            best_edges = get_edges_from_adj(temp_adj)
        else:
            return []

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try flipping random edges
        u, v = random.sample(range(n), 2)
        new_mask = flip_edge_mask(adj_mask, u, v)
        
        if not is_k4_free_mask(new_mask):
            continue
            
        new_score = score_graph_mask(new_mask)
        if new_score is not None:
            if new_score < current_score:
                current_score = new_score
                adj_mask = new_mask
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    temp_adj = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj_mask[i] >> j) & 1:
                                temp_adj[i][j] = 1
                                temp_adj[j][i] = 1
                    best_edges = get_edges_from_adj(temp_adj)
            elif new_score == current_score and random.random() < 0.2:
                # Accept equal score to escape local minima
                adj_mask = new_mask
                # Do not mark as improved, but move state
                pass

    return best_edges
