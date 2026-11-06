import random
import time
import itertools

def construct():
        n = 20
        # Deadline
        start_time = time.time()
        deadline = start_time + 110  # 110s limit
        
        # Initial graph: 4 disjoint K5s
        # Vertices 0-4, 5-9, 10-14, 15-19
        adj = [[False]*n for _ in range(n)]
        for i in range(4):
            base = i * 5
            for u in range(5):
                for v in range(u+1, 5):
                    adj[base+u][base+v] = True
                    adj[base+v][base+u] = True
        
        def edge_count():
            return sum(sum(row) for row in adj) // 2
        
        def is_valid_add(u, v):
            # Check if adding (u,v) creates a K3,3
            # u and v are not connected yet
            # We need to check if there exist a1, a2 in N(v) and b1, b2 in N(u)
            # such that all cross edges exist.
            # Note: N(u) and N(v) are current neighbors.
            # The new K3,3 would have parts A={u, a1, a2} and B={v, b1, b2}
            # or A={v, b1, b2} and B={u, a1, a2}
            # Since it's symmetric, we just check one case.
            
            nu = [x for x in range(n) if adj[u][x]]
            nv = [x for x in range(n) if adj[v][x]]
            
            # We need a1, a2 from nv
            # We need b1, b2 from nu
            # And edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) must exist.
            
            # Optimization: If len(nu) < 2 or len(nv) < 2, return True
            if len(nu) < 2 or len(nv) < 2:
                return True
                
            # Iterate over pairs in nv
            for a1, a2 in itertools.combinations(nv, 2):
                # For these a1, a2, which b's in nu are connected to both?
                # b must be in N(a1) AND N(a2) AND N(u)
                # Wait, b is in N(u) by definition of nu.
                # So we need b in N(a1) AND N(a2).
                # And we need at least 2 such b's.
                
                candidates_b = []
                for b in nu:
                    if adj[a1][b] and adj[a2][b]:
                        candidates_b.append(b)
                        
                if len(candidates_b) >= 2:
                    return False
                    
            return True
            
        def is_valid_remove(u, v):
            # Removing an edge never creates a K3,3
            return True

        def try_add(u, v):
            if not adj[u][v]:
                if is_valid_add(u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    return True
            return False

        def try_remove(u, v):
            if adj[u][v]:
                adj[u][v] = False
                adj[v][u] = False
                return True
            return False

        best_edges = edge_count()
        best_adj = [row[:] for row in adj]
        
        # Phase 1: Greedy Add
        # Try all possible edges
        for i in range(n):
            for j in range(i+1, n):
                if try_add(i, j):
                    pass
                    
        current_edges = edge_count()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            
        # Phase 2: Local Search / Swaps
        # Try to swap: remove (i,j) and add (k,l)
        # This allows breaking a local constraint to add more edges elsewhere
        
        moves = []
        for i in range(n):
            for j in range(i+1, n):
                moves.append((i,j))
                
        random.seed(42)
        
        while time.time() < deadline:
            # Random restart or perturbation?
            # Let's do random swaps
            # Pick a random edge to remove
            # Pick a random non-edge to add
            # If valid, accept
            
            # Collect current edges and non-edges
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append((i,j))
                    else:
                        non_edges.append((i,j))
                        
            if not edges or not non_edges:
                break
                
            e_rm = random.choice(edges)
            e_add = random.choice(non_edges)
            
            # Try the swap
            # Remove e_rm
            u1, v1 = e_rm
            adj[u1][v1] = False
            adj[v1][u1] = False
            
            # Try add e_add
            u2, v2 = e_add
            if is_valid_add(u2, v2):
                adj[u2][v2] = True
                adj[v2][u2] = True
                current_edges = edge_count()
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [row[:] for row in adj]
            else:
                # Revert add (it wasn't done)
                # But we need to revert remove?
                # If add failed, we can either revert remove (no change) or keep remove (loss of edge)
                # Keeping remove is bad for maximizing edges.
                # So revert remove.
                adj[u1][v1] = True
                adj[v1][u1] = True

        # Phase 3: Final Greedy Add
        for i in range(n):
            for j in range(i+1, n):
                if try_add(i, j):
                    pass
                    
        current_edges = edge_count()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            
        # Return edges from best_adj
        res = []
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    res.append([i, j])
        return res
