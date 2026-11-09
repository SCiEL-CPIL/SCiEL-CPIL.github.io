import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit

    def is_c4_free(adj, n):
        # Check if graph is C4 free
        # For each pair of vertices (u, v), if they are connected,
        # they must have no common neighbor.
        # Actually, if (u,v) is an edge, N(u) \cap N(v) must be empty?
        # No, if (u,v) is an edge, and w is in N(u) \cap N(v), then u-w-v-u is a triangle? No.
        # u-w, w-v, v-u. This is a triangle C3.
        # C4 is u-w-x-v-u.
        # So for any edge (u,v), there should be no path of length 2 from u to v.
        # i.e., N(u) \cap N(v) must be empty?
        # If w in N(u) and w in N(v), then u-w-v is a path of length 2.
        # Combined with edge (u,v), we have a triangle u-w-v-u.
        # Wait, C4 is 4 distinct vertices.
        # If we have edge (u,v) and common neighbor w, we have triangle u-w-v-u.
        # This is C3, not C4.
        # The condition for NO C4 is:
        # For any two distinct vertices u, v, the number of common neighbors is at most 1?
        # If u and v have 2 common neighbors w, x, then u-w-v-x-u is a C4.
        # So, for all pairs u, v, |N(u) \cap N(v)| <= 1.
        
        # This check is O(n^2 * n) = O(n^3).
        # n=46, 46^3 = 97k. Very fast.
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) > 1:
                    return False
        return True

    def try_construct(n, target_deg):
        max_attempts = 10000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            adj = [set() for _ in range(n)]
            edges = []
            
            # We need to add n * target_deg / 2 edges
            num_edges_to_add = int(n * target_deg / 2)
            
            # Randomize the order of vertices to break symmetry
            # We will pick vertices to add edges to.
            
            # Strategy:
            # While edges_added < num_edges_to_add:
            #   Pick u with min degree
            #   Pick v with min degree such that |N(u) \cap N(v)| == 0
            #   (Actually, if u,v are not connected, we need |N(u) \cap N(v)| == 0 to avoid creating a C4 with existing edges?
            #    If we add (u,v), and there is w in N(u) \cap N(v), then u-w-v-u is a triangle? No.
            #    u-w, w-v are edges. (u,v) is new edge.
            #    This forms a triangle u-w-v-u.
            #    Triangles are allowed.
            #    C4 is formed if there are TWO common neighbors.
            #    So when adding (u,v), we must ensure that N(u) \cap N(v) has size < 2?
            #    If N(u) \cap N(v) has size 1, say {w}, then u-w-v-u is a triangle. No C4.
            #    If N(u) \cap N(v) has size 2, say {w,x}, then u-w-v-x-u is a C4.
            #    So the condition to add (u,v) is: |N(u) \cap N(v)| <= 1.
            
            success = True
            for _ in range(num_edges_to_add):
                # Find u with min degree
                min_deg = n
                u = -1
                # To add randomness, we can pick from the bottom 20% of degrees
                degs = [(len(adj[i]), i) for i in range(n)]
                degs.sort()
                
                # Pick u from the lowest degree vertices
                # If we always pick the absolute min, we might be too greedy.
                # Let's pick from the first 5 vertices in sorted order, or just the min.
                # For now, pick min.
                u = degs[0][1]
                
                if len(adj[u]) >= target_deg:
                    # All vertices reached target degree?
                    # This shouldn't happen if we count edges correctly, 
                    # but if some have > target, we might stop early.
                    # We just need min degree >= target.
                    break
                
                # Find v
                # v must have len(adj[v]) < target_deg (or we can allow higher if we just want min deg)
                # But to keep it regular-ish, we prefer low degree.
                # Condition: |N(u) \cap N(v)| <= 1
                valid_vs = []
                for v in range(n):
                    if v == u:
                        continue
                    if len(adj[v]) >= target_deg:
                        continue
                    common = adj[u] & adj[v]
                    if len(common) <= 1:
                        valid_vs.append((len(adj[v]), v))
                
                if not valid_vs:
                    success = False
                    break
                
                # Pick v with min degree
                valid_vs.sort()
                min_v_deg = valid_vs[0][0]
                best_vs = [v for d, v in valid_vs if d == min_v_deg]
                v = random.choice(best_vs)
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
            if success:
                # Final check
                min_d = min(len(adj[i]) for i in range(n))
                if min_d >= target_deg:
                    if is_c4_free(adj, n):
                        return {'n': n, 'edges': edges}
        
        return None

    # Try n=46, deg=7
    res = try_construct(46, 7)
    if res:
        return res
        
    # Try n=45, deg=6
    res = try_construct(45, 6)
    if res:
        return res
        
    # Try n=44, deg=5
    res = try_construct(44, 5)
    if res:
        return res
        
    # Fallback
    return {'n': 39, 'edges': []}
