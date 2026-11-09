"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 20
    # adj[u][v] = color or -1
    adj = [[-1 for _ in range(n)] for _ in range(n)]
    
    def check_layer_conflicts(v, u_new, c_new, E_curr, E_prev):
        """
        Check if adding edge (v, u_new) with color c_new creates any non-rainbow C4
        given current edges E_curr (from v to some u < v) and E_prev (among 0..v-1).
        The cycle must be v - u_new - x - w - v.
        Edges: (v, u_new) [new], (u_new, x) [E_prev], (x, w) [E_prev], (w, v) [E_curr or E_prev? No, w < v, so (w,v) is in E_curr if added, or not added].
        Wait, the cycle is v - u_new - x - w - v.
        Vertices: v, u_new, x, w.
        u_new, x, w < v.
        Edges:
        1. (v, u_new) : color c_new
        2. (u_new, x) : must be in E_prev. Let color be c_ux.
        3. (x, w) : must be in E_prev. Let color be c_xw.
        4. (w, v) : must be in E_curr (since w < v and we are building layer v). Let color be c_wv.
        
        So we iterate over all pairs (x, w) such that:
        - x != w, x != u_new, w != u_new
        - (u_new, x) in E_prev
        - (x, w) in E_prev
        - (w, v) in E_curr (already decided in this layer)
        
        Then check if {c_new, c_ux, c_xw, c_wv} are distinct.
        """
        for x in range(v):
            if x == u_new: continue
            if E_prev[u_new][x] == -1: continue
            c_ux = E_prev[u_new][x]
            for w in range(v):
                if w == x or w == u_new: continue
                if E_prev[x][w] == -1: continue
                c_xw = E_prev[x][w]
                # Check if (w, v) is in E_curr
                if w in E_curr:
                    c_wv = E_curr[w]
                    if len({c_new, c_ux, c_xw, c_wv}) == 4:
                        continue
                    else:
                        return False
        return True

    start_time = time.time()
    time_limit = 115.0
    
    # We build the graph layer by layer (vertex by vertex)
    # adj[u][v] stores the color.
    # We maintain adj as the "E_prev" for the current step.
    
    for v in range(n):
        if time.time() - start_time > time_limit:
            break
            
        # E_curr is a dict: u -> color for edges (v, u) decided so far in this layer
        E_curr = {}
        
        # We decide edges (v, u) for u = 0, ..., v-1
        # We use backtracking to maximize edges
        # Order of u: 0 to v-1
        
        def backtrack(u_idx):
            if time.time() - start_time > time_limit:
                return False
            
            if u_idx == v:
                return True
            
            # Option 1: Try to add edge (v, u_idx) with some color
            u = u_idx
            for c in range(n):
                # Check conflicts with E_curr and E_prev (which is adj for vertices < v)
                # Note: adj[v][u] is not set yet, but adj[u][v] is effectively -1 in the "prev" view
                # We need to simulate adding it.
                
                # Check against existing E_curr edges
                # The check function uses E_curr dict.
                # We need to temporarily add (u, c) to E_curr for the check?
                # No, check_layer_conflicts checks if adding (v, u, c) is valid given the *current* state of E_curr.
                # The conflicts are with edges (w, v) where w is in E_curr.
                # The new edge is (u, v).
                # The cycle is v-u-x-w-v.
                # Edges: (v,u) [new], (u,x) [prev], (x,w) [prev], (w,v) [in E_curr].
                
                if check_layer_conflicts(v, u, c, E_curr, adj):
                    # Also need to check if adding this edge creates a conflict with *future* edges?
                    # No, future edges (w', v) with w' > u will be checked when they are added.
                    # When (w', v) is added, it will check against (u, v) which will be in E_curr.
                    # So this is safe.
                    
                    # Also, does (u, v) conflict with itself? No.
                    
                    # Apply
                    adj[v][u] = c
                    adj[u][v] = c
                    E_curr[u] = c
                    
                    if backtrack(u_idx + 1):
                        return True
                    
                    # Undo
                    adj[v][u] = -1
                    adj[u][v] = -1
                    del E_curr[u]
            
            # Option 2: Skip edge (v, u_idx)
            # We try to skip only if adding failed for all colors?
            # To maximize density, we should prefer adding.
            # But if adding blocks many future edges, skipping might be better.
            # For now, greedy: if we can add, we add. If not, we skip.
            # If we can't add with any color, we skip.
            
            if backtrack(u_idx + 1):
                return True
            
            return False
        
        # Start backtracking for this layer
        if not backtrack(0):
             # If time ran out or failed, proceed to next vertex with whatever we have
             pass

    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] != -1:
                coloured_edges.append([u, v, adj[u][v]])
    
    return coloured_edges
# EVOLVE-BLOCK-END
