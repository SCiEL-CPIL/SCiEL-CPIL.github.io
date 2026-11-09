import random
import time

def construct():
    n = 20
    best_edges = None
    best_count = 0
    start_time = time.time()
    deadline = start_time + 115
    
    def is_k33_free_after_add(adj, u, v):
        """
        Check if adding edge (u, v) to the current graph adj creates a K3,3.
        Assumes adj[u][v] is currently False.
        """
        # If we add (u, v), u becomes a neighbor of v and v becomes a neighbor of u.
        # A K3,3 involving this edge must have {u, u2, u3} in one part and {v, v2, v3} in the other.
        # Since (u, v) is in the K3,3, u is connected to v, v2, v3 and u2, u3 are connected to v, v2, v3.
        # Also u2, u3 must be connected to v2, v3.
        
        # Candidates for the other two vertices in the part with u:
        # They must be neighbors of v.
        # Candidates for the other two vertices in the part with v:
        # They must be neighbors of u.
        
        # Let's find common neighbors of u and v? No.
        # Let N_v = {x | adj[v][x] is True}
        # Let N_u = {y | adj[u][y] is True}
        
        # We need to pick u2, u3 from N_v \ {u} (since u is not connected to itself, and we are adding u-v, u is not in N_v yet unless we consider the new state, but u is not a neighbor of v in the old state. Wait, if u is in N_v, then (u,v) was already an edge. We only add if not present. So u is not in N_v.)
        # So u2, u3 must be in N_v.
        # We need to pick v2, v3 from N_u.
        
        # Condition:
        # 1. u2, u3 are in N_v.
        # 2. v2, v3 are in N_u.
        # 3. u2 is connected to v2 and v3.
        # 4. u3 is connected to v2 and v3.
        
        # Note: u2 and u3 must be distinct from u. Since u is not in N_v, this is automatic.
        # v2 and v3 must be distinct from v. Since v is not in N_u, this is automatic.
        
        N_v = [x for x in range(n) if x != v and adj[v][x]]
        N_u = [y for y in range(n) if y != u and adj[u][y]]
        
        # We need to check if there exist u2, u3 in N_v and v2, v3 in N_u
        # such that u2~v2, u2~v3, u3~v2, u3~v3.
        
        # Iterate over pairs in N_v
        for i in range(len(N_v)):
            u2 = N_v[i]
            for j in range(i+1, len(N_v)):
                u3 = N_v[j]
                # Check if u2 and u3 share at least 2 common neighbors in N_u
                # Common neighbors of u2 and u3 that are also in N_u
                common_in_Nu = []
                for y in N_u:
                    if adj[u2][y] and adj[u3][y]:
                        common_in_Nu.append(y)
                if len(common_in_Nu) >= 2:
                    # We found v2, v3 in common_in_Nu
                    return False # Not safe to add
            
        return True # Safe to add

    def count_edges(adj):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
        return count

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        adj = [[False]*n for _ in range(n)]
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not adj[u][v]:
                if is_k33_free_after_add(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
        
        if time.time() > deadline:
            break
            
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        best_edges.append([i, j])

    if best_edges is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
