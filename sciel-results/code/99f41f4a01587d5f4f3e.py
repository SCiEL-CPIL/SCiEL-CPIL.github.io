import itertools
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C4-free graph with no independent set of size 11, maximizing n.
    Uses the polarity graph of the projective plane PG(2,7) as a base.
    The polarity graph of PG(2,q) is C4-free.
    We verify the independence number and remove vertices if necessary to ensure alpha < 11.
    """
    q = 7
    
    def norm_point(p):
        x, y, z = p
        # Find first non-zero
        for i in range(3):
            if p[i] != 0:
                inv = pow(p[i], q-2, q)
                return (p[0]*inv % q, p[1]*inv % q, p[2]*inv % q)
        return (0, 0, 0)

    points = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = norm_point((x, y, z))
                points.add(p)
    
    point_list = list(points)
    n_pts = len(point_list)
    
    def is_orthogonal(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 - z1*z2) % q == 0

    adj = [[False]*n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if is_orthogonal(point_list[i], point_list[j]):
                adj[i][j] = True
                adj[j][i] = True
    
    def find_independent_set(n, adj, k):
        """Find an independent set of size k, return list of indices or None."""
        def backtrack(start, chosen, adj, n, k):
            if len(chosen) == k:
                return chosen[:]
            if len(chosen) + (n - start) < k:
                return None
            
            for i in range(start, n):
                is_indep = True
                for j in chosen:
                    if adj[i][j]:
                        is_indep = False
                        break
                if is_indep:
                    res = backtrack(i+1, chosen+[i], adj, n, k)
                    if res is not None:
                        return res
            return None
        return backtrack(0, [], adj, n, k)
    
    def get_induced_adj(active, full_adj, n_full):
        n_act = len(active)
        adj_act = [[False]*n_act for _ in range(n_act)]
        for i in range(n_act):
            for j in range(i+1, n_act):
                u = active[i]
                v = active[j]
                if full_adj[u][v]:
                    adj_act[i][j] = True
                    adj_act[j][i] = True
        return adj_act

    active_vertices = list(range(n_pts))
    
    # Check if full graph is valid
    if find_independent_set(n_pts, adj, 11) is None:
        edges = []
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': n_pts, 'edges': edges}
    
    # Greedy removal
    while True:
        n_act = len(active_vertices)
        if n_act < 1:
            # Fallback
            return {'n': 30, 'edges': []}
            
        adj_act = get_induced_adj(active_vertices, adj, n_pts)
        
        indep_set = find_independent_set(n_act, adj_act, 11)
        
        if indep_set is None:
            # Valid graph found
            edges = []
            old_to_new = {old: new for new, old in enumerate(active_vertices)}
            
            for i in range(n_act):
                for j in range(i+1, n_act):
                    if adj_act[i][j]:
                        u_old = active_vertices[i]
                        v_old = active_vertices[j]
                        u_new = old_to_new[u_old]
                        v_new = old_to_new[v_old]
                        edges.append([u_new, v_new])
                        
            return {'n': n_act, 'edges': edges}
        
        # Remove a vertex from indep_set
        random.shuffle(indep_set)
        victim_local = indep_set[0]
        victim_global = active_vertices[victim_local]
        
        active_vertices.remove(victim_global)

# EVOLVE-BLOCK-END
