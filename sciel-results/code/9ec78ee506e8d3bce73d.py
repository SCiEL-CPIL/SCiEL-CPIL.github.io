import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 16
    target_edges = 65
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def compute_score(adj, deg):
        """Compute max degree sum over all triangles."""
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def degree_ordered_polish(adj, deg, score, worst_tri, deadline):
        """
        Degree-ordered edge-replacement heuristic.
        Iteratively: find worst triangle, identify highest-degree vertex in it,
        try to remove an edge incident to that vertex and add a non-edge that
        minimally increases other triangles' degree sums.
        """
        cur_adj = adj.copy()
        cur_deg = deg.copy()
        cur_score = score
        cur_worst = worst_tri
        no_improve = 0
        
        while time.time() < deadline - 2 and no_improve < 3:
            if cur_worst is None:
                break
            u, v, w = cur_worst
            tri_verts = [u, v, w]
            # Sort by degree descending - prioritize highest-degree vertex
            tri_verts.sort(key=lambda x: -cur_deg[x])
            hi = tri_verts[0]  # highest degree vertex in worst triangle
            mid = tri_verts[1]
            lo = tri_verts[2]
            
            # Edges in the triangle to potentially remove, prioritizing edges
            # incident to the highest-degree vertex
            remove_candidates = [(hi, mid), (hi, lo), (mid, lo)]
            
            # Get all non-edges
            non_edges = [(a, b) for a, b in all_edges if not cur_adj[a, b]]
            
            improved = False
            for (ru, rv) in remove_candidates:
                if not cur_adj[ru, rv]:
                    continue
                # Try adding each non-edge, prioritizing those that don't
                # create new high-degree-sum triangles
                # Sort non-edges by how much they'd increase degree sums
                candidates = []
                for (au, av) in non_edges:
                    # Estimate: does adding (au,av) create triangles with high degree sum?
                    # Quick filter: skip if it would connect two high-degree vertices
                    # forming a triangle with another high-degree vertex
                    est = 0
                    for x in range(n):
                        if x != au and x != av and cur_adj[au, x] and cur_adj[av, x]:
                            est = max(est, cur_deg[au] + 1 + cur_deg[av] + 1 + cur_deg[x])
                    candidates.append((est, (au, av)))
                candidates.sort(key=lambda x: x[0])
                
                for est, (au, av) in candidates:
                    if time.time() > deadline - 3:
                        break
                    # Skip if adding this edge would create a triangle with sum >= cur_score
                    # and the removed edge doesn't break the worst triangle
                    if est >= cur_score:
                        # Only accept if we're breaking the worst triangle
                        if not (au == ru and av == rv or au == rv and av == ru):
                            continue
                    
                    new_adj = cur_adj.copy()
                    new_deg = cur_deg.copy()
                    
                    new_adj[ru, rv] = False
                    new_adj[rv, ru] = False
                    new_deg[ru] -= 1
                    new_deg[rv] -= 1
                    
                    new_adj[au, av] = True
                    new_adj[av, au] = True
                    new_deg[au] += 1
                    new_deg[av] += 1
                    
                    new_score, new_worst = compute_score(new_adj, new_deg)
                    
                    if new_score < cur_score:
                        cur_adj = new_adj
                        cur_deg = new_deg
                        cur_score = new_score
                        cur_worst = new_worst
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return cur_adj, cur_deg, cur_score, cur_worst
    
    def local_search(adj, deg, score, worst_tri, max_iters, deadline):
        """Simulated annealing with targeted moves on worst triangles."""
        cur_adj = adj.copy()
        cur_deg = deg.copy()
        cur_score = score
        cur_worst = worst_tri
        
        T = 3.0
        cooling = 0.997
        non_edges_cache = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if cur_worst is not None:
                u, v, w = cur_worst
                # Sort by degree to prioritize highest-degree vertex
                tri_verts = sorted([u, v, w], key=lambda x: -cur_deg[x])
                hi = tri_verts[0]
                
                # Try to remove an edge incident to the highest-degree vertex
                remove_edges = [(hi, v) for v in [u, w] if v != hi]
                if not remove_edges:
                    remove_edges = [(u, v), (u, w), (v, w)]
                
                random.shuffle(remove_edges)
                moved = False
                
                for (ru, rv) in remove_edges:
                    if not cur_adj[ru, rv]:
                        continue
                    
                    # Find a good non-edge to add
                    if non_edges_cache is None or it % 50 == 0:
                        non_edges_cache = [(a, b) for a, b in all_edges if not cur_adj[a, b]]
                    
                    if not non_edges_cache:
                        continue
                    
                    # Try several random non-edges, prefer those not involving high-degree vertices
                    for _ in range(8):
                        (au, av) = random.choice(non_edges_cache)
                        
                        # Quick reject: if both endpoints have high degree, skip
                        if cur_deg[au] + cur_deg[av] > cur_deg[hi] + cur_deg[lo] + 2:
                            continue
                        
                        new_adj = cur_adj.copy()
                        new_deg = cur_deg.copy()
                        
                        new_adj[ru, rv] = False
                        new_adj[rv, ru] = False
                        new_deg[ru] -= 1
                        new_deg[rv] -= 1
                        
                        new_adj[au, av] = True
                        new_adj[av, au] = True
                        new_deg[au] += 1
                        new_deg[av] += 1
                        
                        new_score, new_worst = compute_score(new_adj, new_deg)
                        delta = new_score - cur_score
                        
                        if delta < 0 or random.random() < np.exp(-max(0, delta) / T):
                            cur_adj = new_adj
                            cur_deg = new_deg
                            cur_score = new_score
                            cur_worst = new_worst
                            non_edges_cache = None
                            moved = True
                            break
                    
                    if moved:
                        break
                
                if not moved:
                    # Random fallback move
                    (ru, rv) = random.choice(all_edges)
                    if not cur_adj[ru, rv]:
                        continue
                    non_edges = [(a, b) for a, b in all_edges if not cur_adj[a, b]]
                    if not non_edges:
                        continue
                    (au, av) = random.choice(non_edges)
                    
                    new_adj = cur_adj.copy()
                    new_deg = cur_deg.copy()
                    new_adj[ru, rv] = False
                    new_adj[rv, ru] = False
                    new_deg[ru] -= 1
                    new_deg[rv] -= 1
                    new_adj[au, av] = True
                    new_adj[av, au] = True
                    new_deg[au] += 1
                    new_deg[av] += 1
                    
                    new_score, new_worst = compute_score(new_adj, new_deg)
                    delta = new_score - cur_score
                    if delta < 0 or random.random() < np.exp(-max(0, delta) / T):
                        cur_adj = new_adj
                        cur_deg = new_deg
                        cur_score = new_score
                        cur_worst = new_worst
            else:
                # No triangles, just do random moves
                (ru, rv) = random.choice(all_edges)
                if not cur_adj[ru, rv]:
                    continue
                non_edges = [(a, b) for a, b in all_edges if not cur_adj[a, b]]
                if not non_edges:
                    continue
                (au, av) = random.choice(non_edges)
                
                new_adj = cur_adj.copy()
                new_deg = cur_deg.copy()
                new_adj[ru, rv] = False
                new_adj[rv, ru] = False
                new_deg[ru] -= 1
                new_deg[rv] -= 1
                new_adj[au, av] = True
                new_adj[av, au] = True
                new_deg[au] += 1
                new_deg[av] += 1
                
                new_score, new_worst = compute_score(new_adj, new_deg)
                delta = new_score - cur_score
                if delta < 0 or random.random() < np.exp(-max(0, delta) / T):
                    cur_adj = new_adj
                    cur_deg = new_deg
                    cur_score = new_score
                    cur_worst = new_worst
            
            T *= cooling
        
        return cur_adj, cur_deg, cur_score, cur_worst
    
    def make_initial(seed):
        """Create a near-bipartite initial graph."""
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        perm = list(range(n))
        random.shuffle(perm)
        split = 8  # Balanced partition
        A = set(perm[:split])
        B = set(perm[split:])
        
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)  # 64
        remaining = target_edges - edge_count  # 1
        
        if remaining > 0:
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, v = part_list[i], part_list[j]
                        if not adj[u, v]:
                            internal_edges.append((u, v))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        
        deg = adj.sum(axis=1).astype(int)
        return adj, deg
    
    # Multiple restarts
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_worst = None
    
    num_restarts = 40
    for seed in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        adj, deg = make_initial(seed)
        score, worst = compute_score(adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_worst = worst
        
        # Local search
        time_per_search = (deadline - 20 - time.time()) / max(1, num_restarts - seed)
        max_iters = max(500, int(time_per_search * 500))
        
        cur_adj, cur_deg, cur_score, cur_worst = local_search(
            adj, deg, score, worst, max_iters, deadline - 15
        )
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = cur_adj.copy()
            best_deg = cur_deg.copy()
            best_worst = cur_worst
    
    # Final degree-ordered polish
    if best_adj is not None:
        pol_adj, pol_deg, pol_score, pol_worst = degree_ordered_polish(
            best_adj, best_deg, best_score, best_worst, deadline - 1
        )
        if pol_score < best_score:
            best_score = pol_score
            best_adj = pol_adj.copy()
            best_deg = pol_deg.copy()
            best_worst = pol_worst
        
        # One more round of local search from polished state
        if time.time() < deadline - 5:
            cur_adj, cur_deg, cur_score, cur_worst = local_search(
                best_adj, best_deg, best_score, best_worst, 2000, deadline - 2
            )
            if cur_score < best_score:
                best_score = cur_score
                best_adj = cur_adj.copy()
                best_deg = cur_deg.copy()
                best_worst = cur_worst
        
        # Final polish
        if time.time() < deadline - 2:
            pol_adj, pol_deg, pol_score, pol_worst = degree_ordered_polish(
                best_adj, best_deg, best_score, best_worst, deadline
            )
            if pol_score < best_score:
                best_score = pol_score
                best_adj = pol_adj.copy()
                best_deg = pol_deg.copy()
                best_worst = pol_worst
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
