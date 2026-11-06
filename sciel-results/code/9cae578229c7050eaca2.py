import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Precompute triangles: list of (u, v, w)
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    triangles = np.array(triangles, dtype=np.int32)
    num_triangles = len(triangles)
    
    # Precompute edge-to-triangle incidence: for each edge, which triangles contain it
    edge_to_tris = {}
    for idx, (u, v, w) in enumerate(triangles):
        for e in [(u, v), (u, w), (v, w)]:
            if e not in edge_to_tris:
                edge_to_tris[e] = []
            edge_to_tris[e].append(idx)
    
    def compute_score(adj, deg):
        """Compute the maximum degree sum over all triangles using bitmask."""
        if not np.any(adj):
            return 0, None
        max_sum = 0
        worst_tri = None
        for idx in range(num_triangles):
            u, v, w = triangles[idx]
            if adj[u, v] and adj[u, w] and adj[v, w]:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def compute_score_fast(adj, deg, prev_score, prev_worst):
        """Fast score: check if prev_worst still exists, then find new max."""
        if prev_worst is None:
            return compute_score(adj, deg)
        
        u, v, w = prev_worst
        if adj[u, v] and adj[u, w] and adj[v, w]:
            prev_sum = deg[u] + deg[v] + deg[w]
        else:
            prev_sum = -1
        
        max_sum = 0
        worst_tri = None
        for idx in range(num_triangles):
            u, v, w = triangles[idx]
            if adj[u, v] and adj[u, w] and adj[v, w]:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def make_initial_graph(seed):
        """Create a random near-bipartite graph with target_edges edges."""
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Random partition
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(7, 9)
        A = set(perm[:split])
        B = set(perm[split:])
        
        # Add all cross edges
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        remaining = target_edges - edge_count
        
        if remaining > 0:
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        internal_edges.append((part_list[i], part_list[j]))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            cross_edges = [(u, v) for u in A for v in B]
            random.shuffle(cross_edges)
            for u, v in cross_edges[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        deg = adj.sum(axis=1).astype(int)
        return adj, deg
    
    def local_search(adj, deg, score, worst_tri, max_iters, deadline):
        """Local search with edge swap moves, targeting worst triangles."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        T = 3.0
        cooling = 0.997
        min_T = 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Build candidate moves
            candidates = []
            
            if current_worst is not None:
                u, v, w = current_worst
                # Edges in worst triangle to remove
                tri_edges = [(u, v), (u, w), (v, w)]
                # Non-edges to add (prioritize edges that don't create high-degree triangles)
                non_edges = [(a, b) for a, b in all_edges if not current_adj[a, b]]
                
                for e1 in tri_edges:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    # Sample some non-edges
                    sample_size = min(20, len(non_edges))
                    sampled = random.sample(non_edges, sample_size)
                    for e2 in sampled:
                        candidates.append((e1, e2))
            
            # Also add some random moves
            for _ in range(5):
                e1 = random.choice(all_edges)
                if not current_adj[e1[0], e1[1]]:
                    continue
                non_edges = [(a, b) for a, b in all_edges if not current_adj[a, b]]
                if non_edges:
                    e2 = random.choice(non_edges)
                    candidates.append((e1, e2))
            
            if not candidates:
                continue
            
            # Evaluate candidates
            best_candidate = None
            best_delta = 0
            best_new_score = current_score
            best_new_worst = current_worst
            best_new_adj = None
            best_new_deg = None
            
            for e1, e2 in candidates:
                u1, v1 = e1
                u2, v2 = e2
                
                if current_adj[u1, v1] and not current_adj[u2, v2]:
                    new_adj = current_adj.copy()
                    new_deg = current_deg.copy()
                    
                    new_adj[u1, v1] = False
                    new_adj[v1, u1] = False
                    new_deg[u1] -= 1
                    new_deg[v1] -= 1
                    
                    new_adj[u2, v2] = True
                    new_adj[v2, u2] = True
                    new_deg[u2] += 1
                    new_deg[v2] += 1
                    
                    new_score, new_worst = compute_score(new_adj, new_deg)
                    delta = new_score - current_score
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_score = new_score
                        best_new_worst = new_worst
                        best_new_adj = new_adj
                        best_new_deg = new_deg
                        best_candidate = (e1, e2)
            
            if best_candidate is not None:
                # Accept with simulated annealing
                if best_delta < 0 or random.random() < np.exp(-best_delta / max(T, min_T)):
                    current_adj = best_new_adj
                    current_deg = best_new_deg
                    current_score = best_new_score
                    current_worst = best_new_worst
            
            T = max(T * cooling, min_T)
        
        return current_adj, current_deg, current_score, current_worst
    
    def polish(adj, deg, score, worst_tri, deadline):
        """Final polish: exhaustive local search on worst triangles."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            
            improved = False
            if current_worst is not None:
                u, v, w = current_worst
                tri_edges = [(u, v), (u, w), (v, w)]
                non_edges = [(a, b) for a, b in all_edges if not current_adj[a, b]]
                
                for e1 in tri_edges:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    for e2 in non_edges:
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        new_adj = current_adj.copy()
                        new_deg = current_deg.copy()
                        
                        new_adj[u1, v1] = False
                        new_adj[v1, u1] = False
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2, v2] = True
                        new_adj[v2, u2] = True
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score, new_worst = compute_score(new_adj, new_deg)
                        
                        if new_score < current_score:
                            current_adj = new_adj
                            current_deg = new_deg
                            current_score = new_score
                            current_worst = new_worst
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current_adj, current_deg, current_score, current_worst
    
    # Main search: multiple restarts
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_worst_tri = None
    
    num_restarts = 0
    for seed in range(200):
        if time.time() > deadline - 20:
            break
        
        adj, deg = make_initial_graph(seed)
        score, worst_tri = compute_score(adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_worst_tri = worst_tri
        
        # Local search
        current_adj, current_deg, current_score, current_worst = local_search(
            adj, deg, score, worst_tri, 3000, deadline
        )
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_deg = current_deg.copy()
            best_worst_tri = current_worst
        
        num_restarts += 1
    
    # Final polish
    if best_adj is not None:
        polished_adj, polished_deg, polished_score, polished_worst = polish(
            best_adj, best_deg, best_score, best_worst_tri, deadline
        )
        
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj.copy()
            best_deg = polished_deg.copy()
            best_worst_tri = polished_worst
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
