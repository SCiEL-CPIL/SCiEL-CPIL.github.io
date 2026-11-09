import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    target_edges = 65
    
    # Precompute all triangles
    all_tris = list(itertools.combinations(range(n), 3))
    tri_to_idx = {t: i for i, t in enumerate(all_tris)}
    num_tris = len(all_tris)
    
    # Precompute for each edge, the list of triangle indices containing it
    # And for each vertex, the list of triangle indices containing it
    edge_to_tris = {}
    vert_to_tris = {v: [] for v in range(n)}
    for i, (u, v, w) in enumerate(all_tris):
        for e in [(u, v), (u, w), (v, w)]:
            e_sorted = (min(e), max(e))
            edge_to_tris.setdefault(e_sorted, []).append(i)
        for vert in (u, v, w):
            vert_to_tris[vert].append(i)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    num_all_edges = len(all_edges)
    
    def build_initial_graph(seed):
        """Build K_{8,8} + one random intra-partition edge."""
        adj = [0] * n
        deg = [0] * n
        edges = set()
        
        rng_local = random.Random(seed)
        
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.add((u, v))
        
        # Add one random intra-partition edge
        while True:
            u = rng_local.randint(0, 7)
            v = rng_local.randint(0, 7)
            if u != v:
                e = (min(u, v), max(u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.add(e)
                break
        
        return adj, deg, edges
    
    def compute_tri_sums(adj, deg):
        """Compute triangle sums for all triangles."""
        tri_sum = [0] * num_tris
        for i, (u, v, w) in enumerate(all_tris):
            if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                tri_sum[i] = deg[u] + deg[v] + deg[w]
        return tri_sum
    
    def get_non_edges(adj):
        """Return list of non-edges."""
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        return non_edges
    
    # Run multiple restarts
    best_overall_score = float('inf')
    best_overall_edges = None
    
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        restart_deadline = time.time() + time_per_restart - 0.5
        
        # Initialize
        adj, deg, current_edges = build_initial_graph(42 + restart * 1000)
        tri_sum = compute_tri_sums(adj, deg)
        current_score = max(tri_sum)
        
        best_score = current_score
        best_adj = list(adj)
        best_deg = list(deg)
        best_edges = set(current_edges)
        
        rng = random.Random(42 + restart * 7919)
        
        T_start = 8.0
        T_end = 0.005
        
        # Precompute non-edges for faster sampling
        non_edges = get_non_edges(adj)
        
        i = 0
        n_iter_total = 300000
        
        while time.time() < restart_deadline:
            i += 1
            T = T_start * (T_end / T_start) ** (min(i, n_iter_total) / n_iter_total)
            
            # Move Selection
            # 1. Pick edge to remove - bias toward edges in high-sum triangles
            if rng.random() < 0.6 and current_score > 0:
                # Find a triangle with max sum
                max_val = current_score
                candidates = [idx for idx, s in enumerate(tri_sum) if s == max_val]
                if candidates:
                    idx = rng.choice(candidates)
                    u, v, w = all_tris[idx]
                    e_remove = rng.choice([(u, v), (u, w), (v, w)])
                    e_remove = (min(e_remove), max(e_remove))
                else:
                    e_remove = rng.choice(list(current_edges))
                    e_remove = (min(e_remove), max(e_remove))
            else:
                e_remove = rng.choice(list(current_edges))
                e_remove = (min(e_remove), max(e_remove))
                
            # 2. Pick edge to add from non_edges list
            if not non_edges:
                break
            e_add = rng.choice(non_edges)
            
            if e_add == e_remove:
                continue
                
            u1, v1 = e_remove
            u2, v2 = e_add
            
            prev_score = current_score
            
            # Apply Move
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            deg[u1] -= 1
            deg[v1] -= 1
            current_edges.remove(e_remove)
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            deg[u2] += 1
            deg[v2] += 1
            current_edges.add(e_add)
            
            # Update non_edges list
            non_edges.remove(e_add)
            non_edges.append(e_remove)
            
            # Update Triangle Sums Incrementally
            # Triangles affected: those containing (u1,v1) or (u2,v2) or any of u1,v1,u2,v2
            update_indices = set()
            
            # Triangles containing the removed edge
            if e_remove in edge_to_tris:
                update_indices.update(edge_to_tris[e_remove])
            # Triangles containing the added edge
            if e_add in edge_to_tris:
                update_indices.update(edge_to_tris[e_add])
            # Triangles containing any affected vertex (for degree sum changes)
            for vert in (u1, v1, u2, v2):
                update_indices.update(vert_to_tris[vert])
            
            # Update sums for these indices
            for idx in update_indices:
                u, v, w = all_tris[idx]
                if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    tri_sum[idx] = deg[u] + deg[v] + deg[w]
                else:
                    tri_sum[idx] = 0
                    
            current_score = max(tri_sum)
            
            # Acceptance
            delta = current_score - prev_score
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                if current_score < best_score:
                    best_score = current_score
                    best_adj = list(adj)
                    best_deg = list(deg)
                    best_edges = set(current_edges)
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                deg[u1] += 1
                deg[v1] += 1
                current_edges.add(e_remove)
                
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                deg[u2] -= 1
                deg[v2] -= 1
                current_edges.remove(e_add)
                
                # Revert non_edges
                non_edges.remove(e_remove)
                non_edges.append(e_add)
                
                # Revert triangle sums
                for idx in update_indices:
                    u, v, w = all_tris[idx]
                    if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                        tri_sum[idx] = deg[u] + deg[v] + deg[w]
                    else:
                        tri_sum[idx] = 0
                        
                current_score = prev_score
                
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges
    
    if best_overall_edges is None:
        # Fallback: use the initial construction
        adj, deg, current_edges = build_initial_graph(42)
        best_overall_edges = current_edges
    
    edges_list = [list(e) for e in best_overall_edges]
    return {'n': n, 'edges': edges_list}
