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
    
    def find_best_edge_removal_for_triangle(u, v, w, adj, deg, current_edges):
        """Find which edge to remove from triangle (u,v,w) to minimize impact."""
        best_remove = None
        best_new_score = float('inf')
        
        for e in [(u, v), (u, w), (v, w)]:
            a, b = min(e), max(e)
            if (a, b) not in current_edges:
                continue
            
            # Simulate removal
            old_adj_a = adj[a]
            old_adj_b = adj[b]
            old_deg_a = deg[a]
            old_deg_b = deg[b]
            
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            deg[a] -= 1
            deg[b] -= 1
            
            # Find max triangle sum after removal
            max_sum = 0
            for idx in vert_to_tris[a]:
                x, y, z = all_tris[idx]
                if (adj[x] >> y) & 1 and (adj[x] >> z) & 1 and (adj[y] >> z) & 1:
                    s = deg[x] + deg[y] + deg[z]
                    if s > max_sum:
                        max_sum = s
            for idx in vert_to_tris[b]:
                x, y, z = all_tris[idx]
                if (adj[x] >> y) & 1 and (adj[x] >> z) & 1 and (adj[y] >> z) & 1:
                    s = deg[x] + deg[y] + deg[z]
                    if s > max_sum:
                        max_sum = s
            
            # Restore
            adj[a] = old_adj_a
            adj[b] = old_adj_b
            deg[a] = old_deg_a
            deg[b] = old_deg_b
            
            if max_sum < best_new_score:
                best_new_score = max_sum
                best_remove = (a, b)
        
        return best_remove
    
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
        n_iter_total = 400000
        
        while time.time() < restart_deadline:
            i += 1
            # Faster cooling in first half, slower in second half for better exploration
            progress = min(i, n_iter_total) / n_iter_total
            if progress < 0.5:
                T = T_start * (T_end / T_start) ** (progress * 2)
            else:
                T = T_start * (T_end / T_start) ** (0.5 + (progress - 0.5) * 1.5)
            
            # Move Selection
            # 1. Pick edge to remove - bias toward edges in high-sum triangles
            if current_score > 0:
                # Find triangles with max or near-max sum
                max_val = current_score
                # Consider triangles with sum >= max_val - 2 for more diversity
                threshold = max_val - 2
                candidates = [idx for idx, s in enumerate(tri_sum) if s >= threshold]
                
                if candidates and rng.random() < 0.7:
                    idx = rng.choice(candidates)
                    u, v, w = all_tris[idx]
                    # Try to find best edge to remove from this triangle
                    best_remove = find_best_edge_removal_for_triangle(u, v, w, adj, deg, current_edges)
                    if best_remove:
                        e_remove = best_remove
                    else:
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
            
            # Bias toward adding edges that connect lower-degree vertices
            if rng.random() < 0.4:
                # Weight by inverse degree product
                weights = []
                for ne in non_edges:
                    a, b = ne
                    w_val = 1.0 / (max(1, deg[a]) * max(1, deg[b]))
                    weights.append(w_val)
                total_w = sum(weights)
                if total_w > 0:
                    r = rng.random() * total_w
                    cumsum = 0
                    for ne, w_val in zip(non_edges, weights):
                        cumsum += w_val
                        if r <= cumsum:
                            e_add = ne
                            break
                    else:
                        e_add = rng.choice(non_edges)
                else:
                    e_add = rng.choice(non_edges)
            else:
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
            if delta <= 0 or (T > 0.01 and rng.random() < math.exp(-delta / max(T, 1e-10))):
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
    
    # Final polish pass on the best solution found
    if best_overall_edges is not None and time.time() < deadline - 5:
        polish_deadline = deadline - 2
        
        adj = [0] * n
        deg = [0] * n
        for u, v in best_overall_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        
        tri_sum = compute_tri_sums(adj, deg)
        current_score = max(tri_sum)
        best_score = current_score
        best_edges = set(best_overall_edges)
        
        rng = random.Random(99999)
        non_edges = get_non_edges(adj)
        
        # Deterministic local improvement: try all single edge swaps
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            
            improved = False
            max_val = current_score
            
            # Try removing each edge in high-sum triangles
            high_tris = [idx for idx, s in enumerate(tri_sum) if s >= max_val - 1]
            
            for idx in high_tris[:20]:  # Limit to avoid timeout
                u, v, w = all_tris[idx]
                for e_remove in [(u, v), (u, w), (v, w)]:
                    a, b = min(e_remove), max(e_remove)
                    if (a, b) not in best_edges:
                        continue
                    
                    # Try adding each non-edge
                    for e_add in non_edges:
                        c, d = e_add
                        if (c, d) == (a, b):
                            continue
                        
                        # Simulate
                        old_adj_a, old_adj_b = adj[a], adj[b]
                        old_adj_c, old_adj_d = adj[c], adj[d]
                        old_deg_a, old_deg_b = deg[a], deg[b]
                        old_deg_c, old_deg_d = deg[c], deg[d]
                        
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        deg[a] -= 1
                        deg[b] -= 1
                        
                        adj[c] |= (1 << d)
                        adj[d] |= (1 << c)
                        deg[c] += 1
                        deg[d] += 1
                        
                        # Compute new max
                        new_max = 0
                        for v_idx in (a, b, c, d):
                            for t_idx in vert_to_tris[v_idx]:
                                x, y, z = all_tris[t_idx]
                                if (adj[x] >> y) & 1 and (adj[x] >> z) & 1 and (adj[y] >> z) & 1:
                                    s = deg[x] + deg[y] + deg[z]
                                    if s > new_max:
                                        new_max = s
                        
                        # Restore
                        adj[a] = old_adj_a
                        adj[b] = old_adj_b
                        adj[c] = old_adj_c
                        adj[d] = old_adj_d
                        deg[a] = old_deg_a
                        deg[b] = old_deg_b
                        deg[c] = old_deg_c
                        deg[d] = old_deg_d
                        
                        if new_max < best_score:
                            # Apply the move
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                            deg[a] -= 1
                            deg[b] -= 1
                            best_edges.remove((a, b))
                            
                            adj[c] |= (1 << d)
                            adj[d] |= (1 << c)
                            deg[c] += 1
                            deg[d] += 1
                            best_edges.add((c, d))
                            
                            non_edges.remove((c, d))
                            non_edges.append((a, b))
                            
                            tri_sum = compute_tri_sums(adj, deg)
                            current_score = max(tri_sum)
                            best_score = current_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges
    
    if best_overall_edges is None:
        # Fallback: use the initial construction
        adj, deg, current_edges = build_initial_graph(42)
        best_overall_edges = current_edges
    
    edges_list = [list(e) for e in best_overall_edges]
    return {'n': n, 'edges': edges_list}
