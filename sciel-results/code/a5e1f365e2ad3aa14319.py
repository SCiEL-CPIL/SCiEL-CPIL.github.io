import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    n = 16
    m_target = 65
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    num_all_edges = len(all_edges)  # 120
    
    # Precompute triangles: list of (u, v, w) triples
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_triangles = len(triangles)  # 560
    
    # Precompute for each edge, which triangles it belongs to
    # edge_triangles[(u,v)] = list of triangle indices
    edge_triangles = {}
    for i, (u, v, w) in enumerate(triangles):
        for e in [(u, v), (u, w), (v, w)]:
            if e not in edge_triangles:
                edge_triangles[e] = []
            edge_triangles[e].append(i)
    
    # Precompute for each vertex, its degree contribution to each triangle
    # vertex_triangles[v] = list of (triangle_index, other_two_vertices)
    # Actually, let's precompute triangle membership for each vertex
    vertex_in_triangles = {v: [] for v in range(n)}
    for i, (u, v, w) in enumerate(triangles):
        vertex_in_triangles[u].append(i)
        vertex_in_triangles[v].append(i)
        vertex_in_triangles[w].append(i)
    
    def calc_score(adj_mask, deg):
        """
        Fast score calculation using bitmasks.
        """
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    if common:
                        w_mask = common
                        while w_mask:
                            lsb = w_mask & -w_mask
                            w = lsb.bit_length() - 1
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                            w_mask ^= lsb
        return max_sum
    
    def generate_k610_structure():
        """Generate K_{6,10} + 5 matching edges structure."""
        adj_mask = [0] * n
        deg = [0] * n
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:6]
        B = vertices[6:]
        
        for u in A:
            for v in B:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        random.shuffle(B)
        for i in range(5):
            u, v = B[2*i], B[2*i+1]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        return adj_mask, deg
    
    def local_search_with_constraint(adj_mask, deg, target_score, max_iter=20000, time_limit=10):
        """
        Local search that tries to achieve score <= target_score.
        Uses a constraint-based approach: prioritize moves that reduce high-degree-sum triangles.
        """
        current_score = calc_score(adj_mask, deg)
        best_score = current_score
        best_adj_mask = adj_mask.copy()
        best_deg = deg.copy()
        
        if current_score <= target_score:
            return best_adj_mask, best_deg, best_score
        
        edges_set = set()
        non_edges_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges_set.add((u, v))
                else:
                    non_edges_set.add((u, v))
        
        edges_list = list(edges_set)
        non_edges_list = list(non_edges_set)
        
        # Precompute for each edge the triangles it's in (for quick assessment)
        # For each potential swap, estimate the score change
        
        no_improve_count = 0
        T = 0.3
        cooling = 0.9995
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            if not edges_list or not non_edges_list:
                break
            
            # Choose a move: remove edge (u,v), add edge (x,y)
            # Prioritize removing edges that are in high-degree-sum triangles
            u, v = random.choice(edges_list)
            x, y = random.choice(non_edges_list)
            
            # Apply swap
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            adj_mask[x] ^= (1 << y)
            adj_mask[y] ^= (1 << x)
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            new_score = calc_score(adj_mask, deg)
            
            accept = False
            if new_score < current_score:
                accept = True
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_adj_mask = adj_mask.copy()
                    best_deg = deg.copy()
            elif new_score == current_score:
                if random.random() < 0.2:
                    accept = True
                    no_improve_count += 1
            else:
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    accept = True
                    no_improve_count += 1
            
            if accept:
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
                
                edges_list = list(edges_set)
                non_edges_list = list(non_edges_list)
            else:
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                adj_mask[x] ^= (1 << y)
                adj_mask[y] ^= (1 << x)
                
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
            
            T *= cooling
            if T < 0.03:
                T = 0.03
            
            if no_improve_count > 300:
                # Perturb
                for _ in range(5):
                    if not edges_list or not non_edges_list:
                        break
                    u2, v2 = random.choice(edges_list)
                    x2, y2 = random.choice(non_edges_list)
                    
                    adj_mask[u2] ^= (1 << v2)
                    adj_mask[v2] ^= (1 << u2)
                    adj_mask[x2] ^= (1 << y2)
                    adj_mask[y2] ^= (1 << x2)
                    
                    deg[u2] -= 1
                    deg[v2] -= 1
                    deg[x2] += 1
                    deg[y2] += 1
                    
                    edges_set.remove((u2, v2))
                    non_edges_set.add((u2, v2))
                    edges_set.add((x2, y2))
                    non_edges_set.remove((x2, y2))
                    
                    edges_list = list(edges_set)
                    non_edges_list = list(non_edges_list)
                
                current_score = calc_score(adj_mask, deg)
                no_improve_count = 0
                T = 0.3
        
        return best_adj_mask, best_deg, best_score
    
    def branch_and_bound(adj_mask, deg, target_score, depth=0, max_depth=8, time_limit=10):
        """
        Custom branch-and-bound heuristic.
        Tries to find a graph with score <= target_score.
        """
        current_score = calc_score(adj_mask, deg)
        if current_score <= target_score:
            return adj_mask, deg, current_score
        
        if depth >= max_depth:
            return adj_mask, deg, current_score
        
        if time.time() - start_time > time_limit:
            return adj_mask, deg, current_score
        
        # Find the worst triangle (highest degree sum)
        worst_triangle = None
        worst_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    if common:
                        w_mask = common
                        while w_mask:
                            lsb = w_mask & -w_mask
                            w = lsb.bit_length() - 1
                            s = deg[u] + deg[v] + deg[w]
                            if s > worst_sum:
                                worst_sum = s
                                worst_triangle = (u, v, w)
                            w_mask ^= lsb
        
        if worst_triangle is None:
            return adj_mask, deg, current_score
        
        u, v, w = worst_triangle
        
        # Try removing each of the three edges in the worst triangle
        # and adding a random non-edge to maintain edge count
        edges_set = set()
        non_edges_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges_set.add((i, j))
                else:
                    non_edges_set.add((i, j))
        
        if not non_edges_set:
            return adj_mask, deg, current_score
        
        non_edges_list = list(non_edges_set)
        
        best_result = (adj_mask, deg, current_score)
        
        # Try removing each edge in the worst triangle
        for edge_to_remove in [(u, v), (u, w), (v, w)]:
            if edge_to_remove not in edges_set:
                continue
            
            # Try a few random non-edges to add
            for _ in range(3):
                if not non_edges_list:
                    break
                edge_to_add = random.choice(non_edges_list)
                
                # Check if adding this edge creates issues
                new_adj_mask = adj_mask.copy()
                new_deg = deg.copy()
                
                a, b = edge_to_remove
                c, d = edge_to_add
                
                new_adj_mask[a] ^= (1 << b)
                new_adj_mask[b] ^= (1 << a)
                new_adj_mask[c] ^= (1 << d)
                new_adj_mask[d] ^= (1 << c)
                
                new_deg[a] -= 1
                new_deg[b] -= 1
                new_deg[c] += 1
                new_deg[d] += 1
                
                new_score = calc_score(new_adj_mask, new_deg)
                
                if new_score < best_result[2]:
                    best_result = (new_adj_mask, new_deg, new_score)
                    if new_score <= target_score:
                        return best_result
        
        return best_result
    
    # Main search
    best_overall_score = float('inf')
    best_overall_adj_mask = None
    best_overall_deg = None
    
    # Phase 1: Try multiple K_{6,10} based initializations with local search
    num_restarts = 8
    elapsed = time.time() - start_time
    time_per_restart = (deadline - 2 - elapsed) / num_restarts
    time_per_restart = min(time_per_restart, 14)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        adj_mask, deg = generate_k610_structure()
        
        # Apply random perturbations for diversity
        if restart > 0:
            edges_set = set((u, v) for u in range(n) for v in range(u + 1, n) if (adj_mask[u] >> v) & 1)
            non_edges_set = set((u, v) for u in range(n) for v in range(u + 1, n) if not ((adj_mask[u] >> v) & 1))
            
            for _ in range(random.randint(1, 4)):
                if not edges_set or not non_edges_set:
                    break
                u, v = random.choice(list(edges_set))
                x, y = random.choice(list(non_edges_set))
                
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                adj_mask[x] ^= (1 << y)
                adj_mask[y] ^= (1 << x)
                
                deg[u] -= 1
                deg[v] -= 1
                deg[x] += 1
                deg[y] += 1
                
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
        
        # Local search
        adj_ls, deg_ls, score_ls = local_search_with_constraint(adj_mask, deg, 23, max_iter=8000, time_limit=time_per_restart)
        
        if score_ls < best_overall_score:
            best_overall_score = score_ls
            best_overall_adj_mask = adj_ls
            best_overall_deg = deg_ls
            
            if best_overall_score <= 23:
                break
    
    # Phase 2: If we haven't achieved score <= 23, try branch-and-bound from best found
    if best_overall_score > 23 and best_overall_adj_mask is not None:
        remaining_time = deadline - 3 - (time.time() - start_time)
        if remaining_time > 2:
            adj_bb, deg_bb, score_bb = branch_and_bound(
                best_overall_adj_mask, best_overall_deg, 23, 
                depth=0, max_depth=10, time_limit=remaining_time
            )
            if score_bb < best_overall_score:
                best_overall_score = score_bb
                best_overall_adj_mask = adj_bb
                best_overall_deg = deg_bb
    
    # Phase 3: Final polish with more aggressive local search
    if best_overall_adj_mask is not None:
        remaining_time = deadline - 2 - (time.time() - start_time)
        if remaining_time > 1:
            adj_final, deg_final, score_final = local_search_with_constraint(
                best_overall_adj_mask, best_overall_deg, 23,
                max_iter=10000, time_limit=remaining_time
            )
            if score_final < best_overall_score:
                best_overall_score = score_final
                best_overall_adj_mask = adj_final
                best_overall_deg = deg_final
    
    # Fallback
    if best_overall_adj_mask is None:
        adj_mask, deg = generate_k610_structure()
        best_overall_adj_mask = adj_mask
        best_overall_deg = deg
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_overall_adj_mask[i] >> j) & 1:
                final_edges.append([i, j])
    
    return {'n': n, 'edges': final_edges}
