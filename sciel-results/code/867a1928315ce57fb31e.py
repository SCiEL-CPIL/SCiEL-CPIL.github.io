import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    def count_triangles(adj, mask, n):
        # Count triangles in the subgraph induced by vertices in mask
        verts = [i for i in range(n) if mask & (1 << i)]
        if len(verts) < 3:
            return 0
        count = 0
        for i in range(len(verts)):
            vi = verts[i]
            common = adj[vi] & mask
            common &= ~(1 << vi)
            common_list = [j for j in range(n) if common & (1 << j)]
            edges_in_common = 0
            for idx, u in enumerate(common_list):
                for w in common_list[idx+1:]:
                    if adj[u] & (1 << w):
                        edges_in_common += 1
            count += edges_in_common
        return count // 3

    def count_indep_sets_3(adj, mask, n):
        # Count independent sets of size 3 in the subgraph induced by vertices in mask
        # Equivalent to triangles in complement
        verts = [i for i in range(n) if mask & (1 << i)]
        if len(verts) < 3:
            return 0
        count = 0
        for i in range(len(verts)):
            vi = verts[i]
            # Non-neighbors of vi in mask
            all_mask = (1 << n) - 1
            non_neigh = (all_mask ^ (1 << vi) ^ adj[vi]) & mask
            non_neigh &= ~(1 << vi)
            non_neigh_list = [j for j in range(n) if non_neigh & (1 << j)]
            
            # Count non-edges among non_neigh_list
            # Total pairs - edges
            k = len(non_neigh_list)
            total_pairs = k * (k - 1) // 2
            edges = 0
            for idx, u in enumerate(non_neigh_list):
                for w in non_neigh_list[idx+1:]:
                    if adj[u] & (1 << w):
                        edges += 1
            count += (total_pairs - edges)
        return count // 3

    def get_deltas(adj, u, v, n):
        # Returns (delta_k5_if_add, delta_i5_if_add)
        # If we ADD edge (u,v):
        # Creates K5s: triangles in N(u) ∩ N(v)
        # Breaks I5s: indep sets of 3 in common non-neighbors
        
        common_neigh = adj[u] & adj[v]
        common_neigh &= ~(1 << u)
        common_neigh &= ~(1 << v)
        
        tri_k5 = count_triangles(adj, common_neigh, n)
        
        all_mask = (1 << n) - 1
        non_neigh_u = all_mask ^ (1 << u) ^ adj[u]
        non_neigh_v = all_mask ^ (1 << v) ^ adj[v]
        common_non_neigh = non_neigh_u & non_neigh_v
        common_non_neigh &= ~(1 << u)
        common_non_neigh &= ~(1 << v)
        
        tri_i5 = count_indep_sets_3(adj, common_non_neigh, n)
        
        return tri_k5, tri_i5

    def find_any_violation(adj, n):
        # Randomized search
        for _ in range(50):
            verts = random.sample(range(n), 5)
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if not (adj[verts[i]] & (1 << verts[j])):
                        is_clique = False
                        break
                if not is_clique: break
            if is_clique:
                return ('K5', verts)
                
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    if (adj[verts[i]] & (1 << verts[j])):
                        is_indep = False
                        break
                if not is_indep: break
            if is_indep:
                return ('I5', verts)
        
        # Systematic fallback
        for verts in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if not (adj[verts[i]] & (1 << verts[j])):
                        is_clique = False
                        break
                if not is_clique: break
            if is_clique:
                return ('K5', list(verts))
                
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    if (adj[verts[i]] & (1 << verts[j])):
                        is_indep = False
                        break
                if not is_indep: break
            if is_indep:
                return ('I5', list(verts))
                
        return None

    def local_search(adj, n, max_iters=2000):
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False, adj
            
            violation = find_any_violation(adj, n)
            if not violation:
                return True, adj
            
            v_type, verts = violation
            
            best_delta = float('inf')
            best_move = None
            
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = verts[i], verts[j]
                    
                    if v_type == 'K5':
                        # Remove edge
                        d_k5, d_i5 = get_deltas(adj, u, v, n)
                        delta = -d_k5 + d_i5
                    else:
                        # Add edge
                        d_k5, d_i5 = get_deltas(adj, u, v, n)
                        delta = d_k5 - d_i5
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (u, v, v_type)
                        
            if best_move:
                u, v, v_type = best_move
                if v_type == 'K5':
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                T *= cooling
                if T < min_T:
                    T = min_T
            else:
                break
                
        if not find_any_violation(adj, n):
            return True, adj
        return False, adj

    for n_target in [42, 41, 40, 39, 38]:
        if time.time() > deadline - 5:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            adj = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            success, adj = local_search(adj, n_target, max_iters=1000)
            if success:
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n_target, edges

    return q, seed_edges
