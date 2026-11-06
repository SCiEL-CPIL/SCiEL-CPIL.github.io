import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 18
    
    def adj_to_bitmask(adj):
        """Convert adjacency matrix to bitmask representation."""
        bm = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    bm |= (1 << (i*n + j)) | (1 << (j*n + i))
        return bm
    
    def bitmask_to_adj(bm):
        """Convert bitmask to adjacency matrix."""
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if bm & (1 << (i*n + j)):
                    adj[i][j] = adj[j][i] = True
        return adj
    
    def max_triangle_free_induced_bm(bm, timeout=5.0):
        """Exact branch and bound for max triangle-free induced subgraph size using bitmask."""
        best = 0
        start = time.time()
        
        # Precompute: for each vertex, its neighbors as bitmask
        neighbors = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (bm & (1 << (i*n + j))):
                    neighbors[i] |= (1 << j)
        
        # Triangle conflict: for each pair (i,j) that are connected, which k form triangle
        # tri_conflict[i][j] = set of k such that i-k and j-k are edges
        tri_conflict = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if bm & (1 << (i*n + j)):
                    common = neighbors[i] & neighbors[j] & ~(1<<i) & ~(1<<j)
                    tri_conflict[i][j] = common
                    tri_conflict[j][i] = common
        
        def dfs(current_mask, candidates_mask):
            nonlocal best
            if time.time() - start > timeout:
                return
            
            current_size = bin(current_mask).count('1')
            cand_size = bin(candidates_mask).count('1')
            
            if current_size + cand_size <= best:
                return
            if cand_size == 0:
                best = max(best, current_size)
                return
            
            # Pick a candidate vertex with maximum conflicts (more pruning)
            # Compute conflict mask for each candidate: which other candidates would be removed
            # if we add this vertex
            best_v = -1
            best_conflict_count = -1
            v_conflicts = {}
            temp_mask = candidates_mask
            while temp_mask:
                lb = temp_mask & -temp_mask
                v = lb.bit_length() - 1
                # Count how many candidates would conflict with v given current set
                # v conflicts with w if there exists u in current such that (v,u) edge and (w,u) edge
                # Actually: v can't be added if any pair (i,j) in current has v in tri_conflict[i][j]
                # For pruning: adding v removes candidates w where for some u in current,
                # tri_conflict[v][u] contains w, OR for some pair (u1,u2) in current, tri_conflict[u1][u2] contains w
                # Simpler: just pick first vertex for speed
                v_conflicts[v] = 0
                temp_mask &= temp_mask - 1
            
            # Use first set bit for speed
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            v_bit = 1 << v
            remaining = candidates_mask & ~v_bit
            
            # Check if v can be added: for all pairs (i,j) in current, v not in tri_conflict[i][j]
            can_add = True
            cur_list = [i for i in range(n) if current_mask & (1<<i)]
            k = len(cur_list)
            for idx in range(k):
                i = cur_list[idx]
                for jdx in range(idx+1, k):
                    j = cur_list[jdx]
                    if tri_conflict[i][j] & v_bit:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current_mask | v_bit
                # Filter remaining using precomputed conflict masks
                # A candidate w stays if:
                # 1. For all pairs (u1,u2) in cur_list: w not in tri_conflict[u1][u2]
                # 2. For all u in cur_list: w not in tri_conflict[v][u]
                # Precompute base conflict: union of tri_conflict[u1][u2] for all pairs in cur_list
                base_conflict = 0
                for idx in range(k):
                    i = cur_list[idx]
                    for jdx in range(idx+1, k):
                        j = cur_list[jdx]
                        base_conflict |= tri_conflict[i][j]
                # Add conflicts with v
                for idx in range(k):
                    u = cur_list[idx]
                    base_conflict |= tri_conflict[v][u]
                
                new_remaining = remaining & ~base_conflict
                dfs(new_current, new_remaining)
            
            dfs(current_mask, remaining)
        
        dfs(0, (1<<n)-1)
        return best
    
    def greedy_triangle_free_size_bm(bm, trials=50):
        """Greedy estimate of max triangle-free induced subgraph size."""
        neighbors = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (bm & (1 << (i*n + j))):
                    neighbors[i] |= (1 << j)
        
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                can_add = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if neighbors[v] & (1<<current[i]) and neighbors[v] & (1<<current[j]) and \
                           neighbors[current[i]] & (1<<current[j]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current.append(v)
            best = max(best, len(current))
        return best
    
    # Start with K6,6,6
    adj = np.zeros((n, n), dtype=bool)
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = adj[b][a] = True
    
    current_bm = adj_to_bitmask(adj)
    current_score = max_triangle_free_induced_bm(current_bm, timeout=3.0)
    best_bm = current_bm
    best_score = current_score
    
    def try_edge_change_bm(bm, u, v, add):
        """Try adding or removing edge (u,v), return new bm if K4-free, else None."""
        edge_bit = (1 << (u*n + v)) | (1 << (v*n + u))
        if add:
            if bm & edge_bit:
                return None
            new_bm = bm | edge_bit
            # Check K4-freeness: find common neighbors of u,v that form a clique
            neighbors_u = 0
            neighbors_v = 0
            for w in range(n):
                if w != u and w != v and (bm & (1 << (u*n + w))):
                    neighbors_u |= (1 << w)
                if w != u and w != v and (bm & (1 << (v*n + w))):
                    neighbors_v |= (1 << w)
            common = neighbors_u & neighbors_v
            # Check if any pair in common are connected
            common_list = [w for w in range(n) if common & (1<<w)]
            for i in range(len(common_list)):
                for j in range(i+1, len(common_list)):
                    if bm & (1 << (common_list[i]*n + common_list[j])):
                        return None
            return new_bm
        else:
            if not (bm & edge_bit):
                return None
            return bm & ~edge_bit
    
    # Simulated annealing
    T = 2.0
    cooling = 0.995
    no_improve_count = 0
    
    while time.time() < deadline:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        add = not (current_bm & (1 << (u*n + v)))
        new_bm = try_edge_change_bm(current_bm, u, v, add)
        if new_bm is None:
            T *= cooling
            no_improve_count += 1
            continue
        
        new_score_greedy = greedy_triangle_free_size_bm(new_bm, trials=30)
        
        accept = False
        if new_score_greedy < current_score:
            accept = True
        elif T > 0.01:
            delta = new_score_greedy - current_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                accept = True
        
        if accept:
            new_score = max_triangle_free_induced_bm(new_bm, timeout=2.0)
            if new_score < current_score:
                current_bm = new_bm
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_bm = new_bm
                    best_score = new_score
            elif T > 0.01:
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    current_bm = new_bm
                    current_score = new_score
                    no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling
        if no_improve_count > 100:
            T = max(T, 0.5)
            no_improve_count = 0
    
    # Final polish: more thorough local search on best_bm
    if time.time() < deadline - 15:
        polish_bm = best_bm
        polish_score = max_triangle_free_induced_bm(polish_bm, timeout=5.0)
        
        # Try all single edge flips that improve
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if time.time() >= deadline - 5:
                        break
                    add = not (polish_bm & (1 << (u*n + v)))
                    new_bm = try_edge_change_bm(polish_bm, u, v, add)
                    if new_bm is None:
                        continue
                    new_score = max_triangle_free_induced_bm(new_bm, timeout=1.5)
                    if new_score < polish_score:
                        polish_bm = new_bm
                        polish_score = new_score
                        improved = True
                        if polish_score < best_score:
                            best_bm = polish_bm
                            best_score = polish_score
            if time.time() >= deadline - 5:
                break
    
    # Final verification
    final_score = max_triangle_free_induced_bm(best_bm, timeout=10.0)
    if final_score < best_score:
        best_score = final_score
    
    # Return best
    best_adj = bitmask_to_adj(best_bm)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
