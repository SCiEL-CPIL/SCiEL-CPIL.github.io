"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses a 'Start Large, Fix, Prune' strategy.
    1. Generate random graph on 40 vertices.
    2. Remove edges to ensure triangle-freeness.
    3. Add edges to reduce independence number to <= 9.
    4. If alpha still > 9, remove vertices until valid.
    5. Repeat with different seeds to find max n.
    """
    max_time = 110  # Leave 10s for overhead
    start_time = time.time()
    best_result = None
    
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def max_independent_set_size(adj, n):
        # Branch and bound for MIS
        best = [0]
        
        def branch(current_size, candidates):
            if time.time() - start_time > max_time:
                return
            if current_size + len(candidates) <= best[0]:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pick vertex with highest degree in induced subgraph on candidates
            max_deg = -1
            v_idx = 0
            for i, v in enumerate(candidates):
                deg = 0
                for u in candidates:
                    if u != v and adj[v][u]:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    v_idx = i
            
            v = candidates[v_idx]
            
            # Branch 1: Include v
            new_candidates = [u for u in candidates if u != v and not adj[v][u]]
            branch(current_size + 1, new_candidates)
            
            # Branch 2: Exclude v
            branch(current_size, candidates[:v_idx] + candidates[v_idx+1:])
            
        branch(0, list(range(n)))
        return best[0]

    def get_mis(adj, n):
        best_set = []
        best_size = 0
        
        def branch(current_set, candidates):
            nonlocal best_set, best_size
            if len(current_set) + len(candidates) <= best_size:
                return
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            v = candidates[0]
            # Include v
            new_candidates = [u for u in candidates if u != v and not adj[v][u]]
            current_set.append(v)
            branch(current_set, new_candidates)
            current_set.pop()
            
            # Exclude v
            branch(current_set, candidates[1:])
            
        branch([], list(range(n)))
        return best_set

    def try_one(seed):
        rng = random.Random(seed)
        n = 40
        # Init random graph
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Step 1: Remove triangles
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if changed: break
                for j in range(i+1, n):
                    if adj[i][j]:
                        common = []
                        for k in range(n):
                            if k != i and k != j and adj[i][k] and adj[j][k]:
                                common.append(k)
                        if common:
                            adj[i][j] = False
                            adj[j][i] = False
                            changed = True
                            break
        
        # Step 2: Reduce alpha
        for _ in range(500):
            if time.time() - start_time > max_time:
                return None
            
            alpha = max_independent_set_size(adj, n)
            if alpha <= 9:
                break
            
            mis = get_mis(adj, n)
            
            # Try to add an edge (u, v) from MIS such that no triangle is formed
            found_edge = False
            for i in range(len(mis)):
                for j in range(i+1, len(mis)):
                    u = mis[i]
                    v = mis[j]
                    if not adj[u][v]:
                        # Check if adding (u,v) creates triangle
                        common = []
                        for w in range(n):
                            if w != u and w != v and adj[u][w] and adj[v][w]:
                                common.append(w)
                        if not common:
                            adj[u][v] = True
                            adj[v][u] = True
                            found_edge = True
                            break
                if found_edge:
                    break
            
            if not found_edge:
                # Remove a vertex from MIS
                v_to_remove = mis[0]
                new_n = n - 1
                new_adj = [[False]*new_n for _ in range(new_n)]
                idx_map = {}
                new_idx = 0
                for i in range(n):
                    if i != v_to_remove:
                        idx_map[i] = new_idx
                        new_idx += 1
                
                for i in range(n):
                    if i == v_to_remove: continue
                    for j in range(i+1, n):
                        if j == v_to_remove: continue
                        if adj[i][j]:
                            ni = idx_map[i]
                            nj = idx_map[j]
                            new_adj[ni][nj] = True
                            new_adj[nj][ni] = True
                            
                adj = new_adj
                n = new_n
                
                if n < 10:
                    return None

        if max_independent_set_size(adj, n) <= 9 and is_triangle_free(adj, n):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        return None

    for seed in range(100):
        if time.time() - start_time > max_time:
            break
        result = try_one(seed)
        if result:
            if best_result is None or result['n'] > best_result['n']:
                best_result = result
                if best_result['n'] >= 39:
                    break

    if best_result:
        return best_result
    
    # Fallback: Andrásfai(9) on 26 vertices
    n = 26
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
