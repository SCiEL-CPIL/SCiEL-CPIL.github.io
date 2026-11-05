import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    n = 16
    
    # Fast 3-colorability check using bitmasks
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: (-bin(adj[x]).count('1'), x))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        return (adj[u] & adj[v]) != 0
    
    # Generate Clebsch Graph (40 edges, known 4-chromatic)
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)
    best_adj = clebsch_adj[:]
    
    # Verify baseline
    if is_3_colorable(best_adj):
        best_edges = []
        best_count = 0
        best_adj = [0] * n
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def random_maximal_tf():
        """Generate a random maximal triangle-free graph."""
        adj = [0] * n
        edges = []
        possible_edges = all_pairs[:]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges
    
    def local_search(adj, edges):
        """Try to improve by edge swaps: remove one edge, add another."""
        adj = adj[:]
        edge_set = set(edges)
        current_count = len(edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edge_set and (v, u) not in edge_set and not creates_triangle(adj, u, v)]
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                # Try removing each edge that shares a vertex with (u,v)
                for eu, ev in list(edge_set):
                    if eu == u or eu == v or ev == u or ev == v:
                        # Remove edge (eu, ev)
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                        edge_set.discard((eu, ev))
                        
                        # Try to add (u, v)
                        if not creates_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edge_set.add((u, v))
                            
                            if not is_3_colorable(adj):
                                current_count = len(edge_set)
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                edge_set.discard((u, v))
                        
                        # Revert removal
                        adj[eu] |= (1 << ev)
                        adj[ev] |= (1 << eu)
                        edge_set.add((eu, ev))
                
                if improved:
                    break
        
        if improved:
            new_edges = [list(e) for e in edge_set]
            return adj, new_edges, current_count
        return None, None, None
    
    random.seed(42)
    
    # Phase 1: Random maximal TF graphs
    attempts = 0
    while time.time() < deadline - 20 and attempts < 500:
        attempts += 1
        adj, edges = random_maximal_tf()
        
        if len(edges) > best_count:
            if not is_3_colorable(adj):
                best_edges = [list(e) for e in edges]
                best_count = len(edges)
                best_adj = adj
    
    # Phase 2: Local search from best found
    if time.time() < deadline - 5:
        adj_ls, edges_ls, count_ls = local_search(best_adj, best_edges)
        if adj_ls is not None and count_ls > best_count:
            best_adj = adj_ls
            best_edges = edges_ls
            best_count = count_ls
    
    # Phase 3: More random trials with local search on promising candidates
    while time.time() < deadline:
        adj, edges = random_maximal_tf()
        
        if len(edges) >= best_count:
            if not is_3_colorable(adj):
                # Try local search
                adj_ls, edges_ls, count_ls = local_search(adj, edges)
                if adj_ls is not None and count_ls > best_count:
                    best_adj = adj_ls
                    best_edges = edges_ls
                    best_count = count_ls
                elif len(edges) > best_count:
                    best_adj = adj
                    best_edges = [list(e) for e in edges]
                    best_count = len(edges)
    
    return best_edges
