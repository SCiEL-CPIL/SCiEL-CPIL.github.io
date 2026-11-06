import random
import time
import itertools

def _get_time_left(deadline):
    return deadline - time.time()

def _is_k4_free(adj, n=18):
    # Check all 4-cliques using bitmasks
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                    continue
                for d in range(c+1, n):
                    if ((adj[a] >> d) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1):
                        return False
    return True

def _max_triangle_free_induced(adj, n=18):
    # Use branch and bound with bitmasks
    # First, find all triangles
    triangles = []
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if (adj[a] >> c) & 1 and (adj[b] >> c) & 1:
                    triangles.append((a, b, c))
    
    if not triangles:
        return n
    
    # Convert triangles to bitmasks for fast checking
    tri_masks = []
    for a, b, c in triangles:
        mask = (1 << a) | (1 << b) | (1 << c)
        tri_masks.append(mask)
    
    # Try to find largest triangle-free induced subgraph
    # Branch and bound: try to include as many vertices as possible
    # while avoiding all triangles
    
    best = 0
    all_vertices = (1 << n) - 1
    
    def dfs(remaining, selected, tri_idx):
        nonlocal best
        if tri_idx >= len(tri_masks):
            count = bin(selected).count('1')
            if count > best:
                best = count
            return
        
        # Pruning: even if we add all remaining, can't beat best
        if bin(selected).count('1') + bin(remaining).count('1') <= best:
            return
        
        mask = tri_masks[tri_idx]
        
        # Option 1: Exclude one of the vertices in this triangle
        # Try excluding each vertex
        for v in range(n):
            if (mask >> v) & 1:
                new_remaining = remaining & ~(1 << v)
                dfs(new_remaining, selected, tri_idx + 1)
        
        # Also try: if we can include all three, skip this triangle
        # (they're all in remaining, so no conflict)
        # Actually we need to be more careful - let's use a different approach
    
    # Better approach: iterative deepening or direct search
    # For n=18, we can try sizes from n down
    for size in range(n, 0, -1):
        # Check if there exists a subset of size 'size' with no triangles
        # Use backtracking
        found = False
        
        def backtrack(start, chosen, chosen_mask):
            nonlocal found
            if found:
                return True
            if len(chosen) == size:
                # Check if chosen has any triangle
                for tri in tri_masks:
                    if (tri & chosen_mask) == tri:
                        return False
                found = True
                return True
            
            for i in range(start, n):
                v = i
                # Pruning: not enough vertices left
                if len(chosen) + (n - i) < size:
                    break
                # Check if adding v creates a triangle with any pair in chosen
                valid = True
                for j in range(len(chosen)):
                    for k in range(j+1, len(chosen)):
                        u, w = chosen[j], chosen[k]
                        if (adj[u] >> w) & 1 and (adj[u] >> v) & 1 and (adj[w] >> v) & 1:
                            valid = False
                            break
                        if (adj[u] >> w) & 1 and (adj[w] >> v) & 1 and (adj[u] >> v) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                chosen.append(v)
                if backtrack(i + 1, chosen, chosen_mask | (1 << v)):
                    return True
                chosen.pop()
            
            return False
        
        if backtrack(0, [], 0):
            return size
    
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 18
    
    # Strategy: Use a blow-up of a small K4-free graph
    # Find small K4-free graphs with minimal max triangle-free induced subgraph
    # Then blow up each vertex to multiple copies
    
    # Predefined small K4-free graphs to try
    # For each, we'll compute the max triangle-free induced subgraph
    
    # Let's try random search on the full 18-vertex graph with local optimization
    
    def random_k4_free_graph(n, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        adj = [0] * n
        # Randomly add edges, checking K4 constraint
        edges_order = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(edges_order)
        
        for u, v in edges_order:
            # Check if adding (u,v) creates a K4
            # A K4 containing edge (u,v) needs two common neighbors w,x 
            # such that w-x is an edge
            common = []
            for w in range(n):
                if w != u and w != v:
                    if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                        common.append(w)
            
            creates_k4 = False
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    w, x = common[i], common[j]
                    if (adj[w] >> x) & 1:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def local_search(adj, n, max_iter=1000, deadline=None):
        # Try flipping edges to improve (reduce) max triangle-free induced subgraph
        best_adj = adj[:]
        best_score = _max_triangle_free_induced(best_adj, n)
        
        edges_order = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_order)
        
        for _ in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            improved = False
            for u, v in edges_order:
                if deadline and time.time() > deadline:
                    break
                
                has_edge = (adj[u] >> v) & 1
                
                # Try flipping this edge
                new_adj = adj[:]
                if has_edge:
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Check K4 constraint
                    common = []
                    for w in range(n):
                        if w != u and w != v:
                            if (new_adj[u] >> w) & 1 and (new_adj[v] >> w) & 1:
                                common.append(w)
                    
                    creates_k4 = False
                    for i in range(len(common)):
                        for j in range(i+1, len(common)):
                            w, x = common[i], common[j]
                            if (new_adj[w] >> x) & 1:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    
                    if creates_k4:
                        continue
                    
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                
                score = _max_triangle_free_induced(new_adj, n)
                if score < best_score:
                    best_score = score
                    best_adj = new_adj
                    adj = new_adj
                    improved = True
                    if best_score <= 4:
                        return best_adj, best_score
                    break
            
            if not improved:
                # Random restart within local search
                for _ in range(5):
                    u, v = random.choice(edges_order)
                    has_edge = (adj[u] >> v) & 1
                    new_adj = adj[:]
                    if has_edge:
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        common = []
                        for w in range(n):
                            if w != u and w != v:
                                if (new_adj[u] >> w) & 1 and (new_adj[v] >> w) & 1:
                                    common.append(w)
                        creates_k4 = False
                        for i in range(len(common)):
                            for j in range(i+1, len(common)):
                                w, x = common[i], common[j]
                                if (new_adj[w] >> x) & 1:
                                    creates_k4 = True
                                    break
                            if creates_k4:
                                break
                        if creates_k4:
                            continue
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                    
                    score = _max_triangle_free_induced(new_adj, n)
                    if score <= best_score:
                        best_score = score
                        best_adj = new_adj
                        adj = new_adj
                        if best_score <= 4:
                            return best_adj, best_score
                        break
        
        return best_adj, best_score
    
    best_adj = None
    best_score = n + 1
    
    # Try multiple random starts
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        adj = random_k4_free_graph(n, seed)
        score = _max_triangle_free_induced(adj, n)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            if best_score <= 4:
                break
        
        # Local search
        if time.time() > deadline - 10:
            break
        
        adj_ls, score_ls = local_search(adj, n, max_iter=50, deadline=deadline - 5)
        if score_ls < best_score:
            best_score = score_ls
            best_adj = adj_ls[:]
            if best_score <= 4:
                break
    
    if best_adj is None:
        # Fallback: complete 3-partite graph K6,6,6
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        best_adj = adj
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
