import time
import random
import copy

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    def bitmask_from_adj(adj):
        """Convert adjacency matrix to list of bitmasks."""
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
        return masks
    
    def adj_from_mask(masks):
        """Convert bitmask list back to adjacency matrix."""
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if (masks[i] >> j) & 1:
                    adj[i][j] = 1
        return adj
    
    def compute_degrees(masks):
        return [bin(m).count('1') for m in masks]
    
    def get_score(masks):
        """Calculate max degree sum over triangles using bitmasks."""
        deg = compute_degrees(masks)
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] >> j) & 1:
                    continue
                # Common neighbors of i and j
                common = masks[i] & masks[j]
                # Only consider k > j to avoid duplicates
                common &= ~((1 << (j+1)) - 1)  # mask out 0..j
                while common:
                    k = common & (-common)  # lowest set bit
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common ^= k
        return max_sum
    
    def get_worst_triangle(masks):
        """Return (score, triangle) where triangle is (i,j,k)."""
        deg = compute_degrees(masks)
        max_sum = 0
        worst_tri = None
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] >> j) & 1:
                    continue
                common = masks[i] & masks[j]
                common &= ~((1 << (j+1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                        worst_tri = (i, j, k_idx)
                    common ^= k
        return max_sum, worst_tri
    
    def make_initial_adj():
        """Create K_8,8 + one internal edge."""
        adj = [[0]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = 1
                adj[v][u] = 1
        adj[0][1] = 1
        adj[1][0] = 1
        return adj
    
    def make_perturbed_adj(base_adj, num_perturbations=3):
        """Perturb base adj by random swaps."""
        adj = [row[:] for row in base_adj]
        for _ in range(num_perturbations):
            # Pick random edge to remove
            edges_list = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i][j]]
            if not edges_list:
                break
            u, v = random.choice(edges_list)
            # Pick random non-edge to add
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not adj[x][y]:
                    break
            adj[u][v] = 0
            adj[v][u] = 0
            adj[x][y] = 1
            adj[y][x] = 1
        return adj
    
    # Multi-start with diverse initial configurations
    best_score = float('inf')
    best_masks = None
    
    base_adj = make_initial_adj()
    base_masks = bitmask_from_adj(base_adj)
    base_score = get_score(base_masks)
    best_score = base_score
    best_masks = base_masks[:]
    
    # Number of restarts based on time
    num_restarts = 5
    restart_deadline = deadline - 15  # Leave 15s for final polish
    
    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break
        
        if r == 0:
            masks = base_masks[:]
        else:
            adj = make_perturbed_adj(base_adj, random.randint(2, 6))
            masks = bitmask_from_adj(adj)
        
        current_score = get_score(masks)
        T = 1.5
        cooling = 0.9998
        
        while time.time() < restart_deadline:
            # Targeted move: focus on high-degree vertices in worst triangle
            _, worst_tri = get_worst_triangle(masks)
            deg = compute_degrees(masks)
            
            if worst_tri:
                i, j, k = worst_tri
                # With 70% probability, target an edge incident to a worst-triangle vertex
                if random.random() < 0.7:
                    target_vertices = [i, j, k]
                    tv = random.choice(target_vertices)
                    # Find edges incident to tv
                    incident = []
                    for v in range(n):
                        if v != tv and (masks[tv] >> v) & 1:
                            incident.append((min(tv,v), max(tv,v)))
                    if incident:
                        u, v = random.choice(incident)
                    else:
                        u, v = random.choice([(a,b) for a in range(n) for b in range(a+1,n) if (masks[a]>>b)&1])
                else:
                    # Random edge
                    all_edges = [(a,b) for a in range(n) for b in range(a+1,n) if (masks[a]>>b)&1]
                    if not all_edges:
                        break
                    u, v = random.choice(all_edges)
            else:
                all_edges = [(a,b) for a in range(n) for b in range(a+1,n) if (masks[a]>>b)&1]
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
            
            # Pick random non-edge to add
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not (masks[x] >> y) & 1:
                    break
            
            if (x == u and y == v) or (x == v and y == u):
                continue
            
            # Perform swap
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            masks[x] |= (1 << y)
            masks[y] |= (1 << x)
            
            new_score = get_score(masks)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_masks = masks[:]
            elif new_score == current_score:
                if random.random() < 0.4:
                    current_score = new_score
                else:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
            else:
                delta = new_score - current_score
                if random.random() < pow(2.71828, -delta / T):
                    current_score = new_score
                else:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
            
            T *= cooling
            if T < 0.05:
                T = 0.05
    
    # Final polish: deterministic local search on best solution
    if best_masks is not None:
        polish_masks = best_masks[:]
        polish_score = get_score(polish_masks)
        polish_deadline = deadline - 5
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            # Try all single-edge swaps
            all_edges = [(a,b) for a in range(n) for b in range(a+1,n) if (polish_masks[a]>>b)&1]
            all_nonedges = [(a,b) for a in range(n) for b in range(a+1,n) if not (polish_masks[a]>>b)&1]
            
            for u, v in all_edges:
                if time.time() > polish_deadline:
                    break
                for x, y in all_nonedges:
                    if (x == u and y == v) or (x == v and y == u):
                        continue
                    # Try swap
                    polish_masks[u] &= ~(1 << v)
                    polish_masks[v] &= ~(1 << u)
                    polish_masks[x] |= (1 << y)
                    polish_masks[y] |= (1 << x)
                    
                    new_score = get_score(polish_masks)
                    
                    if new_score < polish_score:
                        polish_score = new_score
                        if polish_score < best_score:
                            best_score = polish_score
                            best_masks = polish_masks[:]
                        improved = True
                    else:
                        # Rollback
                        polish_masks[u] |= (1 << v)
                        polish_masks[v] |= (1 << u)
                        polish_masks[x] &= ~(1 << y)
                        polish_masks[y] &= ~(1 << x)
    
    # Reconstruct edge list from best_masks
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_masks[i] >> j) & 1:
                final_edges.append([i, j])
    
    return {'n': n, 'edges': final_edges}
