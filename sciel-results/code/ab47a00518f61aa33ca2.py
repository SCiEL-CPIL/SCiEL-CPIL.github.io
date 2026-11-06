import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    rng = random.Random(42)
    
    # Represent adjacency as list of bitmasks for speed
    def make_adj():
        return [0] * n
    
    def get_degrees(adj):
        return [bin(x).count('1') for x in adj]
    
    def build_triangles(adj):
        """Build list of triangles as (u,v,w) tuples with u<v<w."""
        tris = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    for w in range(v + 1, n):
                        if (adj[v] & (1 << w)) and (adj[w] & (1 << u)):
                            tris.append((u, v, w))
        return tris
    
    def calc_score(adj, tris, degs):
        if not tris:
            return 0
        return max(degs[u] + degs[v] + degs[w] for u, v, w in tris)
    
    def apply_swap(adj, degs, tris, e_rm, e_add):
        """
        Apply edge swap: remove e_rm=(u,v), add e_add=(x,y).
        Returns (new_degs, new_tris).
        """
        u, v = e_rm
        x, y = e_add
        
        # Update degrees
        degs[u] -= 1
        degs[v] -= 1
        degs[x] += 1
        degs[y] += 1
        
        # Update adjacency
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        # Find affected triangles: those containing (u,v) or (x,y)
        affected = set()
        for t in tris:
            a, b, c = t
            if (a == u and b == v) or (a == v and b == u) or \
               (a == u and c == v) or (a == v and c == u) or \
               (b == u and c == v) or (b == v and c == u):
                affected.add(t)
            if (a == x and b == y) or (a == y and b == x) or \
               (a == x and c == y) or (a == y and c == x) or \
               (b == x and c == y) or (b == y and c == x):
                affected.add(t)
        
        new_tris = [t for t in tris if t not in affected]
        
        # Check for newly formed triangles containing (x,y)
        for w in range(n):
            if w == x or w == y:
                continue
            if (adj[x] & (1 << w)) and (adj[y] & (1 << w)):
                # Ensure x < y < w or proper ordering
                verts = sorted([x, y, w])
                new_tri = tuple(verts)
                if new_tri not in affected:  # Not already handled
                    new_tris.append(new_tri)
        
        return degs, new_tris
    
    def undo_swap(adj, degs, e_rm, e_add, old_degs, old_tris):
        """Undo a swap."""
        u, v = e_rm
        x, y = e_add
        
        degs[u] = old_degs[u]
        degs[v] = old_degs[v]
        degs[x] = old_degs[x]
        degs[y] = old_degs[y]
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj[x] &= ~(1 << y)
        adj[y] &= ~(1 << x)
        
        return degs, old_tris
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask -= lsb
        return edges
    
    def make_initial_config(variant=0):
        """Create initial K_{8,8} + 1 edge configuration."""
        adj = make_adj()
        # 0-7 in A, 8-15 in B
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Add one internal edge based on variant
        if variant == 0:
            e = (0, 1)
        elif variant == 1:
            e = (0, 7)
        elif variant == 2:
            e = (3, 5)
        else:
            e = (2, 6)
        
        u, v = e
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        degs = get_degrees(adj)
        tris = build_triangles(adj)
        return adj, degs, tris
    
    def hill_climb(adj, degs, tris, time_budget, start_temp=2.0, end_temp=0.05, seed=None):
        """Simulated annealing with incremental triangle updates."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng
        
        cur_score = calc_score(adj, tris, degs)
        best_score = cur_score
        best_adj = adj[:]
        best_degs = degs[:]
        best_tris = tris[:]
        
        edges = get_edges_from_adj(adj)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        nonedges = [p for p in all_pairs if p not in set(edges)]
        
        max_iters = 300000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        iters = 0
        stall_count = 0
        stall_threshold = 15000
        
        while iters < max_iters:
            if time.time() - start_time > deadline:
                break
            
            if iters % 5000 == 0:
                # Verify edge count
                ec = sum(bin(x).count('1') for x in adj) // 2
                if ec != target_edges:
                    break
            
            if not edges or not nonedges:
                break
            
            # Pick random edge to remove and non-edge to add
            e_rm = edges[local_rng.randrange(len(edges))]
            e_add = nonedges[local_rng.randrange(len(nonedges))]
            
            # Skip if same edge
            if e_rm == e_add:
                iters += 1
                temp *= cooling_rate
                continue
            
            old_degs = degs[:]
            old_tris = tris[:]
            
            new_degs, new_tris = apply_swap(adj, degs, tris, e_rm, e_add)
            new_score = calc_score(adj, new_tris, new_degs)
            
            # Acceptance
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = local_rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > local_rng.random()
            
            if accept:
                degs, tris = new_degs, new_tris
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_tris = tris[:]
                    stall_count = 0
                else:
                    stall_count += 1
                
                # Update edge lists
                edges.remove(e_rm)
                edges.append(e_add)
                nonedges.remove(e_add)
                nonedges.append(e_rm)
            else:
                # Undo
                adj, degs, tris = undo_swap(adj, degs, e_rm, e_add, old_degs, old_tris)
            
            iters += 1
            temp *= cooling_rate
        
        return best_adj, best_degs, best_tris, best_score
    
    # Run multiple restarts with different initial configurations
    best_overall_adj = None
    best_overall_score = float('inf')
    
    for variant in range(4):
        if time.time() - start_time > deadline:
            break
        
        adj, degs, tris = make_initial_config(variant)
        
        # Give each restart a portion of the time budget
        time_per_restart = (deadline - (time.time() - start_time)) / max(1, (4 - variant))
        
        best_adj, best_degs, best_tris, best_score = hill_climb(
            adj, degs, tris, time_per_restart, 
            start_temp=2.0, end_temp=0.03, 
            seed=1000 + variant
        )
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_adj = best_adj[:]
    
    # Final polish: run one more short SA on the best found
    if time.time() - start_time < deadline - 10:
        degs = get_degrees(best_overall_adj)
        tris = build_triangles(best_overall_adj)
        polished_adj, _, _, polished_score = hill_climb(
            best_overall_adj, degs, tris, 
            deadline - (time.time() - start_time) - 5,
            start_temp=0.5, end_temp=0.01,
            seed=9999
        )
        if polished_score < best_overall_score:
            best_overall_adj = polished_adj
            best_overall_score = polished_score
    
    # Convert to edge list
    edges = []
    for u in range(n):
        mask = best_overall_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v > u:
                edges.append([u, v])
            mask -= lsb
    
    return {'n': 16, 'edges': edges}
