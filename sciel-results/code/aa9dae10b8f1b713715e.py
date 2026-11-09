"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # We'll work with a bipartite graph: U = {0..9}, V = {10..19}
    # M[i][j] = color of edge (U[i], V[j]), or -1 if no edge
    # 4-cycles in bipartite graph: (i1,j1), (i2,j1), (i2,j2), (i1,j2) for i1!=i2, j1!=j2
    
    # Precompute all 4-cycles
    # A 4-cycle is determined by two rows and two columns
    cycle_rows = []
    cycle_cols = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    cycle_rows.append((i1, i2))
                    cycle_cols.append((j1, j2))
    
    num_cycles = len(cycle_rows)
    
    # For each cycle, the 4 edges are:
    # (i1, j1), (i2, j1), (i2, j2), (i1, j2)
    # Colors: M[i1,j1], M[i2,j1], M[i2,j2], M[i1,j2]
    
    # Build arrays for vectorized checking
    # cycle_edge_positions: for each cycle, the 4 (row, col) positions
    # We'll store as separate arrays for the 4 edges
    ce1_r = np.array([cr[0] for cr in cycle_rows])
    ce1_c = np.array([cc[0] for cc in cycle_cols])
    ce2_r = np.array([cr[1] for cr in cycle_rows])
    ce2_c = np.array([cc[0] for cc in cycle_cols])
    ce3_r = np.array([cr[1] for cr in cycle_rows])
    ce3_c = np.array([cc[1] for cc in cycle_cols])
    ce4_r = np.array([cr[0] for cr in cycle_rows])
    ce4_c = np.array([cc[1] for cc in cycle_cols])
    
    def check_all_cycles(M):
        """Check all 4-cycles. Returns (num_bad, bad_mask)."""
        c1 = M[ce1_r, ce1_c]
        c2 = M[ce2_r, ce2_c]
        c3 = M[ce3_r, ce3_c]
        c4 = M[ce4_r, ce4_c]
        
        # All 4 edges must be present
        all_present = (c1 != -1) & (c2 != -1) & (c3 != -1) & (c4 != -1)
        
        # All 4 colors must be distinct
        distinct = (c1 != c2) & (c1 != c3) & (c1 != c4) & (c2 != c3) & (c2 != c4) & (c3 != c4)
        
        bad_mask = all_present & ~distinct
        return np.sum(bad_mask), bad_mask
    
    def count_edges(M):
        return np.sum(M != -1)
    
    def try_add_edge(M, i, j):
        """Try to add edge (i,j) with some color. Returns (success, new_M)."""
        for color in range(num_colors):
            M[i, j] = color
            bad, _ = check_all_cycles(M)
            if bad == 0:
                return True, M
        M[i, j] = -1
        return False, M
    
    # Phase 1: Greedy construction with multiple random orderings
    best_M = None
    best_edges = 0
    
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 20.0:
            break
        
        M = np.full((10, 10), -1, dtype=np.int32)
        
        # Random order of cells
        cells = [(i, j) for i in range(10) for j in range(10)]
        random.shuffle(cells)
        
        for (i, j) in cells:
            if time.time() > deadline - 20.0:
                break
            success, M = try_add_edge(M, i, j)
        
        edges = count_edges(M)
        bad, _ = check_all_cycles(M)
        
        if bad == 0 and edges > best_edges:
            best_edges = edges
            best_M = M.copy()
    
    # Phase 2: Local search - try to add more edges to best solution
    if best_M is not None:
        M = best_M.copy()
        current_edges = count_edges(M)
        
        # Try adding missing edges
        missing = [(i, j) for i in range(10) for j in range(10) if M[i, j] == -1]
        random.shuffle(missing)
        
        for (i, j) in missing:
            if time.time() > deadline - 10.0:
                break
            success, M = try_add_edge(M, i, j)
            if success:
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_M = M.copy()
    
    # Phase 3: More aggressive local search with recoloring
    if best_M is not None:
        M = best_M.copy()
        current_edges = count_edges(M)
        current_bad, current_bad_mask = check_all_cycles(M)
        
        # Local search: pick bad cycles, recolor edges to fix them
        for step in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            if current_bad == 0:
                break
            
            # Pick a random bad cycle
            bad_indices = np.where(current_bad_mask)[0]
            if len(bad_indices) == 0:
                break
            target = bad_indices[random.randint(0, len(bad_indices) - 1)]
            
            # Get the 4 edges in this cycle
            i1, i2 = cycle_rows[target]
            j1, j2 = cycle_cols[target]
            edges_in_cycle = [(i1, j1), (i2, j1), (i2, j2), (i1, j2)]
            
            # Try recoloring one of the edges
            r, c = edges_in_cycle[random.randint(0, 3)]
            old_color = M[r, c]
            
            improved = False
            for new_color in range(num_colors):
                if new_color == old_color:
                    continue
                M[r, c] = new_color
                new_bad, new_bad_mask = check_all_cycles(M)
                
                if new_bad < current_bad:
                    current_bad = new_bad
                    current_bad_mask = new_bad_mask
                    improved = True
                    break
                else:
                    M[r, c] = old_color
            
            if not improved:
                # Random perturbation
                rr = random.randint(0, 9)
                cc = random.randint(0, 9)
                M[rr, cc] = random.randint(0, num_colors - 1)
                current_bad, current_bad_mask = check_all_cycles(M)
        
        # Check if we found a valid solution
        final_bad, _ = check_all_cycles(M)
        if final_bad == 0:
            edges = count_edges(M)
            if edges > best_edges:
                best_edges = edges
                best_M = M.copy()
    
    # Phase 4: Try a completely different approach - start with full matrix and remove edges
    if best_edges < 60:
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            
            # Start with a random full coloring
            M = np.random.randint(0, num_colors, size=(10, 10), dtype=np.int32)
            
            current_bad, current_bad_mask = check_all_cycles(M)
            local_best = M.copy()
            local_best_edges = count_edges(M)
            
            # Iteratively remove edges from bad cycles
            for step in range(3000):
                if time.time() > deadline - 5.0:
                    break
                
                if current_bad == 0:
                    break
                
                # Pick a random bad cycle
                bad_indices = np.where(current_bad_mask)[0]
                if len(bad_indices) == 0:
                    break
                target = bad_indices[random.randint(0, len(bad_indices) - 1)]
                
                i1, i2 = cycle_rows[target]
                j1, j2 = cycle_cols[target]
                edges_in_cycle = [(i1, j1), (i2, j1), (i2, j2), (i1, j2)]
                
                # Remove one random edge from the cycle
                r, c = edges_in_cycle[random.randint(0, 3)]
                M[r, c] = -1
                
                new_bad, new_bad_mask = check_all_cycles(M)
                current_bad = new_bad
                current_bad_mask = new_bad_mask
                
                edges = count_edges(M)
                if edges > local_best_edges:
                    local_best_edges = edges
                    local_best = M.copy()
            
            # Check if local best is valid
            final_bad, _ = check_all_cycles(local_best)
            if final_bad == 0:
                edges = count_edges(local_best)
                if edges > best_edges:
                    best_edges = edges
                    best_M = local_best.copy()
    
    # Fallback: if we have no valid solution, return a star
    if best_M is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    # Convert M to edge list
    result = []
    for i in range(10):
        for j in range(10):
            if best_M[i, j] != -1:
                u = i
                v = 10 + j
                c = int(best_M[i, j])
                result.append([u, v, c])
    
    # Try to add intra-set edges
    # Edges within U: (i, k) for i < k, both in 0..9
    # Edges within V: (10+i, 10+k) for i < k, both in 10..19
    # These can form 4-cycles with bipartite edges
    # A 4-cycle with intra-U edges: u1-u2 (intra), u2-v1 (bip), v1-v2 (intra V), v2-u1 (bip)
    # This requires both an intra-U edge and an intra-V edge
    # For safety, let's try adding intra-U edges first, then intra-V edges
    
    # Build adjacency for the current graph
    adj = [set() for _ in range(n)]
    edge_colors = {}
    for (u, v, c) in result:
        adj[u].add(v)
        adj[v].add(u)
        edge_colors[(min(u,v), max(u,v))] = c
    
    def check_4cycle(u, v, w, x):
        """Check if u-v-w-x-u is a 4-cycle and if it's rainbow."""
        # Check all 4 edges exist
        edges = [(u,v), (v,w), (w,x), (x,u)]
        for (a, b) in edges:
            if b not in adj[a]:
                return False
            a, b = min(a,b), max(a,b)
        # Get colors
        colors = []
        for (a, b) in edges:
            a, b = min(a,b), max(a,b)
            colors.append(edge_colors[(a,b)])
        # Check rainbow
        return len(set(colors)) == 4
    
    def can_add_edge(u, v, color):
        """Check if adding edge (u,v) with given color keeps all 4-cycles rainbow."""
        # Find all 4-cycles that include edge (u,v)
        # A 4-cycle containing (u,v) has form u-v-w-x-u or u-v-x-w-u
        # We need to check all such cycles where all other edges exist
        
        # For each pair (w, x) where w != u,v and x != u,v,w:
        # Check if u-v-w-x-u is a cycle (all edges exist except possibly u-v)
        # and if adding u-v with 'color' makes it non-rainbow
        
        for w in range(n):
            if w == u or w == v:
                continue
            if w not in adj[v]:
                continue
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if x not in adj[u]:
                    continue
                if x not in adj[w]:
                    continue
                
                # Check if u-v-w-x-u is a potential 4-cycle
                # Edges: u-v (new), v-w, w-x, x-u
                # v-w, w-x, x-u must exist
                if w not in adj[v] or x not in adj[w] or u not in adj[x]:
                    continue
                
                # Get colors of existing edges
                c_vw = edge_colors[(min(v,w), max(v,w))]
                c_wx = edge_colors[(min(w,x), max(w,x))]
                c_xu = edge_colors[(min(x,u), max(x,u))]
                
                # The new edge u-v has color 'color'
                # All 4 colors must be distinct
                colors = [color, c_vw, c_wx, c_xu]
                if len(set(colors)) < 4:
                    return False
        
        return True
    
    # Try adding intra-U edges
    for i in range(10):
        for k in range(i + 1, 10):
            if time.time() > deadline - 2.0:
                break
            u, v = i, k
            if v in adj[u]:
                continue
            for color in range(num_colors):
                if can_add_edge(u, v, color):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_colors[(min(u,v), max(u,v))] = color
                    result.append([u, v, color])
                    break
    
    # Try adding intra-V edges
    for i in range(10):
        for k in range(i + 1, 10):
            if time.time() > deadline - 2.0:
                break
            u, v = 10 + i, 10 + k
            if v in adj[u]:
                continue
            for color in range(num_colors):
                if can_add_edge(u, v, color):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_colors[(min(u,v), max(u,v))] = color
                    result.append([u, v, color])
                    break
    
    return result

# EVOLVE-BLOCK-END
